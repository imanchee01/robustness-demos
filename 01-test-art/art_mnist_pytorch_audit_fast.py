# art_mnist_pytorch_audit_fast.py
# PyTorch + ART MNIST audit (fast CPU version)
# - trains a small CNN (3 epochs)
# - evasion: FGSM (full set), PGD & AutoPGD on 1,000 samples, DeepFool (1k), Square (500), HSJ (200)
# - backdoor poisoning demo + ActivationDefence
# - preprocessing defences (JPEG, Spatial Smoothing)
# - outputs: audit_out/audit_report.json + robustness_curve.png
# - ALSO exports Robuscope-ready predictions to audit_out/robuscope_predictions.json

import os, time, json, random, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod, ProjectedGradientDescent, AutoProjectedGradientDescent,
    DeepFool, SquareAttack, HopSkipJump
)
from art.attacks.poisoning.perturbations.image_perturbations import add_pattern_bd, add_single_bd
from art.defences.detector.poison import ActivationDefence
from art.utils import load_mnist, to_categorical

# Preprocessing defences
try:
    from art.defences.preprocessor import JpegCompression, SpatialSmoothing
except ImportError:
    from art.defences.preprocessor.jpeg_compression import JpegCompression
    from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

# ---------- knobs to keep runtime low ----------
SEED = 0
TRAIN_N = 10000          # train subset
EPOCHS = 3               # quick train
APGD_EVAL_N = 1000       # eval subset for AutoPGD and PGD (white-box heavy)
DF_EVAL_N = 1000         # DeepFool subset
SQ_EVAL_N = 500          # Square attack subset
HSJ_EVAL_N = 200         # HopSkipJump subset (decision-based)
PGD_ITERS = 20
APGD_ITERS = 50
ATTACK_BS = 256          # larger attack batch if RAM allows
# ----------------------------------------------

np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64*12*12, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        # 🔧 ensure input dtype matches model weights (float32)
        x = x.to(next(self.parameters()).dtype)   # or: x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def load_mnist_arrays(n_train=TRAIN_N):
    # Use ART's loader (avoids torchvision SSL hiccups)
    (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw), vmin, vmax = load_mnist(raw=True)
    idx = np.random.permutation(len(x_tr_raw))[:n_train]
    x_tr_raw = x_tr_raw[idx]; y_tr_raw = y_tr_raw[idx]
    # Normalize to [0,1], add channel dim (N,1,28,28)
    xtr = (x_tr_raw.astype("float32")/255.0)[:, None, :, :]
    xte = (x_te_raw.astype("float32")/255.0)[:, None, :, :]
    ytr = to_categorical(y_tr_raw.astype(int), 10).astype("float32")
    yte = to_categorical(y_te_raw.astype(int), 10).astype("float32")
    return (xtr, ytr), (xte, yte), 0.0, 1.0

def generate_backdoor(x_clean, y_clean_idx, percent_poison,
                      backdoor_type="pattern",
                      sources=np.arange(10), targets=(np.arange(10)+1)%10):
    x2d = x_clean.squeeze(1)  # (N,28,28)
    max_val = float(x2d.max())
    x_p = np.copy(x2d); y_p = np.copy(y_clean_idx)
    is_p = np.zeros_like(y_p, dtype=np.float32)
    for src, tgt in zip(sources, targets):
        n_tgt = int(np.sum(y_clean_idx == tgt))
        num_poison = int(round((percent_poison * n_tgt)/(1.0 - percent_poison))) if n_tgt>0 else 0
        src_imgs = x2d[y_clean_idx == src]
        if len(src_imgs)==0 or num_poison==0: continue
        sel = np.random.choice(len(src_imgs), num_poison, replace=len(src_imgs)<num_poison)
        imgs = np.copy(src_imgs[sel])
        if backdoor_type=="pattern": imgs = add_pattern_bd(x=imgs, pixel_value=max_val)
        elif backdoor_type=="pixel": imgs = add_single_bd(imgs, pixel_value=max_val)
        x_p = np.append(x_p, imgs, axis=0)
        y_p = np.append(y_p, np.full(num_poison, tgt, dtype=y_p.dtype), axis=0)
        is_p = np.append(is_p, np.ones(num_poison, dtype=np.float32), axis=0)
    return (is_p!=0), x_p, y_p

def accuracy(clf, x, y_oh):
    preds = clf.predict(x)
    return float((np.argmax(preds,1)==np.argmax(y_oh,1)).mean())

def run_attack(clf, atk, x, y, name, eps=None):
    t0 = time.time()
    x_adv = atk.generate(x=x)
    ra = accuracy(clf, x_adv, y)
    return {"attack": name, "eps": eps, "robust_acc": ra, "asr": 1.0-ra,
            "secs": time.time()-t0, "n": int(len(x))}

# --- Robuscope export helpers --------------------------------------------------
def _predict_in_batches(clf, x, batch_size=512):
    """Return softmax probabilities for x as a numpy array (N, C)."""
    probs = []
    for i in range(0, len(x), batch_size):
        probs.append(clf.predict(x[i:i+batch_size]))
    return np.concatenate(probs, axis=0)

def _dataset_entries_from_probs(probs, y_idx):
    """Return list[{'prediction': [...], 'label': int}]"""
    return [{"prediction": p.astype(float).tolist(), "label": int(y)}
            for p, y in zip(probs, y_idx)]

def _make_adv_set(clf, name, atk, x_src, y_src, y_idx_src, eps=None):
    x_adv = atk.generate(x=x_src)
    probs = _predict_in_batches(clf, x_adv)
    entries = _dataset_entries_from_probs(probs, y_idx_src)
    return (name if eps is None else f"{name}_eps{eps}"), entries

def export_for_robuscope(clf, x_test, y_test, is_p_te, out_path="audit_out/robuscope_predictions.json"):
    """
    Build a Robuscope-compatible JSON with:
      - MNIST_clean (non-poisoned test points)
      - FGSM/PGD/APGD for eps in {0.05,0.1,0.2,0.3}
      - DeepFool (1k), Square(500, eps=0.3), HopSkipJump(200)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # choose subsets
    clean_mask = ~is_p_te
    x_clean = x_test[clean_mask]
    y_clean_oh = y_test[clean_mask]
    y_clean_idx = np.argmax(y_clean_oh, axis=1)

    # (optional) poisoned subset include toggle
    INCLUDE_POISONED = False
    x_poison = x_test[~clean_mask]
    y_poison_idx = np.argmax(y_test[~clean_mask], axis=1)

    ds = {}

    # Clean (in-distribution)
    probs_clean = _predict_in_batches(clf, x_clean)
    ds["MNIST_clean"] = _dataset_entries_from_probs(probs_clean, y_clean_idx)

    if INCLUDE_POISONED and len(x_poison) > 0:
        probs_poison = _predict_in_batches(clf, x_poison)
        ds["MNIST_poisoned"] = _dataset_entries_from_probs(probs_poison, y_poison_idx)

    # White-box over full/partial clean subset
    eps_grid = [0.05, 0.1, 0.2, 0.3]
    for eps in eps_grid:
        fgsm = FastGradientMethod(estimator=clf, eps=eps, batch_size=ATTACK_BS)
        name, entries = _make_adv_set(clf, "FGSM_Linf", fgsm, x_clean, y_clean_oh, y_clean_idx, eps)
        ds[name] = entries

        x_pgd = x_clean[:APGD_EVAL_N]
        y_pgd_oh = y_clean_oh[:APGD_EVAL_N]
        y_pgd_idx = y_clean_idx[:APGD_EVAL_N]

        pgd = ProjectedGradientDescent(estimator=clf, eps=eps, max_iter=PGD_ITERS,
                                       eps_step=eps/8, batch_size=ATTACK_BS)
        name, entries = _make_adv_set(clf, "PGD_Linf_1k", pgd, x_pgd, y_pgd_oh, y_pgd_idx, eps)
        ds[name] = entries

        apgd = AutoProjectedGradientDescent(estimator=clf, eps=eps, max_iter=APGD_ITERS,
                                            nb_random_init=1, batch_size=ATTACK_BS)
        name, entries = _make_adv_set(clf, "AutoPGD_Linf_1k", apgd, x_pgd, y_pgd_oh, y_pgd_idx, eps)
        ds[name] = entries

    # DeepFool on 1k
    x_df = x_clean[:DF_EVAL_N]; y_df_oh = y_clean_oh[:DF_EVAL_N]; y_df_idx = y_clean_idx[:DF_EVAL_N]
    df = DeepFool(clf, max_iter=50, batch_size=ATTACK_BS)
    name, entries = _make_adv_set(clf, "DeepFool_L2_1k", df, x_df, y_df_oh, y_df_idx)
    ds[name] = entries

    # Square on 500 with eps=0.3
    x_sq = x_clean[:SQ_EVAL_N]; y_sq_oh = y_clean_oh[:SQ_EVAL_N]; y_sq_idx = y_clean_idx[:SQ_EVAL_N]
    sq = SquareAttack(estimator=clf, eps=0.3, max_iter=200, batch_size=ATTACK_BS)
    name, entries = _make_adv_set(clf, "Square_Linf_500", sq, x_sq, y_sq_oh, y_sq_idx, 0.3)
    ds[name] = entries

    # HSJ on 200 (decision-based)
    x_hsj = x_clean[:HSJ_EVAL_N]; y_hsj_oh = y_clean_oh[:HSJ_EVAL_N]; y_hsj_idx = y_clean_idx[:HSJ_EVAL_N]
    hsj = HopSkipJump(classifier=clf, max_iter=10, init_eval=10, max_eval=100)
    name, entries = _make_adv_set(clf, "HopSkipJump_200", hsj, x_hsj, y_hsj_oh, y_hsj_idx)
    ds[name] = entries

    # meta info
    meta = {
        "class_names_model": [str(i) for i in range(10)],
        "dataset_type": {}
    }
    meta["dataset_type"]["MNIST_clean"] = "in-distribution"
    if "MNIST_poisoned" in ds:
        meta["dataset_type"]["MNIST_poisoned"] = "corrupted"
    for k in list(ds.keys()):
        if k.startswith(("FGSM_Linf", "PGD_Linf", "AutoPGD_Linf", "DeepFool", "Square_Linf", "HopSkipJump")):
            meta["dataset_type"][k] = "corrupted"

    robuscope_obj = {**ds, "meta_information": meta}
    with open(out_path, "w") as f:
        json.dump(robuscope_obj, f)
    print(f"Saved Robuscope file: {out_path} (datasets={len(ds)})")
# -------------------------------------------------------------------------------

def main():
    os.makedirs("audit_out", exist_ok=True)
    report = {"seed": SEED, "versions": {}}

    # Data
    (xtr, ytr), (xte, yte), vmin, vmax = load_mnist_arrays()
    ytr_idx = np.argmax(ytr,1); yte_idx = np.argmax(yte,1)

    # Poison (backdoor) demo: poison both train & test copies
    perc_poison = 0.33
    is_p_tr, x_tr_p_raw, y_tr_p_idx = generate_backdoor(xtr, ytr_idx, perc_poison, "pattern")
    is_p_te, x_te_p_raw, y_te_p_idx = generate_backdoor(xte, yte_idx, perc_poison, "pattern")
    x_train = x_tr_p_raw[:, None, :, :].astype("float32")
    y_train = to_categorical(y_tr_p_idx, 10).astype("float32")
    x_test  = x_te_p_raw[:, None, :, :].astype("float32")
    y_test  = to_categorical(y_te_p_idx, 10).astype("float32")

    # Shuffle poisoned train
    sh = np.random.permutation(len(y_train))
    x_train, y_train, is_p_tr = x_train[sh], y_train[sh], is_p_tr[sh]

    # Model + ART wrapper
    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    classifier = PyTorchClassifier(
        model=model, loss=criterion, optimizer=optimizer,
        input_shape=(1,28,28), nb_classes=10, clip_values=(vmin, vmax)
    )

    # Train (quick)
    classifier.fit(x_train, y_train, batch_size=128, nb_epochs=EPOCHS)

    # Baselines
    clean_subset = ~is_p_te
    report["clean_acc_all_test"]   = accuracy(classifier, x_test, y_test)
    report["clean_acc_clean_only"] = accuracy(classifier, x_test[clean_subset], y_test[clean_subset])
    report["poison_subset_acc"]    = accuracy(classifier, x_test[~clean_subset], y_test[~clean_subset])

    # Evasion on clean subset
    results = []
    x_clean = x_test[clean_subset]; y_clean = y_test[clean_subset]

    # uniform noise baseline
    def noise_acc(eps):
        xn = np.clip(x_clean + np.random.uniform(-eps, eps, x_clean.shape), vmin, vmax).astype("float32")
        return accuracy(classifier, xn, y_clean)

    eps_grid = [0.05, 0.1, 0.2, 0.3]
    for eps in eps_grid:
        results.append({"attack":"UniformNoise", "eps":eps, "robust_acc": noise_acc(eps)})

        fgsm = FastGradientMethod(estimator=classifier, eps=eps, batch_size=ATTACK_BS)
        results.append(run_attack(classifier, fgsm, x_clean, y_clean, "FGSM_Linf", eps))

        # PGD / APGD on subset with fewer iters and large batch
        x_pgd, y_pgd = x_clean[:APGD_EVAL_N], y_clean[:APGD_EVAL_N]
        pgd = ProjectedGradientDescent(
            estimator=classifier, eps=eps, max_iter=PGD_ITERS, eps_step=eps/8, batch_size=ATTACK_BS
        )
        results.append(run_attack(classifier, pgd, x_pgd, y_pgd, "PGD_Linf", eps))

        apgd = AutoProjectedGradientDescent(
            estimator=classifier, eps=eps, max_iter=APGD_ITERS, nb_random_init=1, batch_size=ATTACK_BS
        )
        results.append(run_attack(classifier, apgd, x_pgd, y_pgd, "AutoPGD_Linf", eps))

    # DeepFool on 1k
    df = DeepFool(classifier, max_iter=50, batch_size=ATTACK_BS)
    results.append(run_attack(classifier, df, x_clean[:DF_EVAL_N], y_clean[:DF_EVAL_N], "DeepFool_L2"))

    # Black-box: Square / HSJ on small slices
    sq = SquareAttack(estimator=classifier, eps=0.3, max_iter=200, batch_size=ATTACK_BS)
    results.append(run_attack(classifier, sq, x_clean[:SQ_EVAL_N], y_clean[:SQ_EVAL_N], "Square_Linf", 0.3))
    hsj = HopSkipJump(classifier=classifier, max_iter=10, init_eval=10, max_eval=100)
    results.append(run_attack(classifier, hsj, x_clean[:HSJ_EVAL_N], y_clean[:HSJ_EVAL_N], "HopSkipJump"))

    report["evasion"] = results

    # Preprocessing defences (trade-off on clean inputs)
    jpeg = JpegCompression(clip_values=(0.0, 1.0), quality=50, channels_first=True)
    smooth = SpatialSmoothing(window_size=3)
    x_jpeg, _ = jpeg(x_clean); x_smooth, _ = smooth(x_clean)
    report["preproc_defences"] = {
        "jpeg_q50_clean_acc": accuracy(classifier, x_jpeg, y_clean),
        "spatial_smooth_w3_clean_acc": accuracy(classifier, x_smooth, y_clean),
    }

    # Poison detector (ActivationDefence)
    defence = ActivationDefence(classifier, x_train, y_train)
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", cluster_analysis="smaller")
    cm_size = json.loads(defence.evaluate_defence((~is_p_tr).astype(bool)))
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", cluster_analysis="distance")
    cm_dist = json.loads(defence.evaluate_defence((~is_p_tr).astype(bool)))
    report["poison_defence"] = {"activation_defence_size": cm_size,
                                "activation_defence_distance": cm_dist}

    # Save report
    os.makedirs("audit_out", exist_ok=True)
    report["versions"] = {"python": sys.version.split()[0], "torch": torch.__version__, "art": __import__("art").__version__}
    with open("audit_out/audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved: audit_out/audit_report.json")

    # Plot robustness curve
    import matplotlib.pyplot as plt
    def series(tag):
        pts = [(r["eps"], r["robust_acc"]) for r in results if r["attack"]==tag and r.get("eps") is not None]
        return sorted(pts, key=lambda t: t[0])
    fg, pg, ap, nz = series("FGSM_Linf"), series("PGD_Linf"), series("AutoPGD_Linf"), series("UniformNoise")
    plt.figure()
    if nz: plt.plot([e for e,_ in nz],[a for _,a in nz], marker="o", label="Uniform noise")
    if fg: plt.plot([e for e,_ in fg],[a for _,a in fg], marker="o", label="FGSM")
    if pg: plt.plot([e for e,_ in pg],[a for _,a in pg], marker="o", label="PGD (1k)")
    if ap: plt.plot([e for e,_ in ap],[a for _,a in ap], marker="o", label="Auto-PGD (1k)")
    plt.xlabel("ε (L∞)"); plt.ylabel("Robust accuracy")
    plt.title(f"MNIST (clean subset) • clean acc={report['clean_acc_clean_only']:.3f}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("audit_out/robustness_curve.png")
    print("Saved: audit_out/robustness_curve.png")

    # === Robuscope export ===
    export_for_robuscope(classifier, x_test, y_test, is_p_te)

    print("Done ✓")

if __name__ == "__main__":
    main()
