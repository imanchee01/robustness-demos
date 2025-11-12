# 🔍 Model Auditing & Robustness Demos

This repository provides simple, reproducible scripts for evaluating and probing models using:

- [**Garak**](https://github.com/leondz/garak) — automated safety & red-teaming probes for language models  
- [**Adversarial Robustness Toolbox (ART)**](https://github.com/Trusted-AI/adversarial-robustness-toolbox) — adversarial and poisoning robustness auditing for ML models  
- [**Robuscope**](https://robuscope.ai2.io/) — visualization and reporting of robustness evaluation outputs

All scripts are lightweight, CPU-friendly, and ready to run on Python **3.13**.

---

## 🧩 Repository Contents

| File | Description |
|------|--------------|
| `run_garak.py` | Runs Garak probes against a Hugging Face model and saves a safety report. |
| `art_mnist_pytorch_audit_fast.py` | Trains a small CNN on MNIST, runs multiple adversarial attacks and defences using ART, and exports a full audit report + Robuscope-ready JSON. |
| `convert_logits_to_probs.py` | Converts raw logits in Robuscope JSON outputs to proper probability distributions (softmaxed). |

---

## 🐍 Requirements

Python **3.13**

Install dependencies with:

```bash
pip install garak transformers huggingface_hub safetensors torch ai2-olmo adversarial-robustness-toolbox matplotlib
