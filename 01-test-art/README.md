# 🧠 Fast MNIST Robustness Audit (PyTorch + ART)

Lightweight demo for evaluating adversarial and poisoning robustness using the **Adversarial Robustness Toolbox (ART)** and **PyTorch**.

This project trains a small CNN on MNIST, runs multiple adversarial attacks and defences, and exports ready-to-visualize results for [Robuscope](https://robuscope.ai2.io/).

---

## 🐍 Requirements

Python **3.13**

Install dependencies:

```bash
pip install torch adversarial-robustness-toolbox matplotlib numpy
