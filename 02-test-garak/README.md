# Garak Model Probe Demo

A simple script to run [garak](https://github.com/leondz/garak) safety probes against a Hugging Face model.

This repo provides a lightweight way to test models like `DialoGPT` or `OLMo` using garak’s built-in probe sets.

---

## 🧰 Requirements

- Python **3.13**
- Packages:
  - `garak`
  - `transformers`
  - `huggingface_hub`
  - `safetensors`
  - `torch`
  - `ai2-olmo` (replaces old `hf_olmo` package)

Install everything with:

```bash
pip install garak transformers huggingface_hub safetensors torch ai2-olmo
Run the included script:
python run_garak.py
This will:
Load the target model (default: microsoft/DialoGPT-small).
Run a small demo probe set (promptinject,dan.Dan_11_0).
Save reports with a timestamped prefix like:
olmo_demo_20251112_101234.session.json
olmo_demo_20251112_101234.probes.csv
olmo_demo_20251112_101234.findings.csv

