# run_garak.py
import datetime
import subprocess
import sys

# MODEL_ID = "allenai/OLMo-2-0425-1B"  # Updated to OLMo-2 model
MODEL_ID = "microsoft/DialoGPT-small"  # Much smaller model for testing

PROBES = "promptinject,dan.Dan_11_0"  # small demo set; expand later
REPORT_PREFIX = f"olmo_demo_{datetime.datetime.now():%Y%m%d_%H%M%S}"

def main():
    cmd = [
        sys.executable, "-m", "garak",
        "--target_type", "huggingface",
        "--target_name", MODEL_ID,
        "--probes", PROBES,
        "--report_prefix", REPORT_PREFIX,
        "--verbose"
    ]
    print("Running garak:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. Reports prefixed with:", REPORT_PREFIX)

if __name__ == "__main__":
    main()
