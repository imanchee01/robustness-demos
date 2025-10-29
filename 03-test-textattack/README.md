textattack tutorial
https://textattack.readthedocs.io/en/master/2notebook/0_End_to_End.html

This tutorial provides a broad end-to-end overview of training, evaluating, and attacking a model using TextAttack.
Terminal in PyCharm:

# activate your project venv
source .venv/bin/activate

# install TextAttack (TF extra is fine; you can also use [torch])
pip install "textattack[tensorflow]"

# pin transformers to a version TextAttack expects (fixes AdamW)
pip install "transformers==4.49.0" --upgrade --force-reinstall

# (one-time) download NLTK data needed by TextFooler’s POS constraint
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt')"

textattack peek-dataset --dataset-from-huggingface rotten_tomatoes

python -m textattack train \
  --model-name-or-path distilbert-base-uncased \
  --dataset rotten_tomatoes \
  --model-num-labels 2 \
  --model-max-length 64 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 16 \
  --num-epochs 3

python -m textattack eval \
  --num-examples 50 \
  --model ./outputs/2025-10-29-10-18-33-101509/best_model/ \
  --dataset-from-huggingface rotten_tomatoes \
  --dataset-split test

 python -m textattack attack \
  --recipe textfooler \
  --num-examples 20 \
  --model ./outputs/2025-10-29-10-18-33-101509/best_model/ \
  --dataset-from-huggingface rotten_tomatoes \
  --dataset-split test
