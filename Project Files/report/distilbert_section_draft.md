# DistilBERT section (DRAFT, Interim Part B, Section 3 and notes) - owner: Rudra Rajpurohit

## Model
Fine-tuned DistilBERT (distilbert-base-uncased, 66,955,010 parameters) with a new two-class classification head. Inputs are tokenised with the DistilBERT WordPiece tokenizer and truncated to 128 tokens. Label 1 = computer-generated (CG), label 0 = original (OR).

## Training configuration
- Data: full training split, 28,343 reviews (stratified 70/15/15 split, seed 42, 33 exact-duplicate texts removed beforehand)
- AdamW, learning rate 2e-5, weight decay 0.01, 3 epochs, batch size 32, linear warm-up (10% of steps) then linear decay, gradient clipping at 1.0
- Hardware: Google Colab, NVIDIA Tesla T4 GPU
- Training time: 929.5 s (about 15.5 minutes)

## Results (test split, 6,074 reviews: 3,035 original, 3,039 computer-generated)
| Metric | Value |
|---|---|
| Accuracy | 0.9709 |
| Precision (CG class) | 0.9520 |
| Recall (CG class) | 0.9918 |
| Macro F1 | 0.9708 |
| ROC-AUC | 0.9980 |
| Inference latency | 3.59 ms per review (T4 GPU, batch 16) |

Confusion matrix: TN 2,883, FP 152, FN 25, TP 3,014. Validation accuracy after the last epoch was 0.9714, within 0.05 points of the test accuracy, so there is no sign of overfitting to the validation split.

## Preliminary error analysis
- The model errs mostly in one direction: 152 genuine reviews were flagged as fake (5.0% of genuine reviews) against 25 fakes missed (0.8% of fakes). It prefers to over-flag.
- Short reviews are harder: accuracy is 0.931 on reviews of 20 words or fewer (1,467 reviews) and 0.984 on longer ones (4,607). Missed fakes average 15.9 words against 69.5 for correctly classified reviews.
- Many false positives are short or generic genuine reviews, for example "This novel is a masterpiece. The plot is gripping and the characters are well-developed.", which looks like template praise. Genuine reviews flagged fake average 47 words.
- Some computer-generated reviews are cut off mid-sentence (an artefact of the GPT-2 generation); the model usually catches these, which suggests truncation is itself a signal. We will test this in the final error analysis.

## Notes
- This is the first full run, with no hyperparameter tuning and one seed. The 97.09% accuracy sits close to the 97% precision/recall reported for fakeRoBERTa by Salminen et al. [6], on the same kind of data. The comparison is indicative only: their split and preprocessing differ from ours.
- The code and harness are in `src/distilbert_model.py` and `src/evaluate.py`; the raw outputs are in `results/distilbert.json` and `results/distilbert_test_predictions.csv`.
