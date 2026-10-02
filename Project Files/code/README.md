# Detection of Fake and AI-Generated Product Reviews Using Deep Learning

ICT 4442 Deep Learning Mini Project, MIT Manipal. A comparison of four architecture
families on one dataset, one split and one evaluation protocol.

| Member | Reg. No. | Model owned | Shared responsibility |
|---|---|---|---|
| Rudra Rajpurohit | 230911570 | Fine-tuned DistilBERT | Evaluation harness, metric scripts, comparison table |
| Vedant Totla | 230911526 | BiLSTM + attention | Cross-domain transfer, error analysis |
| Ishan Satyanand Thakur | 230953356 | Text-CNN | Preprocessing / tokenisation, embeddings |
| Vedant Agarwal | 230953312 | TF-IDF + MLP | Dataset acquisition, EDA, literature table |

## Layout
- `src/evaluate.py` : common evaluation harness (shared by all models)
- `src/` : model scripts, one per owner
- `results/` : JSON metric files, one per model, produced by the harness
- `data/` : dataset CSV (not tracked; see below)

## Data
Download the Salminen et al. (2022) Fake Reviews Dataset
(OSF: https://osf.io/tyue9/) and place the CSV in `data/`.
Labels: `OR` = original (human), `CG` = computer-generated.

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Common evaluation protocol
Stratified 70/15/15 split, fixed seed, metrics: accuracy, precision, recall,
macro F1, ROC-AUC, plus parameter count, training time and inference latency.
Every model must call `evaluate.evaluate_model(...)` so the numbers are comparable.
