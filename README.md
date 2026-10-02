# Detection of Fake and AI-Generated Product Reviews Using Deep Learning

**Course:** ICT 4442 Deep Learning Mini Project, MIT Manipal

This repository contains the code, datasets, and reports for our project focused on detecting fake and AI-generated product reviews. We aim to compare four different neural network architecture families on a single dataset, using a unified evaluation protocol to draw fair conclusions.

## Team Members & Responsibilities

| Member | Reg. No. | Model Owned | Shared Responsibility |
|---|---|---|---|
| **Rudra Rajpurohit** | 230911570 | Fine-tuned DistilBERT | Evaluation harness, metric scripts, comparison table |
| **Vedant Totla** | 230911526 | BiLSTM + attention | Cross-domain transfer, error analysis |
| **Ishan Satyanand Thakur** | 230953356 | Text-CNN | Preprocessing / tokenisation, embeddings |
| **Vedant Agarwal** | 230953312 | TF-IDF + MLP | Dataset acquisition, EDA, literature table |

## Repository Structure

- `Project Files/code/`: Contains all model scripts, data loaders, evaluation harness, and experimental results.
  - `src/` : Individual model scripts and shared evaluation utilities.
  - `results/` : Generated JSON metric files and prediction CSVs for each model.
- `Project Files/report/`: Drafts, literature review, and the final report.
- `Project Files/DataSet/`: The datasets used for training and testing.
- `Submission GuideLines/`: Documentation and guidelines provided for the project submission.
- `nextSteps.md`: Important instructions and tasks assigned to individual team members.

## Dataset

We use the **Salminen et al. (2022) Fake Reviews Dataset** (`fake_reviews_dataset.csv`).
- **Label 1 (CG)**: Computer-generated (fake) review.
- **Label 0 (OR)**: Original human review.

The evaluation protocol enforces a strict, stratified `70/15/15` split (Train/Val/Test) with a fixed seed across all models to ensure that the comparison is perfectly valid.

## Setup and Usage

To run the models, navigate into the `Project Files/code/` directory, set up your Python virtual environment, and install dependencies:

```bash
cd "Project Files/code/"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To see the current standing of all models, you can run the comparison script:
```bash
python src/compare.py
```

## Current Results

Below is the current standing of our models based on the test split:

| Model | Owner | Acc | Prec | Rec | Macro-F1 | ROC-AUC | Params | Train (s) | Latency (ms/review) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + MLP | - | pending | pending | pending | pending | pending | pending | pending | pending |  |
| Text-CNN | Ishan Satyanand Thakur | 0.9294 | 0.9525 | 0.9039 | 0.9293 | 0.9811 | 2,120,902 | 136 | 0.08 | epochs=5, device=cuda, embed_dim=100, num_filters=100, max_len=128 |
| BiLSTM + Attention | - | pending | pending | pending | pending | pending | pending | pending | pending |  |
| DistilBERT | Rudra Rajpurohit | 0.9709 | 0.9520 | 0.9918 | 0.9708 | 0.9980 | 66,955,010 | 930 | 3.59 | preliminary: train_size=28343, epochs=3, max_len=128, lr=2e-05, device=cuda |

