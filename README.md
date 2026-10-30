# Detection of Fake and AI-Generated Product Reviews Using Deep Learning

**A Comparative Study of Four Architecture Families**
ICT 4442 Deep Learning Mini Project, Manipal Institute of Technology. Team 14. Guide: Dr Roshan David Jathanna.

We compare four deep learning architecture families on separating computer-generated (GPT-2) product reviews from original Amazon reviews. All four models use one dataset, one stratified 70/15/15 split and one evaluation harness, so the differences in results come from the architecture and not from the data.

## Reports

| Phase | Report |
|---|---|
| Phase 1, Synopsis (Part A) | [Reports/Phase1_Synopsis_Team14.pdf](Reports/Phase1_Synopsis_Team14.pdf) |
| Phase 2, Interim (Part B) | [Reports/Phase2_Interim_Report_Team14.pdf](Reports/Phase2_Interim_Report_Team14.pdf) |
| Phase 3, Final (Part C) | [Reports/Phase3_Final_Report_Team14.pdf](Reports/Phase3_Final_Report_Team14.pdf) |

## Team and contributions

Each member owns one model. The commit history of this repository shows who wrote which file.

| Member | Reg. No. | Model owned | Files | Other contributions |
|---|---|---|---|---|
| Rudra Rajpurohit | 230911570 | DistilBERT | `distilbert_model.py`, `evaluate.py`, `data.py`, `compare.py` | Repository setup, shared evaluation harness, data split, comparison table, literature review, `nextSteps.md` |
| Vedant Totla | 230911526 | BiLSTM + Attention | `bilstm_model.py` | Attention-weight export for the error analysis |
| Ishan Satyanand Thakur | 230953356 | Text-CNN | `textcnn_model.py`, `preprocess.py` | Tokeniser and embedding utilities |
| Vedant Agarwal | 230953312 | TF-IDF + MLP | `tfidf_mlp_model.py`, `eda.py` | Exploratory data analysis |

## Objectives

1. Quantify the detection performance of four architecture families, feed-forward, convolutional, recurrent and self-attention, under one common evaluation protocol.
2. Characterise the trade-off between accuracy and computational cost (parameter count, training time, inference latency).
3. Determine how far detectors trained on machine-generated product reviews transfer to human-written deceptive reviews from another domain. **Not carried out**, listed as future work in the final report.

## Dataset

- **Primary:** Fake Reviews Dataset of Salminen et al. (2022), Kaggle copy of the OSF release (https://osf.io/tyue9/), stored at `Project Files/DataSet/fake_reviews_dataset.csv`. 40,526 reviews with category, rating, text and label. Label `1` = computer-generated (GPT-2), label `0` = original Amazon review.
- **Secondary (planned, not used):** Deceptive Opinion Spam Corpus v1.4 of Ott et al. (2011, 2013), reserved for the cross-domain test.

## Preprocessing pipeline

`src/data.py` is shared by every model:
1. Drop rows with a missing text or label, strip whitespace, drop empty texts.
2. Remove exact duplicate texts (35 rows), leaving 40,491 reviews (20,260 computer-generated, 20,231 original).
3. One stratified 70/15/15 split with seed 42: 28,343 train, 6,074 validation, 6,074 test.

Each model then fits its own encoding on the training split only: TF-IDF word and character n-grams (MLP), a 20,000-word vocabulary with length 128 (Text-CNN), a 30,000-token vocabulary with length 256 (BiLSTM), and the DistilBERT WordPiece tokenizer with length 128. `src/eda.py` produces the dataset statistics and the distribution figure in `results/`.

## Models

| Model | Family | Script | Main settings |
|---|---|---|---|
| TF-IDF + MLP | Feed-forward | `tfidf_mlp_model.py` | 15,000 word 1-2-grams + 25,000 char 3-5-grams, MLP 128-64, Adam, 8 epochs |
| Text-CNN | Convolutional | `textcnn_model.py` | Embedding 100, filters of width 3/4/5 x 100, max-pool, dropout 0.5, 5 epochs |
| BiLSTM + Attention | Recurrent | `bilstm_model.py` | Embedding 128, BiLSTM 2 x 128, attention pooling, early stopping, up to 10 epochs |
| DistilBERT | Transformer | `distilbert_model.py` | distilbert-base-uncased, AdamW lr 2e-5, warm-up and linear decay, 3 epochs |

## Evaluation protocol

All models report through `src/evaluate.py`: accuracy, precision, recall (computer-generated class positive), macro F1 and ROC-AUC on the same test split, plus parameter count, training time and inference latency. `src/compare.py` builds the comparison table from `results/*.json`. The test split is used once, for the reported numbers.

## Results (test split, 6,074 reviews)

| Model | Owner | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC | Params | Train (s) | Latency (ms/review) | Device |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + MLP | Vedant Agarwal | 0.9503 | 0.9563 | 0.9437 | 0.9503 | 0.9904 | 5,128,449 | 240 | 0.71 | CPU |
| Text-CNN | Ishan Satyanand Thakur | 0.9294 | 0.9525 | 0.9039 | 0.9293 | 0.9811 | 2,120,902 | 136 | 0.08 | GPU |
| BiLSTM + Attention | Vedant Totla | 0.9555 | 0.9416 | 0.9714 | 0.9555 | 0.9920 | 4,104,706 | 1,126 | 0.95 | CPU |
| DistilBERT | Rudra Rajpurohit | **0.9709** | 0.9520 | **0.9918** | **0.9708** | **0.9980** | 66,955,010 | 930 | 3.59 | GPU (Colab T4) |

Training time and latency were measured on different devices, so compare them only within one device type. Each model was trained once.

**Key findings** (full discussion and error analysis in the final report, Section VI):
- DistilBERT is the most accurate and makes 41% fewer errors than TF-IDF + MLP, at thirteen times the parameters.
- Every model is least accurate on reviews of 20 words or fewer (0.843 to 0.931, against more than 0.96 above 50 words).
- 64.5% of computer-generated test reviews end without closing punctuation, against 9.2% of original ones, and every model is more accurate on computer-generated reviews that end this way. Part of the accuracy therefore comes from how the GPT-2 reviews were generated.
- Text-CNN misses the most computer-generated reviews (292), while BiLSTM + Attention and DistilBERT flag more original reviews as computer-generated.

## Repository structure

```
Reports/                    Phase 1, 2 and 3 reports (PDF)
Project Files/
  DataSet/                  fake_reviews_dataset.csv
  code/
    requirements.txt
    src/
      data.py               shared loader and 70/15/15 split
      evaluate.py           shared evaluation harness
      compare.py            builds the comparison table
      eda.py                exploratory data analysis
      preprocess.py         tokeniser and embeddings (Text-CNN)
      tfidf_mlp_model.py    TF-IDF + MLP
      textcnn_model.py      Text-CNN
      bilstm_model.py       BiLSTM + Attention
      distilbert_model.py   DistilBERT
    results/                metrics (JSON), test predictions (CSV), comparison table, EDA outputs
  report/                   report source files (.docx)
Submission GuideLines/      course guidelines, report template, synopsis
nextSteps.md                task hand-off for team members
```

## How to reproduce

```bash
cd "Project Files/code"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python eda.py                                          # dataset statistics and figure
python tfidf_mlp_model.py                              # CPU, a few minutes
python textcnn_model.py                                # CPU or GPU
python bilstm_model.py                                 # CPU or GPU, slower on CPU
python distilbert_model.py --epochs 3 --batch-size 32  # GPU recommended (e.g. Colab T4)
python compare.py                                      # writes results/comparison_table.md
```

Every script reads the dataset from `Project Files/DataSet/` and writes its results to `results/`.

## Tools and references

Built with PyTorch, scikit-learn and Hugging Face Transformers. Main references: Salminen et al. (2022) for the dataset, Ott et al. (2011, 2013), Kim (2014), Hochreiter and Schmidhuber (1997), Sanh et al. (2019). The full IEEE reference list is in the final report.

Claude (Anthropic) was used to write scaffolding code for the shared evaluation pipeline and the DistilBERT training script, and to help draft parts of the reports. All experiments were run by the team.
