| Model | Owner | Acc | Prec | Rec | Macro-F1 | ROC-AUC | Params | Train (s) | Latency (ms/review) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + MLP | Vedant Agarwal | 0.9503 | 0.9563 | 0.9437 | 0.9503 | 0.9904 | 5,128,449 | 240 | 0.71 | epochs=8, device=cpu, word_features=15000, char_features=25000, hidden=(128,64), batch_size=256 |
| Text-CNN | Ishan Satyanand Thakur | 0.9294 | 0.9525 | 0.9039 | 0.9293 | 0.9811 | 2,120,902 | 136 | 0.08 | epochs=5, device=cuda, embed_dim=100, num_filters=100, max_len=128 |
| BiLSTM + Attention | Vedant Totla | 0.9555 | 0.9416 | 0.9714 | 0.9555 | 0.9920 | 4,104,706 | 1126 | 0.95 | epochs_run=10, device=cpu, emb_dim=128, hidden=128 (bi), dropout=0.5, lr=1e-3, batch=64, max_len=256, vocab=30000, best_val_f1=0.9597 |
| DistilBERT | Rudra Rajpurohit | 0.9709 | 0.9520 | 0.9918 | 0.9708 | 0.9980 | 66,955,010 | 930 | 3.59 | preliminary: train_size=28343, epochs=3, max_len=128, lr=2e-05, device=cuda |
