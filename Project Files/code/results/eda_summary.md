# Exploratory Data Analysis (EDA) Report

**Dataset:** Salminen et al. (2022) Fake Reviews Dataset  
**Author:** Vedant Agarwal (230953312)  

---

## 1. Class Balance

| Split / Dataset | Total Reviews | CG Count (1) | CG % | OR Count (0) | OR % |
|---|---|---|---|---|---|
| Raw CSV | 40,526 | 20,294 | 50.08% | 20,232 | 49.92% |
| Deduplicated Clean | 40,491 | 20,260 | 50.04% | 20,231 | 49.96% |
| Train Split (70%) | 28,343 | 14,182 | 50.04% | 14,161 | 49.96% |
| Val Split (15%) | 6,074 | 3,039 | 50.03% | 3,035 | 49.97% |
| Test Split (15%) | 6,074 | 3,039 | 50.03% | 3,035 | 49.97% |

> **Key Insight:** The dataset is exceptionally well balanced (~50.0% CG vs 50.0% OR across all splits). Deduplication removed 35 exact-duplicate reviews, leaving 40,491 clean reviews.

---

## 2. Review Length Statistics

| Subset | Mean Words | Median Words | Std Words | Min - Max Words | Mean Chars | Median Chars |
|---|---|---|---|---|---|---|
| Overall | 67.38 | 38.0 | 69.56 | 1 - 373 | 351.59 | 198.0 |
| Cg (Computer-Generated) | 61.17 | 35.0 | 61.78 | 1 - 318 | 305.59 | 174.0 |
| Or (Original) | 73.59 | 42.0 | 76.07 | 5 - 373 | 397.66 | 224.0 |

> **Key Insight:** Original human reviews (OR) are slightly longer on average (73.59 words, std: 76.07) with a wider spread, whereas computer-generated (CG) reviews are more concise on average (61.17 words, std: 61.78) with a more bounded length distribution.

---

## 3. Category Distribution

| Category | Total Reviews | CG Count | CG % | OR Count | OR % |
|---|---|---|---|---|---|
| Kindle_Store | 4,727 | 2,362 | 49.97% | 2,365 | 50.03% |
| Books | 4,377 | 2,190 | 50.03% | 2,187 | 49.97% |
| Pet_Supplies | 4,251 | 2,125 | 49.99% | 2,126 | 50.01% |
| Home_and_Kitchen | 4,055 | 2,027 | 49.99% | 2,028 | 50.01% |
| Electronics | 3,997 | 2,003 | 50.11% | 1,994 | 49.89% |
| Sports_and_Outdoors | 3,943 | 1,970 | 49.96% | 1,973 | 50.04% |
| Tools_and_Home_Improvement | 3,857 | 1,928 | 49.99% | 1,929 | 50.01% |
| Clothing_Shoes_and_Jewelry | 3,845 | 1,921 | 49.96% | 1,924 | 50.04% |
| Toys_and_Games | 3,790 | 1,893 | 49.95% | 1,897 | 50.05% |
| Movies_and_TV | 3,585 | 1,791 | 49.96% | 1,794 | 50.04% |
| Home Appliances | 10 | 8 | 80.00% | 2 | 20.00% |
| Automotive | 8 | 6 | 75.00% | 2 | 25.00% |
| Beauty | 8 | 6 | 75.00% | 2 | 25.00% |
| Clothing | 8 | 6 | 75.00% | 2 | 25.00% |
| Sporting Goods | 8 | 6 | 75.00% | 2 | 25.00% |
| Food | 8 | 6 | 75.00% | 2 | 25.00% |
| Toys | 8 | 6 | 75.00% | 2 | 25.00% |
| Gardening | 6 | 6 | 100.00% | 0 | 0.00% |

---

## 4. Star Rating Distribution

- **Mean Rating (Overall):** 4.26 / 5.0
- **Mean Rating (CG):** 4.26 / 5.0
- **Mean Rating (OR):** 4.25 / 5.0

| Rating | Total Count | CG Count | CG % | OR Count | OR % |
|---|---|---|---|---|---|
| 1.0 Stars | 2,155 | 1,063 | 49.33% | 1,092 | 50.67% |
| 2.0 Stars | 1,978 | 970 | 49.04% | 1,008 | 50.96% |
| 3.0 Stars | 3,806 | 1,970 | 51.76% | 1,836 | 48.24% |
| 4.0 Stars | 7,983 | 3,935 | 49.29% | 4,048 | 50.71% |
| 5.0 Stars | 24,569 | 12,322 | 50.15% | 12,247 | 49.85% |
