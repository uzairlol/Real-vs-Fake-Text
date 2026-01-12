# Impostor Hunt: Kaggle Competition Solution (Public Score: 0.90663)

## Overview

This repository contains my solution for the Impostor Hunt Kaggle competition, where the goal is to classify whether `text_1` or `text_2` in each pair is real (1) or fake (2) across 1068 test pairs, using a small training set of 95 samples (both being LLM generated). My best public leaderboard score is **0.90663**, achieved using a CatBoost model with SciBERT embeddings, augmented by features like perplexity, bigram Jaccard, n-gram KL divergence, and burstiness. The solution addresses the label correction issue announced on July 7, 2025, which previously caused scores to drop from \~0.78–0.79 to \~0.44–0.45.

## Approach

- **Data**: Used the corrected `train.csv` (95 samples, balanced after augmentation to 190 samples) and `test/` directory (1068 pairs). Applied data augmentation by swapping text pairs.
- **Features**:
  - **SciBERT Embeddings**: PCA-reduced to \~51 components (`emb_diff_pca_0` importance: 79.82).
  - **Text Features**: Character/word count, Flesch reading ease, type-token ratio (TTR), punctuation count.
  - **Watermark**: Emdash count.
  - **Perplexity**: GPT-2-based, including sentence-level mean and std.
  - **Similarity**: Cosine similarity, bigram Jaccard, n-gram KL divergence.
  - **Burstiness**: Sentence length variation.
- **Model**: CatBoost with Optuna hyperparameter tuning (`N_TRIALS=100`, 5-fold CV, early stopping).
- **Challenges Overcome**:
  - Handled label correction in `train.csv`, recovering from score drop (\~0.44–0.45) to 0.90663.
  - Reduced overfitting (CV accuracy: 0.95263, fold std: 0.03069, CV-to-public gap \~0.05).
  - Managed short texts (&lt;2 words) with unigram fallbacks for Jaccard and KL divergence.

## Results

- **Public Score**: 0.90663 (personal best, achieved July 13, 2025).
- **Cross-Validation**: Mean accuracy 0.95263, fold std 0.03069.
- **Key Features**: `emb_diff_pca_0` (79.82), `perplexity_diff` (5.07), `cosine_similarity` (4.89).
- **Runtime**: \~25–30 minutes on Kaggle with GPU (SciBERT + GPT-2 embeddings).
- **Feature Importance**: See `results/feature_importance.csv` for details.

## Repository Structure

```
impostor_hunt/
├── impostor_hunt_kaggle.ipynb      # Main notebook (CatBoost, 0.90663 score)
├── requirements.txt                # Python dependencies
├── data/                          # Placeholder for Kaggle data
│   ├── train.csv                  # Placeholder (download from Kaggle)
│   ├── test/                      # Placeholder directory (download from Kaggle)
│   └── README.md                  # Instructions to obtain data
├── results/                       # Analysis outputs
│   └── feature_importance.csv     # Feature importance from CatBoost
├── submission_catboost.csv        # Sample submission (0.90663 score)
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/impostor_hunt.git
   cd impostor_hunt
   ```
2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Download Competition Data**:
   - Obtain `train.csv` and `test/` from the Impostor Hunt competition.
   - Place `train.csv` in `data/` and `test/` in `data/test/`.
4. **Run the Notebook**:
   - Use a GPU-enabled environment (e.g., Kaggle, Google Colab, or local with CUDA).
   - Execute `impostor_hunt_kaggle.ipynb` to train the model and generate predictions.
5. **Output**:
   - Generates `submission_catboost.csv` for Kaggle submission.
   - Feature importance saved in `results/feature_importance.csv`.

## Dependencies

See `requirements.txt` for the full list. Key packages:

- Python 3.8+
- pandas, numpy, scikit-learn
- transformers (SciBERT, GPT-2)
- catboost, optuna
- nltk, textstat

## Notes

- **Data Restrictions**: `train.csv` and `test/` are not included due to Kaggle’s data usage rules. Download them from the competition page.
- **Reproducibility**: Set `RANDOM_STATE=42` for consistent results.
- **Future Improvements**:
  - Experiment with DeBERTa embeddings (`microsoft/deberta-base`) for better performance.
  - Ensemble CatBoost with LightGBM models to aim for \~0.92 score.
  - Add visualization of feature importance (e.g., bar plot).

## License

This project is licensed under the MIT License (see `LICENSE`).
