# Data Directory

This directory is a placeholder for the "Impostor Hunt" Kaggle competition dataset, which cannot be included due to Kaggle’s data usage restrictions.

## Instructions
1. **Download the Data**:
   - Visit the [Impostor Hunt competition](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt).
   - Download `train.csv` and the `test/` directory (containing `article_*` folders).
2. **Place the Files**:
   - Copy `train.csv` to `impostor_hunt/data/`.
   - Copy the `test/` directory to `impostor_hunt/data/test/`.
3. **Verify**:
   - Ensure `train.csv` has 95 rows and columns `id`, `real_text_id`.
   - Ensure `test/` contains 1068 `article_*` folders, each with `file_1.txt` and `file_2.txt`.

## Notes
- The code in `impostor_hunt_kaggle.ipynb` expects this directory structure.
- If running on Kaggle, update paths in the notebook to `/kaggle/input/fake-or-real-the-impostor-hunt/data/`.