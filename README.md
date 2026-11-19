# Video Perceived Quality: JND Prediction with SVR

Diploma thesis, **Department of Computer Science & Engineering, University of Ioannina** (2025)
Author: **Panteleimon Kakos** · Supervisor: Prof. Lisimachos P. Kondis

Full thesis (Greek): [`thesis/KAKOS_5060.pdf`](thesis/KAKOS_5060.pdf)

## About
Finding the **Just Noticeable Difference (JND)** points of a video, meaning the compression levels where viewers start to notice a drop in quality, normally needs long subjective experiments with real viewers.

This project **predicts the JND points with machine learning** instead, using only features that are computed automatically from the video:

| Feature | Meaning |
|---|---|
| **B** | Brightness |
| **SC** | Spatial Complexity (detail inside each frame) |
| **TC**, **TC2** | Temporal Complexity (motion between frames) |

## Pipeline
```
VideoSet videos (YUV 4:2:0)
        │
        ▼
EVCA (Enhanced Video Complexity Analyzer)  →  B, SC, TC, TC2 per frame
        │
        ▼
Average per video  →  features with jnds.xlsx  (features + JND1, JND2, JND3)
        │
        ▼
SVR (RBF kernel) + grid search over C, epsilon, gamma
        │
        ▼
Evaluation with MAE and RMSE
```

- **Dataset:** [VideoSet](https://doi.org/10.1016/j.jvcir.2017.04.009): videos with experimentally measured JND points
- **Feature extraction:** [EVCA](https://github.com/cd-athena/EVCA)
- **Model:** `sklearn.svm.SVR`

## Experiments
The model is tested on different test sets:
- `random`: 10 random videos
- `highB` / `lowB`: videos with high / low brightness
- `highSC` / `lowSC`: videos with high / low spatial complexity
- `highTC` / `lowTC`: videos with high / low temporal complexity

### Results on the random test set (in QP units)
| Target | MAE | RMSE |
|---|---|---|
| JND1 | 1.781 | 2.423 |
| JND2 | 1.703 | 2.125 |
| JND3 | 1.282 | 1.514 |

Full results and discussion are in Chapter 4 of the thesis.

## How to run
```bash
pip install -r requirements.txt
python SvrModel.py
```
The file `features with jnds.xlsx` must be in the same folder as `SvrModel.py`.

## Tech
Python · pandas · NumPy · scikit-learn
