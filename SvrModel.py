
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("features with jnds.xlsx")


X = df[["B", "SC", "TC", "TC2"]].values
targets = {
    "JND1": df["JND1"].values,
    "JND2": df["JND2"].values,
    "JND3": df["JND3"].values,
}


TEST_SETS = {
    "random":[2, 7, 15, 23, 31, 42, 55, 66, 78, 89],
    'highB': [74, 11, 75, 57, 64, 81, 62],
    'lowB': [42, 20, 101, 43, 49, 47, 48],
    'highSC': [12, 103, 15, 102, 18, 6, 115],
    'lowSC': [101, 42, 43, 49, 47, 44, 48],
    'highTC': [15, 12, 103, 13, 18, 102, 6],
    'lowTC': [101, 42, 43, 49, 47, 44, 48]
}


C_values     = [0.1, 0.4, 0.6, 1, 5, 10, 20, 50, 70, 100]
eps_values   = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
gamma_values = ["scale", "auto", 0.001, 0.005, 0.01, 0.05, 0.1]

def Svr(name, y, test_mask, set_name):
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]


    best_mae = 10
    best_pred = None
    best_params = (None, None, None)

    for C in C_values:
        for eps in eps_values:
            for gam in gamma_values:
                model = make_pipeline(SVR(kernel="rbf", C=C, epsilon=eps, gamma=gam))
                model.fit(X_train, y_train)
                y_pred_tmp = model.predict(X_test)
                mae_tmp = mean_absolute_error(y_test, y_pred_tmp)
                if mae_tmp < best_mae:
                    best_mae = mae_tmp
                    best_pred = y_pred_tmp
                    best_params = (C, eps, gam)


    y_pred = best_pred



    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))


    Cb, Eb, Gb = best_params
    print(f"\n========== {set_name} / {name} ==========")
    print(f"Best params: C={Cb}, epsilon={Eb}, gamma={Gb}")
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} ")

    out = pd.DataFrame({
        "VideoNumber": df.loc[test_mask, "VideoNumber"].values,
        f"True_{name}": y_test,
        f"Pred_{name}": y_pred,
    }).sort_values("VideoNumber")
    print(out.to_string(index=False))


for set_name, test_videos in TEST_SETS.items():
    print(f"\n\n===== Αποτελέσματα για {set_name} (test videos={test_videos}) =====")
    test_mask = df["VideoNumber"].isin(test_videos)
    for name, y in targets.items():
        Svr(name, y, test_mask, set_name)







