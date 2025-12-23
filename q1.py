import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

RND = 42
np.random.seed(RND)

#load and time sorted split
df = pd.read_csv("train.csv", parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

#features
def add_time_features(df_):
    df = df_.copy()
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    return df

def add_cyclical(df_):
    df = df_.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df

def add_extra_features(df_):
    df = df_.copy()
    df["temp_diff"] = df["temp"] - df["atemp"]
    df["humid_ws"] = df["humidity"] * df["windspeed"]
    return df

train_df = add_extra_features(add_cyclical(add_time_features(train_df)))
test_df  = add_extra_features(add_cyclical(add_time_features(test_df)))

#features and target
numeric_features = [
    "temp","atemp","humidity","windspeed",
    "hour_sin","hour_cos","temp_diff","humid_ws",
    "month","weekday"
]

categorical_features = ["season","weather","holiday","workingday"]
target = "count"

y_train = train_df[target].values
y_test  = test_df[target].values
y_train_log = np.log1p(y_train)

def invert_log_safe(y_log):
    return np.expm1(np.clip(y_log, -20, 20))

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(train_df[categorical_features])

scaler_input = StandardScaler().fit(train_df[numeric_features].values)

#to remove the interactions
def single_feature_mask_and_names(poly, input_names):
    names = poly.get_feature_names_out(input_names)
    mask = np.array([(" " not in n) and ("*" not in n) for n in names])
    return mask

#linear model
Xtr_num = scaler_input.transform(train_df[numeric_features])
Xte_num = scaler_input.transform(test_df[numeric_features])
Xtr_cat = ohe.transform(train_df[categorical_features])
Xte_cat = ohe.transform(test_df[categorical_features])

Xtr_lin = np.hstack([Xtr_num, Xtr_cat])
Xte_lin = np.hstack([Xte_num, Xte_cat])

lin = LinearRegression()
lin.fit(Xtr_lin, y_train_log)

pred_tr = invert_log_safe(lin.predict(Xtr_lin))
pred_tr = np.clip(pred_tr, 1e-6, None)
smear_lin = np.mean(y_train / pred_tr)

pred_te = invert_log_safe(lin.predict(Xte_lin))
pred_te_corr = np.clip(pred_te * smear_lin, 0, None)

lin_mse = mean_squared_error(y_test, pred_te_corr)
lin_r2  = r2_score(y_test, pred_te_corr)

#polynomial w/o interactions (degree 2,3,4)
cv = KFold(n_splits=5, shuffle=True, random_state=RND)

def eval_poly_no_interactions(deg, clip, svd_dim):
    Xtr_s = scaler_input.transform(train_df[numeric_features])
    Xte_s = scaler_input.transform(test_df[numeric_features])

    poly = PolynomialFeatures(deg, include_bias=False)
    Xtr_p = poly.fit_transform(Xtr_s)
    Xte_p = poly.transform(Xte_s)

    mask = single_feature_mask_and_names(poly, numeric_features)
    Xtr_p = Xtr_p[:, mask]
    Xte_p = Xte_p[:, mask]

    Xtr_p = np.clip(Xtr_p, -clip, clip)
    Xte_p = np.clip(Xte_p, -clip, clip)

    sc = StandardScaler().fit(Xtr_p)
    Xtr_p = sc.transform(Xtr_p)
    Xte_p = sc.transform(Xte_p)

    if svd_dim is not None:
        svd = TruncatedSVD(n_components=svd_dim, random_state=RND)
        Xtr_p = svd.fit_transform(Xtr_p)
        Xte_p = svd.transform(Xte_p)

    Xtr_f = np.hstack([Xtr_p, Xtr_cat])
    Xte_f = np.hstack([Xte_p, Xte_cat])

    alphas = np.logspace(-6, 8, 35)
    ridge = RidgeCV(alphas=alphas, cv=cv)
    ridge.fit(Xtr_f, y_train_log)

    pred_tr = invert_log_safe(ridge.predict(Xtr_f))
    pred_tr = np.clip(pred_tr, 1e-6, None)
    smear = np.mean(y_train / pred_tr)

    pred_te = invert_log_safe(ridge.predict(Xte_f))
    pred_te = np.clip(pred_te * smear, 0, None)

    return {
        "mse": mean_squared_error(y_test, pred_te),
        "r2": r2_score(y_test, pred_te),
        "alpha": ridge.alpha_,
        "n_feats": Xtr_p.shape[1]
    }

res2 = eval_poly_no_interactions(2, 50, None)
res3 = eval_poly_no_interactions(3, 30, 20)
res4 = eval_poly_no_interactions(4, 15, 25)

#quadratic with interactions(degree 2)
poly2 = PolynomialFeatures(2, include_bias=False)

Xtr_p2 = poly2.fit_transform(Xtr_num)
Xte_p2 = poly2.transform(Xte_num)

Xtr_p2 = np.clip(Xtr_p2, -50, 50)
Xte_p2 = np.clip(Xte_p2, -50, 50)

sc2 = StandardScaler().fit(Xtr_p2)
Xtr_p2 = sc2.transform(Xtr_p2)
Xte_p2 = sc2.transform(Xte_p2)

Xtr_q = np.hstack([Xtr_p2, Xtr_cat])
Xte_q = np.hstack([Xte_p2, Xte_cat])

quad = RidgeCV(alphas=np.logspace(-6, 8, 35), cv=cv)
quad.fit(Xtr_q, y_train_log)

pred_tr = invert_log_safe(quad.predict(Xtr_q))
pred_tr = np.clip(pred_tr, 1e-6, None)
smear_q = np.mean(y_train / pred_tr)

pred_te = invert_log_safe(quad.predict(Xte_q))
pred_te = np.clip(pred_te * smear_q, 0, None)

quad_mse = mean_squared_error(y_test, pred_te)
quad_r2  = r2_score(y_test, pred_te)

models = [
    ("Linear", lin_mse, lin_r2, False, 10, "N/A"),
    ("Degree 2 (no interactions)", res2["mse"], res2["r2"], False, res2["n_feats"], int(res2["alpha"])),
    ("Degree 3 (no interactions)", res3["mse"], res3["r2"], False, res3["n_feats"], int(res3["alpha"])),
    ("Degree 4 (no interactions)", res4["mse"], res4["r2"], False, res4["n_feats"], int(res4["alpha"])),
    ("Degree 2 (WITH interactions)", quad_mse, quad_r2, True, Xtr_p2.shape[1], int(quad.alpha_))
]

print("\n---- MODEL PERFORMANCE (TEST SET) ----\n")
for m in models:
    print(f"{m[0]:<30} | MSE = {m[1]:8.2f} | R2 = {m[2]:7.4f} | Interactions: {'Yes' if m[3] else 'No':<3} | n_num_feats = {m[4]:3d} | alpha = {m[5]}")

best = min(models, key=lambda x: x[1])

print("\n---- BEST MODEL (TEST SET MSE) ----")
print(f"Best Model: {best[0]}")
print(f"MSE  : {best[1]:.2f}")
print(f"R2   : {best[2]:.4f}")
print(f"Uses interactions? : {'Yes' if best[3] else 'No'}")
print(f"Selected alpha : {best[5]}")
