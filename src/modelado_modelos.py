"""Bateria de modelos, espacios de busqueda y utilidades de ajuste compartidas por los scripts 08 y 09."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

import config as C
import modelado_comun as M

SEM = C.SEMILLA
SEL = ["ninguno", "lasso_suave", "lasso_fuerte"]

# nombre -> (constructor(params), espacio de busqueda del estimador, tarea)
MODELOS = {
    "Ridge": (lambda p: Ridge(**p), {"alpha": loguniform(0.1, 1e3)}, "reg"),
    "RandomForest": (lambda p: RandomForestRegressor(n_estimators=100, max_samples=0.3, n_jobs=-1, random_state=SEM, **p),
                     {"max_depth": [8, 12, 16], "min_samples_leaf": [20, 50, 100], "max_features": [0.3, 0.5, 0.8]}, "reg"),
    "HistGradientBoosting": (lambda p: HistGradientBoostingRegressor(early_stopping=True, random_state=SEM, **p),
                             {"learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [15, 31, 63],
                              "min_samples_leaf": [50, 200, 500], "l2_regularization": [0.0, 1.0, 10.0], "max_iter": [300, 500]}, "reg"),
    "LightGBM": (lambda p: LGBMRegressor(subsample_freq=1, random_state=SEM, n_jobs=8, verbose=-1, **p),
                 {"n_estimators": [300, 600], "learning_rate": [0.03, 0.06, 0.1], "num_leaves": [15, 31, 63],
                  "min_child_samples": [50, 200, 500], "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8],
                  "reg_lambda": [0.0, 5.0, 20.0]}, "reg"),
    "XGBoost": (lambda p: XGBRegressor(tree_method="hist", random_state=SEM, n_jobs=8, **p),
                {"n_estimators": [300, 500], "learning_rate": [0.03, 0.06, 0.1], "max_depth": [4, 6, 8],
                 "min_child_weight": [20, 100], "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8],
                 "reg_lambda": [1.0, 10.0]}, "reg"),
    "Regresión logística": (lambda p: LogisticRegression(max_iter=300, **p), {"C": loguniform(0.01, 10)}, "clf"),
    "RandomForest (clasif.)": (lambda p: RandomForestClassifier(n_estimators=100, max_samples=0.3, n_jobs=-1, random_state=SEM, **p),
                               {"max_depth": [8, 12, 16], "min_samples_leaf": [20, 50, 100], "max_features": [0.3, 0.5]}, "clf"),
    "HistGradientBoosting (clasif.)": (lambda p: HistGradientBoostingClassifier(early_stopping=True, random_state=SEM, **p),
                                       {"learning_rate": [0.03, 0.06, 0.1], "max_leaf_nodes": [15, 31, 63],
                                        "min_samples_leaf": [50, 200, 500], "l2_regularization": [0.0, 1.0, 10.0], "max_iter": [300, 500]}, "clf"),
    "LightGBM (clasif.)": (lambda p: LGBMClassifier(subsample_freq=1, random_state=SEM, n_jobs=8, verbose=-1, **p),
                           {"n_estimators": [300, 600], "learning_rate": [0.03, 0.06, 0.1], "num_leaves": [15, 31, 63],
                            "min_child_samples": [50, 200, 500], "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8],
                            "reg_lambda": [0.0, 5.0, 20.0]}, "clf"),
    "XGBoost (clasif.)": (lambda p: XGBClassifier(tree_method="hist", random_state=SEM, n_jobs=8, **p),
                          {"n_estimators": [300, 500], "learning_rate": [0.03, 0.06, 0.1], "max_depth": [4, 6, 8],
                           "min_child_weight": [20, 100], "subsample": [0.7, 0.9], "colsample_bytree": [0.5, 0.8],
                           "reg_lambda": [1.0, 10.0]}, "clf"),
}
ENSAMBLES_REG = ["RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost"]


def ensamblar(nombre, params, selector):
    """Parte del Pipeline posterior a la preparacion: seleccion de variables + estimador."""
    return Pipeline([("sel", M.SELECTORES[selector]), ("est", MODELOS[nombre][0](params))])


def predecir(pipe, X, tarea):
    return pipe.predict(X) if tarea == "reg" else pipe.predict_proba(X)[:, 1]


def etiquetas_estratificacion(train, minimo=10):
    vc = train["inst_cod"].value_counts()
    return train["inst_cod"].where(train["inst_cod"].map(vc) >= minimo, "otras").values


def pliegues(train, esquema, k=5):
    if esquema == "conocidas":
        return list(StratifiedKFold(k, shuffle=True, random_state=SEM).split(train, etiquetas_estratificacion(train)))
    return list(GroupKFold(k).split(train, groups=train["inst_cod"].values))


def preparar(Xtr, ytr, Xva, imputador="modelo", columnas=None):
    """Ajusta imputador C1 + codificaciones SOLO con Xtr y transforma (Xtr, Xva)."""
    pre = M.construir_pipeline("passthrough", imputador=imputador, columnas=columnas, cache=False)[:-2]
    A = pre.fit_transform(Xtr if columnas is None else Xtr[columnas], ytr)
    B = pre.transform(Xva if columnas is None else Xva[columnas])
    return pre, A, B
