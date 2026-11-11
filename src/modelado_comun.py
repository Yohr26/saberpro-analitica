"""Componentes comunes de la fase de modelado (CRISP-DM: Modelado y Evaluacion).

Todo lo que "aprende" de los datos (imputacion C1, codificaciones, seleccion de variables y modelo) vive dentro de
un sklearn Pipeline que se ajusta SOLO con el conjunto de entrenamiento (2020-2021) o con los pliegues de
entrenamiento de la validacion cruzada. El conjunto de prueba (2022) solo se transforma, nunca se ajusta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

import config as C

SEMILLA = C.SEMILLA
CACHE = C.RAIZ / "data" / "processed" / "cache_pipeline"

# ---------------------------------------------------------------- variables
OBJETIVO = "puntaje_global"
OBJETIVOS_IMPUTACION = ["estrato", "educ_madre", "educ_padre"]  # categoricas con nulos (metodo C1)
ONEHOT = ["estrato", "educ_madre", "educ_padre", "genero", "inst_caracter", "sector", "inst_origen", "region_inst",
          "modalidad", "pago_beca", "pago_credito", "pago_padres", "pago_propio"]
NUMERICAS = ["edad", "matricula_ord", "horas_trabajo_ord", "tiene_automovil", "tiene_lavadora", "tiene_computador",
             "tiene_internet"]
DERIVADAS = ["estrato_num", "educ_madre_ord", "educ_padre_ord"]  # las crea el imputador tras imputar
TARGET_ENC = ["inst_cod", "nucleo", "depto_inst"]
ENTRADAS = ONEHOT + NUMERICAS + TARGET_ENC  # variables conocidas ANTES de la prueba (sin puntajes de modulos)

# Bloques para la tabla de ablacion
BLOQUES = {
    "Identidad institucional (código)": ["inst_cod"],
    "Tipo y territorio de la institución": ["inst_caracter", "sector", "inst_origen", "region_inst", "depto_inst"],
    "Modalidad y núcleo del programa": ["modalidad", "nucleo"],
    "Estrato y educación de los padres": ["estrato", "educ_madre", "educ_padre"],
    "Bienes y conectividad del hogar": ["tiene_automovil", "tiene_lavadora", "tiene_computador", "tiene_internet"],
    "Matrícula y forma de pago": ["matricula_ord", "pago_beca", "pago_credito", "pago_padres", "pago_propio"],
    "Edad, género y trabajo": ["edad", "genero", "horas_trabajo_ord"],
}


def cargar_datos():
    """Devuelve (train 2020-2021, test 2022) SIN imputar, con los cinco modulos validos (n=467.356 y 193.356)."""
    df = pd.read_parquet(C.LIMPIO)
    d = df[df[OBJETIVO].notna()].reset_index(drop=True)
    train, test = d[d["anio"] <= 2021].reset_index(drop=True), d[d["anio"] == 2022].reset_index(drop=True)
    for x in (train, test):
        for c in ONEHOT + TARGET_ENC:
            if c in x:
                x[c] = x[c].astype("object").where(x[c].notna(), np.nan)
        x["inst_cod"] = x["inst_cod"].astype(str)
    return train, test


def umbral_riesgo(y_train, q=0.25):
    return float(np.quantile(y_train, q))


# ---------------------------------------------------------------- imputador C1
class ImputadorC1(BaseEstimator, TransformerMixin):
    """Imputacion de variables categoricas con un modelo de gradient boosting (metodo C1 de la seccion 9.2.4).

    method = "modelo" (C1) | "moda" (A) | "ninguno" (los faltantes pasan a ser la categoria 'FALTANTE').
    fit() usa unicamente los datos recibidos (entrenamiento); transform() solo aplica lo aprendido.
    Tambien deriva las versiones ordinales (estrato_num, educ_*_ord) a partir de las categorias ya imputadas.
    """

    def __init__(self, method="modelo", max_train=150_000, random_state=SEMILLA):
        self.method = method
        self.max_train = max_train
        self.random_state = random_state

    def _features(self, X, target):
        cols = [c for c in X.columns if c != target and c not in DERIVADAS]
        F = X[cols].copy()
        for c in cols:
            if c == "inst_cod":
                F[c] = F[c].map(self.inst_codes_).fillna(-1).astype(float)
            elif not pd.api.types.is_numeric_dtype(F[c]):
                F[c] = pd.Categorical(F[c], categories=self.cats_.get(c, sorted(F[c].dropna().unique())))
        return F

    def fit(self, X, y=None):
        X = X.reset_index(drop=True)
        self.targets_ = [t for t in OBJETIVOS_IMPUTACION if t in X.columns]
        self.modes_ = {t: X[t].mode().iat[0] for t in self.targets_}
        self.inst_codes_ = {k: i for i, k in enumerate(sorted(X["inst_cod"].unique()))} if "inst_cod" in X else {}
        self.cats_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in X.columns
                      if not pd.api.types.is_numeric_dtype(X[c]) and c != "inst_cod"}
        self.models_ = {}
        if self.method == "modelo":
            for t in self.targets_:
                known = X[X[t].notna()]
                if len(known) > self.max_train:
                    known = known.sample(self.max_train, random_state=self.random_state)
                m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features="from_dtype",
                                                   random_state=self.random_state)
                m.fit(self._features(known, t), known[t].astype(str))
                self.models_[t] = m
        return self

    def transform(self, X):
        X = X.copy().reset_index(drop=True)
        for t in self.targets_:
            faltan = X[t].isna()
            if faltan.any():
                if self.method == "modelo":
                    X.loc[faltan, t] = self.models_[t].predict(self._features(X.loc[faltan], t))
                elif self.method == "moda":
                    X.loc[faltan, t] = self.modes_[t]
                else:
                    X.loc[faltan, t] = "FALTANTE"
        if "estrato" in X:
            X["estrato_num"] = X["estrato"].map(C.ORD_ESTRATO).astype(float)
        if "educ_madre" in X:
            X["educ_madre_ord"] = X["educ_madre"].map(C.ORD_EDUCACION).astype(float)
        if "educ_padre" in X:
            X["educ_padre_ord"] = X["educ_padre"].map(C.ORD_EDUCACION).astype(float)
        return X


# ---------------------------------------------------------------- pipeline
def _selector(alpha):
    return SelectFromModel(Lasso(alpha=alpha, max_iter=2000), threshold=1e-6)


SELECTORES = {"ninguno": "passthrough", "lasso_suave": _selector(0.01), "lasso_fuerte": _selector(0.1)}


def construir_pipeline(estimador, tarea="regresion", imputador="modelo", selector="ninguno", columnas=None, cache=True):
    """Pipeline = imputador C1 -> codificacion -> seleccion de variables -> estimador (todo se ajusta solo con train)."""
    columnas = list(columnas or ENTRADAS)
    oh = [c for c in ONEHOT if c in columnas]
    te = [c for c in TARGET_ENC if c in columnas]
    num = [c for c in NUMERICAS if c in columnas] + [d for d, base in
                                                      (("estrato_num", "estrato"), ("educ_madre_ord", "educ_madre"),
                                                       ("educ_padre_ord", "educ_padre")) if base in columnas]
    partes = []
    if oh:
        partes.append(("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False), oh))
    if te:
        partes.append(("te", TargetEncoder(target_type="continuous" if tarea == "regresion" else "binary",
                                           smooth="auto", cv=KFold(5, shuffle=True, random_state=SEMILLA)), te))
    if num:
        partes.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                        ("esc", StandardScaler())]), num))
    prep = ColumnTransformer(partes, remainder="drop", verbose_feature_names_out=True)
    prep.set_output(transform="pandas")
    memory = Memory(str(CACHE), verbose=0) if cache else None
    sel = SELECTORES[selector] if isinstance(selector, str) else selector
    return Pipeline([("imp", ImputadorC1(method=imputador)), ("prep", prep), ("sel", sel), ("est", estimador)],
                    memory=memory)


# ---------------------------------------------------------------- metricas
def alerta_top(score_riesgo, y_bajo, frac=0.25):
    """Recall y precision al alertar al 25 % de estudiantes con mayor riesgo."""
    score = np.asarray(score_riesgo)
    k = int(round(frac * len(score)))
    top = np.argsort(-score, kind="stable")[:k]
    y = np.asarray(y_bajo)
    return float(y[top].sum() / max(y.sum(), 1)), float(y[top].mean())


def metricas_regresion(y, pred, umbral):
    y, pred = np.asarray(y), np.asarray(pred)
    bajo = (y < umbral).astype(int)
    rec, pre = alerta_top(-pred, bajo)
    return {"RMSE": float(np.sqrt(np.mean((y - pred) ** 2))), "MAE": float(mean_absolute_error(y, pred)),
            "R2": float(r2_score(y, pred)), "AUC": float(roc_auc_score(bajo, -pred)),
            "AP": float(average_precision_score(bajo, -pred)), "recall25": rec, "precision25": pre}


def metricas_clasificacion(bajo, proba):
    bajo, proba = np.asarray(bajo), np.asarray(proba)
    rec, pre = alerta_top(proba, bajo)
    return {"AUC": float(roc_auc_score(bajo, proba)), "AP": float(average_precision_score(bajo, proba)),
            "recall25": rec, "precision25": pre, "Brier": float(brier_score_loss(bajo, proba))}


def nombres_origen(nombres):
    """Mapea nombres de columnas transformadas (oh__estrato_Estrato 1) a la variable original."""
    out = []
    for n in nombres:
        base = n.split("__", 1)[-1]
        for v in ONEHOT + TARGET_ENC + NUMERICAS + DERIVADAS:
            if base == v or base.startswith(v + "_") or base.startswith("missingindicator_" + v):
                out.append(v)
                break
        else:
            out.append(base)
    return out
