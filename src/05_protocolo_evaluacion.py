"""05 - Protocolo de evaluacion y lineas base (CRISP-DM: puente hacia Modelado y Evaluacion).

Define y ejecuta lo que NO depende de un modelo avanzado:
  * particion de datos (temporal: entrenamiento 2020-2021, prueba 2022),
  * metricas principal y secundarias,
  * lineas base simples que cualquier modelo posterior debe superar.
El protocolo completo (criterios de exito incluidos) esta en docs/protocolo_evaluacion.md.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

import config as C

df = pd.read_parquet(C.IMPUTADO)
d = df[df["puntaje_global"].notna()].copy()
train, test = d[d["anio"] <= 2021].copy(), d[d["anio"] == 2022].copy()
print(f"Entrenamiento 2020-2021: {len(train):,} | Prueba 2022: {len(test):,}")

# Etiqueta de riesgo: cuartil inferior segun la distribucion del ENTRENAMIENTO (sin fuga de informacion)
umbral = train["puntaje_global"].quantile(0.25)
for x in (train, test):
    x["bajo"] = (x["puntaje_global"] < umbral).astype(int)
pd.DataFrame([
    {"conjunto": "Entrenamiento (2020-2021)", "n": len(train), "pct_bajo_desempeno": round(100 * train["bajo"].mean(), 1)},
    {"conjunto": "Prueba (2022)", "n": len(test), "pct_bajo_desempeno": round(100 * test["bajo"].mean(), 1)},
    {"conjunto": f"Umbral de bajo desempeño (P25 del entrenamiento)", "n": np.nan, "pct_bajo_desempeno": round(umbral, 2)},
]).to_csv(C.TABLAS / "protocolo_particion.csv", index=False, encoding="utf-8-sig")


def media_por(col):
    m = train.groupby(col)["puntaje_global"].mean()
    return test[col].map(m).fillna(train["puntaje_global"].mean())


pred = {
    "B0. Media global": pd.Series(train["puntaje_global"].mean(), index=test.index),
    "B1. Media por estrato": media_por("estrato"),
    "B2. Media por carácter académico": media_por("inst_caracter"),
    "B3. Media por institución": media_por("inst_cod"),
}

# B4: regresion lineal multiple (mismas variables del EDA, sin puntajes)
X_cols = ["estrato_num", "educ_madre_ord", "educ_padre_ord", "matricula_ord", "horas_trabajo_ord",
          "tiene_internet", "tiene_computador", "edad"]
dum = ["sector", "inst_caracter", "region_inst", "modalidad", "genero"]
Xa = pd.get_dummies(pd.concat([train, test])[X_cols + dum], columns=dum, drop_first=True, dtype=float)
Xa = Xa.fillna(Xa.iloc[:len(train)].median())
ols = sm.OLS(train["puntaje_global"].values, sm.add_constant(Xa.iloc[:len(train)])).fit()
pred["B4. Regresión lineal múltiple"] = pd.Series(ols.predict(sm.add_constant(Xa.iloc[len(train):], has_constant="add")),
                                                    index=test.index)

filas = []
n_alerta = int(round(0.25 * len(test)))
for nombre, p in pred.items():
    rmse = float(np.sqrt(mean_squared_error(test["puntaje_global"], p)))
    fila = {"linea_base": nombre, "RMSE": round(rmse, 2), "MAE": round(mean_absolute_error(test["puntaje_global"], p), 2),
            "R2": round(r2_score(test["puntaje_global"], p), 4)}
    if p.nunique() > 1:  # clasificacion: puntaje esperado bajo => riesgo alto
        riesgo = -p
        fila["AUC_ROC_bajo_desempeno"] = round(roc_auc_score(test["bajo"], riesgo), 4)
        top = riesgo.sort_values(ascending=False).index[:n_alerta]
        fila["recall_en_25pct_alertado"] = round(test.loc[top, "bajo"].sum() / test["bajo"].sum(), 4)
        fila["precision_en_25pct_alertado"] = round(test.loc[top, "bajo"].mean(), 4)
    else:
        fila["AUC_ROC_bajo_desempeno"] = 0.5
        fila["recall_en_25pct_alertado"] = 0.25
        fila["precision_en_25pct_alertado"] = round(test["bajo"].mean(), 4)
    filas.append(fila)
res = pd.DataFrame(filas)
res.to_csv(C.TABLAS / "protocolo_lineas_base.csv", index=False, encoding="utf-8-sig")
print(res.to_string())
