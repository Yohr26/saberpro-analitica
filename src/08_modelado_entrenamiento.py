"""08 - Modelado: bateria de modelos supervisados, optimizacion de hiperparametros y validacion cruzada.

Cumple el protocolo de docs/protocolo_evaluacion.md:
  * Entrenamiento 2020-2021 (467.356 registros); prueba 2022 (193.356), usada UNA vez al final.
  * Imputacion C1, codificaciones y seleccion de variables se ajustan solo con los datos de entrenamiento
    (o con el pliegue de entrenamiento) dentro de un sklearn Pipeline (ver modelado_comun.py).
  * Dos esquemas de validacion cruzada de 5 pliegues (ver bitacora D-19):
      - "conocidas": estratificado por institucion (replica el uso real: el 99,99 % de los estudiantes de 2022 pertenece
        a instituciones vistas en el entrenamiento);
      - "grupos": agrupado por institucion (generalizacion a instituciones nuevas, tal como define el protocolo).
Salidas: reports/tables/mod_*.csv|json y data/processed/{pred_test_2022.parquet, modelos/*.joblib}

Uso:  python src/08_modelado_entrenamiento.py            (completo)
      python src/08_modelado_entrenamiento.py --rapido   (humo: submuestra y pocas iteraciones)
"""
import json
import sys
import time
import warnings


import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, StratifiedKFold

import config as C
import modelado_comun as M
import modelado_modelos as MM
from modelado_modelos import ensamblar, etiquetas_estratificacion, pliegues, predecir, preparar

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
RAPIDO = "--rapido" in sys.argv
SEM = C.SEMILLA
TAB = C.TABLAS
TAB.mkdir(parents=True, exist_ok=True)
(C.RAIZ / "data" / "processed" / "modelos").mkdir(parents=True, exist_ok=True)
SEL = MM.SEL
N_ITER_REG, N_ITER_CLF = (2, 2) if RAPIDO else (8, 6)
N_TUNE = 15_000 if RAPIDO else 120_000

# Bateria de modelos y espacios de busqueda: ver modelado_modelos.py
MODELOS = MM.MODELOS
if RAPIDO:
    MODELOS = {k: v for k, v in MODELOS.items() if k in ("Ridge", "LightGBM", "LightGBM (clasif.)")}


def evaluar(tarea, y, bajo, salida, umbral):
    return M.metricas_regresion(y, salida, umbral) if tarea == "reg" else M.metricas_clasificacion(bajo, salida)




# --------------------------------------------------------------- datos
train, test = M.cargar_datos()
if RAPIDO:
    train = train.sample(40_000, random_state=SEM).reset_index(drop=True)
    test = test.sample(10_000, random_state=SEM).reset_index(drop=True)
print(f"Entrenamiento: {len(train):,} | Prueba: {len(test):,}", flush=True)
UMBRAL = M.umbral_riesgo(train[M.OBJETIVO])
y_tr = train[M.OBJETIVO].values
bajo_tr = (y_tr < UMBRAL).astype(int)
Xtr_all = train[M.ENTRADAS]

# --------------------------------------------------------------- A. optimizacion de hiperparametros
ruta_hp = TAB / "mod_hiperparametros.json"
if ruta_hp.exists() and not RAPIDO:
    HP = json.loads(ruta_hp.read_text(encoding="utf-8"))
else:
    t0 = time.time()
    sub = train.sample(min(N_TUNE, len(train)), random_state=SEM).reset_index(drop=True)
    ysub = sub[M.OBJETIVO].values
    bsub = (ysub < UMBRAL).astype(int)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEM).split(sub, etiquetas_estratificacion(sub)))
    prepped = []
    for tr_i, va_i in folds:
        _, A, B = preparar(sub.loc[tr_i, M.ENTRADAS], ysub[tr_i], sub.loc[va_i, M.ENTRADAS])
        prepped.append((A, B, tr_i, va_i))
    print(f"Preparacion de {len(folds)} pliegues de ajuste: {time.time() - t0:.0f}s", flush=True)
    HP, filas = {}, []
    for nombre, (_, espacio, tarea) in MODELOS.items():
        n_iter = N_ITER_REG if tarea == "reg" else N_ITER_CLF
        espacio = {**espacio, "__sel": SEL}
        mejor = (np.inf, None, None)
        for i, cand in enumerate(ParameterSampler(espacio, n_iter, random_state=SEM)):
            sel = cand.pop("__sel")
            pt = time.time()
            puntajes = []
            for A, B, tr_i, va_i in prepped:
                pipe = ensamblar(nombre, cand, sel).fit(A, ysub[tr_i] if tarea == "reg" else bsub[tr_i])
                s = predecir(pipe, B, tarea)
                puntajes.append(np.sqrt(np.mean((ysub[va_i] - s) ** 2)) if tarea == "reg" else -M.metricas_clasificacion(bsub[va_i], s)["AUC"])
            val = float(np.mean(puntajes))
            filas.append({"modelo": nombre, "iteracion": i + 1, "selector": sel, "params": json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in cand.items()}, default=float),
                          "objetivo_cv": val, "metrica": "RMSE" if tarea == "reg" else "-AUC", "segundos": round(time.time() - pt, 1)})
            if val < mejor[0]:
                mejor = (val, dict(cand), sel)
        HP[nombre] = {"params": {k: (float(v) if isinstance(v, (np.floating,)) else (int(v) if isinstance(v, np.integer) else v)) for k, v in mejor[1].items()},
                      "selector": mejor[2], "objetivo_cv": mejor[0], "tarea": MODELOS[nombre][2]}
        print(f"  {nombre}: mejor {mejor[0]:.4f} con selector={mejor[2]} {mejor[1]}", flush=True)
    pd.DataFrame(filas).to_csv(TAB / f"mod_busqueda_hiperparametros{'_rapido' if RAPIDO else ''}.csv", index=False, encoding="utf-8-sig")
    if not RAPIDO:
        ruta_hp.write_text(json.dumps(HP, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Optimizacion de hiperparametros: {(time.time() - t0) / 60:.1f} min", flush=True)

# --------------------------------------------------------------- B. validacion cruzada por pliegue
sufijo = "_rapido" if RAPIDO else ""
filas = []
for esquema in ("conocidas", "grupos"):
    ruta = TAB / f"mod_cv_pliegues_{esquema}{sufijo}.csv"
    if ruta.exists() and not RAPIDO:
        continue
    t0 = time.time()
    out = []
    for k, (tr_i, va_i) in enumerate(pliegues(train, esquema), 1):
        Xa, Xb = Xtr_all.iloc[tr_i], Xtr_all.iloc[va_i]
        ya, yb = y_tr[tr_i], y_tr[va_i]
        ba, bb = bajo_tr[tr_i], bajo_tr[va_i]
        _, A, B = preparar(Xa, ya, Xb)
        # lineas base (se ajustan con el pliegue de entrenamiento)
        g = ya.mean()
        est_mean = pd.Series(ya).groupby(Xa["estrato"].fillna("NA").values).mean()
        ins_mean = pd.Series(ya).groupby(Xa["inst_cod"].values).mean()
        bases = {"B0. Media global": np.full(len(yb), g),
                 "B1. Media por estrato": Xb["estrato"].fillna("NA").map(est_mean).fillna(g).values,
                 "B3. Media por institución": Xb["inst_cod"].map(ins_mean).fillna(g).values}
        for nm, p in bases.items():
            out.append({"esquema": esquema, "pliegue": k, "modelo": nm, "tarea": "linea_base", "n_train": len(tr_i), "n_val": len(va_i),
                        **M.metricas_regresion(yb, p, UMBRAL), "segundos": 0.0})
        for nombre, hp in HP.items():
            tarea = hp["tarea"]
            pt = time.time()
            pipe = ensamblar(nombre, hp["params"], hp["selector"]).fit(A, ya if tarea == "reg" else ba)
            s = predecir(pipe, B, tarea)
            m = M.metricas_regresion(yb, s, UMBRAL) if tarea == "reg" else M.metricas_clasificacion(bb, s)
            out.append({"esquema": esquema, "pliegue": k, "modelo": nombre, "tarea": "regresion" if tarea == "reg" else "clasificacion",
                        "n_train": len(tr_i), "n_val": len(va_i), **m, "segundos": round(time.time() - pt, 1)})
        print(f"  CV {esquema}: pliegue {k}/5 listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
        pd.DataFrame(out).to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"CV {esquema}: {(time.time() - t0) / 60:.1f} min", flush=True)

# --------------------------------------------------------------- C. ajuste final y prueba 2022 (una sola vez)
t0 = time.time()
pre, A, B = preparar(Xtr_all, y_tr, test[M.ENTRADAS])
y_te = test[M.OBJETIVO].values
bajo_te = (y_te < UMBRAL).astype(int)
pred = test[["id_anonimo", "inst_cod", "genero", "region_inst", "estrato", "modalidad", "inst_caracter", "sector"]].copy()
pred["y"], pred["bajo"] = y_te, bajo_te
g = y_tr.mean()
pred["B0. Media global"] = g
pred["B1. Media por estrato"] = test["estrato"].fillna("NA").map(pd.Series(y_tr).groupby(train["estrato"].fillna("NA").values).mean()).fillna(g).values
pred["B3. Media por institución"] = test["inst_cod"].map(pd.Series(y_tr).groupby(train["inst_cod"].values).mean()).fillna(g).values
res = []
for nm in ("B0. Media global", "B1. Media por estrato", "B3. Media por institución"):
    res.append({"modelo": nm, "tarea": "linea_base", **M.metricas_regresion(y_te, pred[nm], UMBRAL)})
for nombre, hp in HP.items():
    tarea = hp["tarea"]
    pipe = ensamblar(nombre, hp["params"], hp["selector"]).fit(A, y_tr if tarea == "reg" else bajo_tr)
    s = predecir(pipe, B, tarea)
    pred[nombre] = s
    m = M.metricas_regresion(y_te, s, UMBRAL) if tarea == "reg" else M.metricas_clasificacion(bajo_te, s)
    res.append({"modelo": nombre, "tarea": "regresion" if tarea == "reg" else "clasificacion", **m})
    print(f"  prueba 2022 - {nombre}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    if not RAPIDO:
        joblib.dump(pipe, C.RAIZ / "data" / "processed" / "modelos" / f"{nombre.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.joblib")
pd.DataFrame(res).to_csv(TAB / f"mod_test_2022_metricas{sufijo}.csv", index=False, encoding="utf-8-sig")
pred.to_parquet(C.RAIZ / "data" / "processed" / f"pred_test_2022{sufijo}.parquet", index=False)
if not RAPIDO:
    joblib.dump(pre, C.RAIZ / "data" / "processed" / "modelos" / "preparacion_ajustada_en_train.joblib")
print(f"Ajuste final y prueba: {(time.time() - t0) / 60:.1f} min", flush=True)
