"""03 - Comparacion de metodos de imputacion (CRISP-DM: Preparacion de los datos).

Variables a imputar : estrato, educ_madre, educ_padre (categoricas con nulos).
Metodos comparados  :
    A  Moda global
    B  Moda por institucion (inst_cod), con respaldo en la moda global
    C1 Modelo (HistGradientBoosting) - clase mas probable
    C2 Modelo (HistGradientBoosting) - muestreo estocastico segun predict_proba
Protocolo (declarado ANTES de ver resultados, ver bitacora D-10):
    se enmascara al azar el 10 % de los valores conocidos (3 repeticiones), se imputan y se compara
    contra el valor real con: accuracy, F1 macro, distancia de Jensen-Shannon entre la distribucion
    original y la imputada, y variacion de la correlacion de Spearman con puntaje_global.
    Criterio de seleccion: menor JSD (conserva mejor la distribucion); empate (<0,0005) -> mayor accuracy.
    El puntaje NO se usa como predictor, para no inflar artificialmente la asociacion estrato-puntaje.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

import config as C

df = pd.read_parquet(C.LIMPIO)
FEATURES = ["inst_origen", "inst_caracter", "depto_inst", "region_inst", "modalidad", "matricula",
            "horas_trabajo", "pago_beca", "pago_credito", "pago_padres", "pago_propio", "genero", "nucleo",
            "tiene_automovil", "tiene_lavadora", "tiene_computador", "tiene_internet", "edad", "anio",
            "estrato", "educ_madre", "educ_padre"]
TARGETS = {"estrato": C.ORD_ESTRATO, "educ_madre": C.ORD_EDUCACION, "educ_padre": C.ORD_EDUCACION}
MAX_TRAIN = 300_000
rng_global = np.random.default_rng(C.SEMILLA)


def matriz(d, target):
    """Predictores sin la propia variable objetivo; categoricas como dtype category."""
    X = d[[f for f in FEATURES if f != target]].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    return X


def modelo():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, categorical_features="from_dtype",
                                          random_state=C.SEMILLA)


def ajustar_predecir_proba(train, test, target):
    if len(train) > MAX_TRAIN:
        train = train.sample(MAX_TRAIN, random_state=C.SEMILLA)
    X_all = matriz(pd.concat([train, test]), target)
    Xtr, Xte = X_all.iloc[:len(train)], X_all.iloc[len(train):]
    m = modelo().fit(Xtr, train[target].astype(str))
    return m.classes_, m.predict_proba(Xte)


def muestrear(clases, proba, rng):
    acum = proba.cumsum(axis=1)
    u = rng.random((len(proba), 1))
    return clases[(u > acum).sum(axis=1).clip(0, len(clases) - 1)]


def metodos(train, test, target, rng):
    """Devuelve {metodo: serie imputada para las filas de test}."""
    moda = train[target].mode().iat[0]
    por_inst = train.groupby("inst_cod")[target].agg(lambda s: s.mode().iat[0])
    clases, proba = ajustar_predecir_proba(train, test, target)
    return {
        "A. Moda global": pd.Series(moda, index=test.index),
        "B. Moda por institución": test["inst_cod"].map(por_inst).fillna(moda),
        "C1. Modelo (clase más probable)": pd.Series(clases[proba.argmax(axis=1)], index=test.index),
        "C2. Modelo (muestreo estocástico)": pd.Series(muestrear(clases, proba, rng), index=test.index),
    }


def jsd(a, b):
    cats = sorted(set(a.unique()) | set(b.unique()))
    pa = a.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(cats, fill_value=0)
    return float(jensenshannon(pa, pb, base=2) ** 2)  # divergencia (JSD^2), en [0,1]


def rho(cat, ordmap, puntaje):
    x = cat.map(ordmap)
    ok = x.notna() & puntaje.notna()
    return float(spearmanr(x[ok], puntaje[ok])[0])


# ------------------------------------------------------------------ Experimento
filas = []
for target, ordmap in TARGETS.items():
    conocido = df[df[target].notna()]
    rho_orig = rho(conocido[target], ordmap, conocido["puntaje_global"])
    for rep in range(3):
        rng = np.random.default_rng(C.SEMILLA + rep)
        mascara = rng.random(len(conocido)) < 0.10
        train, test = conocido[~mascara].copy(), conocido[mascara].copy()
        test_mask = test.copy()
        test_mask[target] = np.nan
        imputados = metodos(train, test_mask, target, rng)
        for nombre, imp in imputados.items():
            completo = pd.concat([train[target], imp])  # distribucion tras imputar
            pts = pd.concat([train["puntaje_global"], test["puntaje_global"]])
            filas.append({
                "variable": target, "repeticion": rep + 1, "metodo": nombre,
                "n_enmascarados": len(test),
                "accuracy": accuracy_score(test[target], imp),
                "f1_macro": f1_score(test[target], imp, average="macro", zero_division=0),
                "jsd_distribucion": jsd(conocido[target], completo),
                "delta_rho_spearman": rho(completo, ordmap, pts) - rho_orig,
            })
    print(f"{target}: experimento terminado")

exp = pd.DataFrame(filas)
exp.to_csv(C.TABLAS / "imputacion_experimento_detalle.csv", index=False, encoding="utf-8-sig")
resumen = (exp.groupby(["variable", "metodo"])
           .agg(accuracy=("accuracy", "mean"), f1_macro=("f1_macro", "mean"),
                jsd_distribucion=("jsd_distribucion", "mean"),
                abs_delta_rho=("delta_rho_spearman", lambda s: s.abs().mean()))
           .reset_index())
resumen.to_csv(C.TABLAS / "imputacion_comparacion.csv", index=False, encoding="utf-8-sig")
print(resumen.round(5).to_string())

# ------------------------------------------------- Seleccion segun criterio declarado
def ganador(g):
    g = g.sort_values("jsd_distribucion")
    mejor = g.iloc[0]
    empatados = g[g["jsd_distribucion"] - mejor["jsd_distribucion"] < 0.0005]
    return empatados.sort_values("accuracy", ascending=False).iloc[0]["metodo"]


metodo_elegido = resumen.groupby("variable").apply(ganador, include_groups=False)
print("\nMétodo elegido por variable:\n", metodo_elegido.to_string())
metodo_elegido.rename("metodo_elegido").to_csv(C.TABLAS / "imputacion_metodo_elegido.csv", encoding="utf-8-sig")

# ------------------------------------------------- Aplicacion a los nulos reales
out = df.copy()
distribs = []
for target, ordmap in TARGETS.items():
    nulos = out[target].isna()
    out[f"{target}_imputado"] = nulos
    if nulos.sum() == 0:
        continue
    antes = out[target].value_counts(normalize=True)
    conocido, faltante = out.loc[~nulos], out.loc[nulos].copy()
    rng = np.random.default_rng(C.SEMILLA)
    nombre = metodo_elegido[target]
    imp = metodos(conocido, faltante, target, rng)[nombre]
    out.loc[nulos, target] = imp.astype(str).values
    despues = out[target].value_counts(normalize=True)
    for cat in sorted(set(antes.index) | set(despues.index)):
        distribs.append({"variable": target, "categoria": cat, "pct_antes": round(100 * antes.get(cat, 0), 2),
                         "pct_despues": round(100 * despues.get(cat, 0), 2)})
    print(f"{target}: {int(nulos.sum())} nulos imputados con '{nombre}'")

out["estrato_num"] = out["estrato"].map(C.ORD_ESTRATO)
out["educ_madre_ord"] = out["educ_madre"].map(C.ORD_EDUCACION)
out["educ_padre_ord"] = out["educ_padre"].map(C.ORD_EDUCACION)
pd.DataFrame(distribs).to_csv(C.TABLAS / "imputacion_distribucion_antes_despues.csv", index=False, encoding="utf-8-sig")
out.to_parquet(C.IMPUTADO, index=False)
print("\nGuardado:", C.IMPUTADO)
