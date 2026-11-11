"""09 - Inferencia estadistica, ablacion, equidad y explicabilidad (SHAP) del modelo final.

Entradas : reports/tables/mod_*.csv (script 08), data/processed/pred_test_2022.parquet, data/processed/modelos/*.joblib
Salidas  : reports/tables/mod_*.csv y reports/figures/mod_*.png
Reglas   : el mejor modelo se elige SOLO con la validacion cruzada (nunca con la prueba 2022); la prueba se usa para
           la evaluacion final y los analisis descriptivos posteriores (ablacion, equidad, SHAP).
"""
import json
import sys
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config as C
import modelado_comun as M
import modelado_modelos as MM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")
TAB, FIG = C.TABLAS, C.FIGURAS
SEM = C.SEMILLA
rng = np.random.default_rng(SEM)
HP = json.loads((TAB / "mod_hiperparametros.json").read_text(encoding="utf-8"))
cv = {e: pd.read_csv(TAB / f"mod_cv_pliegues_{e}.csv") for e in ("conocidas", "grupos")}
test_m = pd.read_csv(TAB / "mod_test_2022_metricas.csv")
pred = pd.read_parquet(C.RAIZ / "data" / "processed" / "pred_test_2022.parquet")
train, test = M.cargar_datos()
UMBRAL = M.umbral_riesgo(train[M.OBJETIVO])
B3 = "B3. Media por institución"

# ------------------------------------------------------------------ 1. Resumen de la validacion cruzada
res = []
for e, d in cv.items():
    g = d.groupby(["modelo", "tarea"]).agg(
        RMSE_media=("RMSE", "mean"), RMSE_sd=("RMSE", "std"), MAE_media=("MAE", "mean"), R2_media=("R2", "mean"),
        AUC_media=("AUC", "mean"), AUC_sd=("AUC", "std"), recall25_media=("recall25", "mean"),
        precision25_media=("precision25", "mean"), seg_media=("segundos", "mean")).reset_index()
    g.insert(0, "esquema", e)
    res.append(g)
resumen = pd.concat(res)
resumen.round(4).to_csv(TAB / "mod_cv_resumen.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 2. Seleccion del mejor modelo (solo CV, esquema 'conocidas')
cc = resumen[resumen.esquema == "conocidas"].set_index("modelo")
mejor_ens = cc.loc[MM.ENSAMBLES_REG, "RMSE_media"].idxmin()
clfs = [m for m in HP if HP[m]["tarea"] == "clf"]
mejor_clf = cc.loc[clfs, "AUC_media"].idxmax()
print("Mejor ensamble (RMSE CV):", mejor_ens, "| mejor clasificador (AUC CV):", mejor_clf, flush=True)
pd.DataFrame([{"criterio": "Mejor modelo de ensamble (regresión): menor RMSE medio en validación cruzada", "modelo": mejor_ens,
               "RMSE_cv": cc.loc[mejor_ens, "RMSE_media"]},
              {"criterio": "Mejor clasificador de bajo desempeño: mayor AUC medio en validación cruzada", "modelo": mejor_clf,
               "AUC_cv": cc.loc[mejor_clf, "AUC_media"]}]).to_csv(TAB / "mod_modelo_elegido.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 3. Pruebas de hipotesis
# 3a. Entre pliegues (CV 'conocidas'): diferencia de RMSE por pliegue frente a la linea base institucional
def pareadas(d, a, b, k=5):
    x = d[d.modelo == a].set_index("pliegue")["RMSE"]
    y = d[d.modelo == b].set_index("pliegue")["RMSE"]
    diff = (x - y).dropna()
    n_tr, n_va = d[d.modelo == a]["n_train"].iloc[0], d[d.modelo == a]["n_val"].iloc[0]
    t, p = stats.ttest_rel(x.loc[diff.index], y.loc[diff.index])
    nb_t = diff.mean() / np.sqrt((1 / len(diff) + n_va / n_tr) * diff.var(ddof=1))  # Nadeau-Bengio
    nb_p = 2 * stats.t.sf(abs(nb_t), len(diff) - 1)
    try:
        w = stats.wilcoxon(diff).pvalue
    except ValueError:
        w = np.nan
    return {"comparacion": f"{a} vs {b}", "dif_RMSE_media": diff.mean(), "dif_RMSE_sd": diff.std(ddof=1),
            "t_pareada": t, "p_t_pareada": p, "t_Nadeau_Bengio": nb_t, "p_Nadeau_Bengio": nb_p, "p_Wilcoxon": w,
            "pliegues_a_favor": int((diff < 0).sum()), "n_pliegues": len(diff)}


filas = []
dk = cv["conocidas"]
for m in MM.ENSAMBLES_REG + ["Ridge"]:
    filas.append({"esquema": "conocidas", **pareadas(dk, m, B3)})
for m in MM.ENSAMBLES_REG:
    if m != mejor_ens:
        filas.append({"esquema": "conocidas", **pareadas(dk, mejor_ens, m)})
filas.append({"esquema": "conocidas", **pareadas(dk, mejor_ens, "Ridge")})
sig_cv = pd.DataFrame(filas)
sig_cv["p_Nadeau_Bengio_Holm"] = multipletests(sig_cv["p_Nadeau_Bengio"].values, method="holm")[1]
sig_cv.round(5).to_csv(TAB / "mod_significancia_cv.csv", index=False, encoding="utf-8-sig")

# 3b. En la prueba 2022: diferencia de perdida cuadratica por estudiante, error estandar agrupado por institucion
y = pred["y"].values
inst = pred["inst_cod"].values
se_m, se_b = (y - pred[mejor_ens].values) ** 2, (y - pred[B3].values) ** 2
codes = pd.factorize(inst)[0]
G = codes.max() + 1
n_i = np.bincount(codes, minlength=G).astype(float)
sm_i, sb_i = np.bincount(codes, weights=se_m, minlength=G), np.bincount(codes, weights=se_b, minlength=G)
rmse_m, rmse_b = np.sqrt(se_m.mean()), np.sqrt(se_b.mean())
dmse = se_m - se_b
ols = sm.OLS(dmse, np.ones(len(dmse))).fit(cov_type="cluster", cov_kwds={"groups": codes})
B_RMSE = 2000
boot = np.empty((B_RMSE, 2))
for b in range(B_RMSE):
    idx = rng.integers(0, G, G)
    nn = n_i[idx].sum()
    rm, rb = np.sqrt(sm_i[idx].sum() / nn), np.sqrt(sb_i[idx].sum() / nn)
    boot[b] = (rm - rb, 100 * (rm - rb) / rb)
wil = stats.wilcoxon(dmse[rng.choice(len(dmse), 50000, replace=False)]).pvalue
# AUC y recall@25 con bootstrap por institucion
bajo = pred["bajo"].values
orden_inst = [np.where(codes == g)[0] for g in range(G)]
from sklearn.metrics import roc_auc_score

B_AUC = 300
bs = np.empty((B_AUC, 4))
for b in range(B_AUC):
    idx = np.concatenate([orden_inst[g] for g in rng.integers(0, G, G)])
    yy, pm, pb = bajo[idx], -pred[mejor_ens].values[idx], -pred[B3].values[idx]
    rm_, _ = M.alerta_top(pm, yy)
    rb_, _ = M.alerta_top(pb, yy)
    bs[b] = (roc_auc_score(yy, pm) - roc_auc_score(yy, pb), rm_ - rb_, roc_auc_score(yy, pm), rb_)
auc_m, auc_b = roc_auc_score(bajo, -pred[mejor_ens]), roc_auc_score(bajo, -pred[B3])
rec_m, _ = M.alerta_top(-pred[mejor_ens].values, bajo)
rec_b, _ = M.alerta_top(-pred[B3].values, bajo)
ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
sig_test = pd.DataFrame([
    {"metrica": "RMSE (puntos)", "modelo": mejor_ens, "linea_base_B3": rmse_b, "valor_modelo": rmse_m, "diferencia": rmse_m - rmse_b,
     "IC95_inf": ci(boot[:, 0])[0], "IC95_sup": ci(boot[:, 0])[1],
     "mejora_relativa_pct": 100 * (rmse_b - rmse_m) / rmse_b, "p_valor": float(ols.pvalues[0]),
     "prueba": "diferencia de pérdida cuadrática por estudiante, error estándar agrupado por institución; IC por bootstrap de instituciones (2.000)"},
    {"metrica": "AUC-ROC", "modelo": mejor_ens, "linea_base_B3": auc_b, "valor_modelo": auc_m, "diferencia": auc_m - auc_b,
     "IC95_inf": ci(bs[:, 0])[0], "IC95_sup": ci(bs[:, 0])[1], "mejora_relativa_pct": 100 * (auc_m - auc_b) / auc_b,
     "p_valor": np.nan, "prueba": "IC por bootstrap de instituciones (300)"},
    {"metrica": "Recall al alertar al 25 %", "modelo": mejor_ens, "linea_base_B3": rec_b, "valor_modelo": rec_m, "diferencia": rec_m - rec_b,
     "IC95_inf": ci(bs[:, 1])[0], "IC95_sup": ci(bs[:, 1])[1], "mejora_relativa_pct": 100 * (rec_m - rec_b) / rec_b,
     "p_valor": np.nan, "prueba": "IC por bootstrap de instituciones (300)"}])
sig_test.round(5).to_csv(TAB / "mod_significancia_prueba2022.csv", index=False, encoding="utf-8-sig")
print(sig_test[["metrica", "linea_base_B3", "valor_modelo", "diferencia", "IC95_inf", "IC95_sup", "p_valor"]].round(4).to_string(), flush=True)
IC_RMSE = (ci(boot[:, 0]), ci(boot[:, 1]), float(wil))

# IC bootstrap del RMSE de cada modelo en la prueba (tabla de resultados)
filas = []
for nm in [c for c in test_m.modelo if c in pred.columns and HP.get(c, {}).get("tarea", "reg") == "reg"]:
    se = (y - pred[nm].values) ** 2
    s_i = np.bincount(codes, weights=se, minlength=G)
    bb = [np.sqrt(s_i[ix].sum() / n_i[ix].sum()) for ix in (rng.integers(0, G, G) for _ in range(500))]
    filas.append({"modelo": nm, "RMSE": np.sqrt(se.mean()), "IC95_inf": np.percentile(bb, 2.5), "IC95_sup": np.percentile(bb, 97.5)})
pd.DataFrame(filas).round(4).to_csv(TAB / "mod_test_2022_rmse_ic.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------ 4. Equidad (criterio 5 de la Tabla 2)
def equidad(score_col, nombre):
    riesgo = -pred[score_col].values if HP.get(score_col, {}).get("tarea", "reg") == "reg" else pred[score_col].values
    k = int(round(0.25 * len(pred)))
    alerta = np.zeros(len(pred), bool)
    alerta[np.argsort(-riesgo, kind="stable")[:k]] = True
    rec_g = (alerta & (bajo == 1)).sum() / bajo.sum()
    filas = []
    for var, etiqueta in (("genero", "Género"), ("region_inst", "Región de la institución")):
        for cat, ix in pred.groupby(var).groups.items():
            ix = np.asarray(ix)
            if bajo[ix].sum() == 0:
                continue
            r = (alerta[ix] & (bajo[ix] == 1)).sum() / bajo[ix].sum()
            p = (bajo[ix][alerta[ix]]).mean() if alerta[ix].any() else np.nan
            filas.append({"modelo": nombre, "variable": etiqueta, "grupo": cat, "n": len(ix), "pct_bajo_desempeno": 100 * bajo[ix].mean(),
                          "recall25": r, "dif_vs_global_pp": 100 * (r - rec_g), "precision25": p,
                          "RMSE": np.sqrt(np.mean((y[ix] - pred[score_col].values[ix]) ** 2)) if HP.get(score_col, {}).get("tarea", "reg") == "reg" else np.nan,
                          "error_medio": np.mean(pred[score_col].values[ix] - y[ix]) if HP.get(score_col, {}).get("tarea", "reg") == "reg" else np.nan,
                          "cumple_10pp": abs(100 * (r - rec_g)) <= 10, "recall_global": rec_g})
    return pd.DataFrame(filas)


eq = pd.concat([equidad(mejor_ens, mejor_ens), equidad(mejor_clf, mejor_clf)])
eq.round(4).to_csv(TAB / "mod_equidad.csv", index=False, encoding="utf-8-sig")
print(eq[eq.modelo == mejor_ens][["variable", "grupo", "n", "recall25", "dif_vs_global_pp", "cumple_10pp"]].round(3).to_string(), flush=True)

# ------------------------------------------------------------------ 5. Ablacion
hp = HP[mejor_ens]
y_tr = train[M.OBJETIVO].values
y_te = test[M.OBJETIVO].values
b_te = (y_te < UMBRAL).astype(int)
filas = []


def ajustar_evaluar(etiqueta, grupo, columnas=None, imputador="modelo", selector=None):
    cols = columnas or M.ENTRADAS
    _, A, B = MM.preparar(train[M.ENTRADAS], y_tr, test[M.ENTRADAS], imputador=imputador, columnas=cols)
    pipe = MM.ensamblar(mejor_ens, hp["params"], selector or hp["selector"]).fit(A, y_tr)
    m = M.metricas_regresion(y_te, pipe.predict(B), UMBRAL)
    filas.append({"grupo_ablacion": grupo, "configuracion": etiqueta, "n_variables_entrada": len(cols), **m})
    print(f"  ablación: {etiqueta}: RMSE={m['RMSE']:.3f} AUC={m['AUC']:.3f}", flush=True)
    return A, B, pipe


A_full, B_full, pipe_full = ajustar_evaluar("Modelo completo (referencia)", "Referencia")
for nombre, cols in M.BLOQUES.items():
    resto = [c for c in M.ENTRADAS if c not in cols]
    ajustar_evaluar(f"Sin: {nombre}", "Bloque de variables", columnas=resto)
ajustar_evaluar("Imputación por moda (método A) en vez de C1", "Pipeline", imputador="moda")
ajustar_evaluar("Sin imputación (faltante como categoría)", "Pipeline", imputador="ninguno")
for sel in ("ninguno", "lasso_suave", "lasso_fuerte"):
    if sel != hp["selector"]:
        _, B2 = A_full, B_full
        pipe = MM.ensamblar(mejor_ens, hp["params"], sel).fit(A_full, y_tr)
        m = M.metricas_regresion(y_te, pipe.predict(B_full), UMBRAL)
        filas.append({"grupo_ablacion": "Pipeline", "configuracion": f"Selector de variables: {sel} (elegido: {hp['selector']})",
                      "n_variables_entrada": len(M.ENTRADAS), **m})
ab = pd.DataFrame(filas)
ref = ab.iloc[0]
ab["delta_RMSE"] = ab["RMSE"] - ref["RMSE"]
ab["delta_AUC"] = ab["AUC"] - ref["AUC"]
ab["delta_recall25"] = ab["recall25"] - ref["recall25"]
ab.round(4).to_csv(TAB / "mod_ablacion.csv", index=False, encoding="utf-8-sig")

# variables retenidas por el selector (modelo final)
pre = joblib.load(C.RAIZ / "data" / "processed" / "modelos" / "preparacion_ajustada_en_train.joblib")
nombres = np.array(list(pre.named_steps["prep"].get_feature_names_out()))
sel_step = pipe_full.named_steps["sel"]
soporte = sel_step.get_support() if sel_step != "passthrough" else np.ones(len(nombres), bool)
pd.DataFrame({"variable_transformada": nombres, "variable_original": M.nombres_origen(nombres), "retenida": soporte}
             ).to_csv(TAB / "mod_seleccion_variables.csv", index=False, encoding="utf-8-sig")

# seleccion algoritmica (Lasso) ajustada SOLO con entrenamiento: que conserva y que descarta cada intensidad
from sklearn.base import clone  # noqa: E402

filas_sel = []
for s_nombre in ("lasso_suave", "lasso_fuerte"):
    sop_s = clone(M.SELECTORES[s_nombre]).fit(A_full, y_tr).get_support()
    for n_, o_, r_ in zip(nombres, M.nombres_origen(nombres), sop_s):
        filas_sel.append({"selector": s_nombre, "variable_transformada": n_, "variable_original": o_, "retenida": bool(r_)})
sel_lasso = pd.DataFrame(filas_sel)
sel_lasso.to_csv(TAB / "mod_seleccion_lasso.csv", index=False, encoding="utf-8-sig")
print(sel_lasso.groupby("selector")["retenida"].agg(["sum", "count"]).to_string(), flush=True)

# ------------------------------------------------------------------ 6. SHAP
ETQ = {"inst_cod": "Institución (código)", "nucleo": "Núcleo del programa", "depto_inst": "Departamento de la institución",
       "estrato": "Estrato", "educ_madre": "Educación de la madre", "educ_padre": "Educación del padre", "genero": "Género",
       "inst_caracter": "Carácter académico", "sector": "Sector", "inst_origen": "Naturaleza jurídica",
       "region_inst": "Región de la institución", "modalidad": "Modalidad", "pago_beca": "Pago con beca",
       "pago_credito": "Pago con crédito", "pago_padres": "Pago con recursos de padres", "pago_propio": "Pago con recursos propios",
       "edad": "Edad", "matricula_ord": "Valor de matrícula", "horas_trabajo_ord": "Horas de trabajo",
       "tiene_automovil": "Tiene automóvil", "tiene_lavadora": "Tiene lavadora", "tiene_computador": "Tiene computador",
       "tiene_internet": "Tiene internet", "estrato_num": "Estrato", "educ_madre_ord": "Educación de la madre",
       "educ_padre_ord": "Educación del padre"}
muestra = np.sort(rng.choice(len(B_full), 12000, replace=False))
Xs = B_full.iloc[muestra]
if sel_step != "passthrough":
    Xs = pd.DataFrame(sel_step.transform(Xs), columns=nombres[soporte], index=Xs.index)
estimador = pipe_full.named_steps["est"]
explainer = shap.TreeExplainer(estimador)
sv = explainer.shap_values(Xs)
nom = list(Xs.columns)
orig = [ETQ.get(o, o) for o in M.nombres_origen(nom)]
imp = pd.DataFrame({"variable": nom, "variable_original": orig, "mean_abs_shap": np.abs(sv).mean(axis=0)})
agg = imp.groupby("variable_original")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
agg["pct"] = 100 * agg["mean_abs_shap"] / agg["mean_abs_shap"].sum()
agg.round(4).to_csv(TAB / "mod_shap_importancia.csv", index=False, encoding="utf-8-sig")
imp.sort_values("mean_abs_shap", ascending=False).round(4).to_csv(TAB / "mod_shap_importancia_detalle.csv", index=False, encoding="utf-8-sig")

import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.95)
fig, ax = plt.subplots(figsize=(8, 5.2))
top = agg.head(15).iloc[::-1]
ax.barh(top["variable_original"], top["mean_abs_shap"], color="#2E5C8A")
ax.set(xlabel="Importancia media |SHAP| (puntos de puntaje global)", title=f"Importancia de las variables ({mejor_ens})")
fig.tight_layout()
fig.savefig(FIG / "mod_shap_importancia.png", dpi=160, bbox_inches="tight")
plt.close(fig)

Xs_leg = Xs.copy()
Xs_leg.columns = [f"{ETQ.get(o, o)} | {n.split('__', 1)[-1][len(o) + 1:] if n.split('__', 1)[-1].startswith(o + '_') else ''}".rstrip(" |") for n, o in zip(nom, M.nombres_origen(nom))]
plt.figure(figsize=(8.5, 6.2))
shap.summary_plot(sv, Xs_leg, max_display=15, show=False, plot_size=None)
plt.title("Valores SHAP por estudiante (muestra de 12.000 de la prueba 2022)", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "mod_shap_resumen.png", dpi=160, bbox_inches="tight")
plt.close()

# dependencia de las tres variables numericas / codificadas mas importantes
num_top = [n for n in imp.sort_values("mean_abs_shap", ascending=False)["variable"] if n.startswith(("te__", "num__")) and "missingindicator" not in n][:3]
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, n in zip(ax, num_top):
    j = nom.index(n)
    a.scatter(Xs[n], sv[:, j], s=3, alpha=0.25, color="#2E5C8A")
    o = M.nombres_origen([n])[0]
    a.set(xlabel=f"{ETQ.get(o, o)} (valor transformado)", ylabel="SHAP (puntos)", title=ETQ.get(o, o))
fig.tight_layout()
fig.savefig(FIG / "mod_shap_dependencia.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 7. Figuras de resultados
rmse_cv = cv["conocidas"][cv["conocidas"].tarea != "clasificacion"]
orden = rmse_cv.groupby("modelo")["RMSE"].mean().sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for a, e, tit in zip(ax, ("conocidas", "grupos"), ("Instituciones conocidas (estratificada)", "Instituciones nuevas (agrupada)")):
    d = cv[e][cv[e].tarea != "clasificacion"]
    sns.boxplot(data=d, y="modelo", x="RMSE", order=orden, color="#9DB7D5", ax=a)
    sns.stripplot(data=d, y="modelo", x="RMSE", order=orden, color="#1F3A5F", size=4, ax=a)
    a.set(title=tit, xlabel="RMSE por pliegue (puntos)", ylabel="")
fig.tight_layout()
fig.savefig(FIG / "mod_cv_rmse_pliegues.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ 8. Criterios de exito
ens_t = test_m.set_index("modelo").loc[mejor_ens]
clf_t = test_m.set_index("modelo").loc[mejor_clf]
b3_t = test_m.set_index("modelo").loc[B3]
rmse_val = cc.loc[mejor_ens, "RMSE_media"]
rmse_val_g = resumen[(resumen.esquema == "grupos") & (resumen.modelo == mejor_ens)]["RMSE_media"].iloc[0]
eq_m = eq[eq.modelo == mejor_ens]
eq_gen, eq_reg = eq_m[eq_m.variable == "Género"], eq_m[eq_m.variable == "Región de la institución"]
big = eq_reg[eq_reg.n >= 1000]
crit = pd.DataFrame([
    {"n": 1, "criterio": "RMSE ≤ 19,4 (≥ 5 % mejor que la mejor línea base)", "resultado": f"{ens_t.RMSE:.2f} (B3 = {b3_t.RMSE:.2f}; mejora {100 * (b3_t.RMSE - ens_t.RMSE) / b3_t.RMSE:.1f} %)", "cumple": bool(ens_t.RMSE <= 19.4)},
    {"n": 2, "criterio": "AUC-ROC ≥ 0,75", "resultado": f"{ens_t.AUC:.3f} (ensamble de regresión); {clf_t.AUC:.3f} ({mejor_clf})", "cumple": bool(max(ens_t.AUC, clf_t.AUC) >= 0.75)},
    {"n": 3, "criterio": "Recall ≥ 0,50 al alertar al 25 %", "resultado": f"{ens_t.recall25:.3f} (ensamble); {clf_t.recall25:.3f} ({mejor_clf})", "cumple": bool(max(ens_t.recall25, clf_t.recall25) >= 0.50)},
    {"n": 4, "criterio": "|RMSE de validación − RMSE de prueba| ≤ 10 % del de validación",
     "resultado": f"validación {rmse_val:.2f} vs prueba {ens_t.RMSE:.2f}: {100 * abs(rmse_val - ens_t.RMSE) / rmse_val:.1f} % (instituciones nuevas: {rmse_val_g:.2f}, {100 * abs(rmse_val_g - ens_t.RMSE) / rmse_val_g:.1f} %)",
     "cumple": bool(abs(rmse_val - ens_t.RMSE) / rmse_val <= 0.10)},
    {"n": 5, "criterio": "Equidad: recall por género y región a ±10 pp del global",
     "resultado": f"género: máx {eq_gen.dif_vs_global_pp.abs().max():.1f} pp; regiones con n ≥ 1.000: máx {big.dif_vs_global_pp.abs().max():.1f} pp; todas las regiones: máx {eq_reg.dif_vs_global_pp.abs().max():.1f} pp",
     "cumple": bool(eq_gen.dif_vs_global_pp.abs().max() <= 10 and big.dif_vs_global_pp.abs().max() <= 10)},
    {"n": 6, "criterio": "Factores explicables y coherentes con el EDA", "resultado": "Top-5 SHAP: " + ", ".join(agg.head(5)["variable_original"]), "cumple": None},
])
crit.to_csv(TAB / "mod_criterios_exito.csv", index=False, encoding="utf-8-sig")
print(crit.to_string(), flush=True)
print("Listo.", flush=True)
