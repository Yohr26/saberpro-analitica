"""04 - Analisis exploratorio multivariado (CRISP-DM: Comprension / Preparacion de los datos).

Variable objetivo: puntaje_global (0-300). Se cruza con region, sector, caracter institucional, genero,
estrato, educacion de los padres, modalidad, matricula, conectividad y anio.
Salidas: reports/tables/eda_*.csv y reports/figures/fig_*.png
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats

import config as C

sns.set_theme(style="whitegrid", font_scale=0.95)
AZUL, GRIS, NARANJA = "#2E5C8A", "#8A8F98", "#D9822B"
C.FIGURAS.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(C.IMPUTADO)
d = df[df["puntaje_global"].notna()].copy()  # analisis sobre quienes tienen los 5 modulos validos
d["genero"] = d["genero"].map({"F": "Mujer", "M": "Hombre"})
OBJ = "puntaje_global"
ORD_ESTRATO = [f"Estrato {i}" for i in range(1, 7)] + ["Sin Estrato"]
ORD_EDU = list(C.ORD_EDUCACION) + ["No sabe", "No Aplica"]
print("Registros analizados (5 modulos validos):", len(d))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(C.FIGURAS / nombre, dpi=160, bbox_inches="tight")
    plt.close(fig)


def cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


# ===================================================== 1. Variable objetivo
resumen = d[OBJ].describe(percentiles=[.05, .25, .5, .75, .95]).round(2)
resumen.to_csv(C.TABLAS / "eda_objetivo_resumen.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(d[OBJ], bins=60, color=AZUL, ax=ax[0])
ax[0].axvline(d[OBJ].mean(), color=NARANJA, ls="--", label=f"Media {d[OBJ].mean():.1f}")
ax[0].set(xlabel="Puntaje global (promedio de 5 módulos)", ylabel="Estudiantes",
          title="Distribución de la variable objetivo")
ax[0].legend()
sns.boxplot(data=d, x="anio", y=OBJ, color=AZUL, showfliers=False, ax=ax[1])
ax[1].set(xlabel="Año", ylabel="Puntaje global", title="Puntaje global por año")
guardar(fig, "fig_01_distribucion_objetivo.png")

# ============================================ 2. Modulos por anio (tendencia)
tm = df.groupby("anio")[C.PUNT].mean().rename(columns=C.MODULOS).round(2)
tm["Puntaje global"] = d.groupby("anio")[OBJ].mean().round(2)
tm["n_registros"] = df.groupby("anio").size()
tm.to_csv(C.TABLAS / "eda_modulos_por_anio.csv", encoding="utf-8-sig")
m = (df.melt(id_vars="anio", value_vars=C.PUNT, var_name="modulo", value_name="puntaje")
     .dropna().assign(modulo=lambda x: x["modulo"].map(C.MODULOS)))
medias = m.groupby(["modulo", "anio"])["puntaje"].mean().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=medias, x="modulo", y="puntaje", hue="anio", palette=["#9DB7D5", "#5C86B3", AZUL], ax=ax)
ax.set(xlabel="", ylabel="Puntaje promedio (0-300)", title="Puntaje promedio por módulo y año")
ax.set_ylim(100, 175)
plt.setp(ax.get_xticklabels(), rotation=15)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.0f", fontsize=7, padding=2)
guardar(fig, "fig_02_modulos_por_anio.png")

# ====================== 3. Comparacion entre modulos (H2a: razonamiento mas bajo?)
fil = []
for a in C.PUNT:
    fil.append({"modulo": C.MODULOS[a], "media": df[a].mean(), "mediana": df[a].median(),
                "desv": df[a].std(), "p25": df[a].quantile(.25), "p75": df[a].quantile(.75)})
pd.DataFrame(fil).round(2).to_csv(C.TABLAS / "eda_modulos_resumen.csv", index=False, encoding="utf-8-sig")
pares = []
for a in C.PUNT:
    if a == "pt_razonamiento":
        continue
    x = d[["pt_razonamiento", a]]
    t = stats.ttest_rel(x["pt_razonamiento"], x[a])
    pares.append({"comparacion": f"Razonamiento vs {C.MODULOS[a]}",
                  "diferencia_media": round(x["pt_razonamiento"].mean() - x[a].mean(), 2),
                  "d_cohen_pareado": round((x["pt_razonamiento"] - x[a]).mean() / (x["pt_razonamiento"] - x[a]).std(), 3),
                  "p_valor": t.pvalue})
pd.DataFrame(pares).to_csv(C.TABLAS / "eda_razonamiento_vs_otros.csv", index=False, encoding="utf-8-sig")

# ====================================== 4. Por grupo: tabla + pruebas Kruskal
GRUPOS = {"region_inst": "Región de la institución", "sector": "Sector", "inst_origen": "Naturaleza jurídica",
          "inst_caracter": "Carácter académico", "genero": "Género", "estrato": "Estrato",
          "modalidad": "Modalidad", "educ_madre": "Educación de la madre", "educ_padre": "Educación del padre",
          "matricula": "Valor de matrícula", "horas_trabajo": "Horas de trabajo", "anio": "Año",
          "tiene_internet": "Tiene internet", "tiene_computador": "Tiene computador",
          "pago_beca": "Pagó con beca", "pago_credito": "Pagó con crédito"}
tabla_g, kw = [], []
for col, nombre in GRUPOS.items():
    g = d.groupby(col)[OBJ]
    t = g.agg(["count", "mean", "median", "std"]).reset_index().rename(columns={col: "categoria"})
    t.insert(0, "variable", nombre)
    tabla_g.append(t)
    grupos = [x.values for _, x in g if len(x) > 30]
    h, p = stats.kruskal(*grupos)
    n = sum(len(x) for x in grupos)
    kw.append({"variable": nombre, "n": n, "H_kruskal": round(h, 1), "p_valor": p,
               "epsilon2": round(h * (n + 1) / (n ** 2 - 1), 4)})
pd.concat(tabla_g).round(2).to_csv(C.TABLAS / "eda_puntaje_por_grupo.csv", index=False, encoding="utf-8-sig")
kw = pd.DataFrame(kw).sort_values("epsilon2", ascending=False)
kw["magnitud"] = pd.cut(kw["epsilon2"], [-1, .01, .06, .14, 1], labels=["insignificante", "pequeña", "mediana", "grande"])
kw.to_csv(C.TABLAS / "eda_kruskal_efectos.csv", index=False, encoding="utf-8-sig")
print(kw.to_string())

# ================================== 5. Graficos objetivo vs categoricas
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
orden = d.groupby("region_inst")[OBJ].median().sort_values().index
sns.boxplot(data=d, y="region_inst", x=OBJ, order=orden, color=AZUL, showfliers=False, ax=ax[0])
ax[0].set(ylabel="", xlabel="Puntaje global", title="Por región de la institución")
orden = d.groupby("inst_origen")[OBJ].median().sort_values().index
sns.boxplot(data=d, y="inst_origen", x=OBJ, order=orden, color=AZUL, showfliers=False, ax=ax[1])
ax[1].set(ylabel="", xlabel="Puntaje global", title="Por naturaleza jurídica")
guardar(fig, "fig_03_region_y_origen.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(data=d, x="estrato", y=OBJ, order=ORD_ESTRATO, color=AZUL, showfliers=False, ax=ax[0])
ax[0].set(xlabel="", ylabel="Puntaje global", title="Por estrato socioeconómico")
plt.setp(ax[0].get_xticklabels(), rotation=30, ha="right")
ed = d[d["educ_madre"].isin(list(C.ORD_EDUCACION))]
sns.boxplot(data=ed, x="educ_madre", y=OBJ, order=list(C.ORD_EDUCACION), color=AZUL, showfliers=False, ax=ax[1])
ax[1].set(xlabel="", ylabel="", title="Por nivel educativo de la madre")
plt.setp(ax[1].get_xticklabels(), rotation=40, ha="right", fontsize=7)
guardar(fig, "fig_04_estrato_y_educacion_madre.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.pointplot(data=d, x="anio", y=OBJ, hue="modalidad", errorbar=("ci", 95), palette=[AZUL, NARANJA, GRIS], ax=ax[0])
ax[0].set(xlabel="Año", ylabel="Puntaje global promedio", title="Modalidad del programa por año")
sns.boxplot(data=d, x="inst_caracter", y=OBJ, color=AZUL, showfliers=False, ax=ax[1])
ax[1].set(xlabel="", ylabel="", title="Carácter académico de la institución")
plt.setp(ax[1].get_xticklabels(), rotation=15, ha="right")
guardar(fig, "fig_05_modalidad_y_caracter.png")

# ================================== 6. Genero por modulo (H2b)
fil = []
for a in C.PUNT:
    f, h = d.loc[d.genero == "Mujer", a].dropna(), d.loc[d.genero == "Hombre", a].dropna()
    dif = h.mean() - f.mean()
    se = np.sqrt(h.var() / len(h) + f.var() / len(f))
    fil.append({"modulo": C.MODULOS[a], "media_mujeres": round(f.mean(), 2), "media_hombres": round(h.mean(), 2),
                "dif_hombres_menos_mujeres": round(dif, 2), "ic95_inf": round(dif - 1.96 * se, 2),
                "ic95_sup": round(dif + 1.96 * se, 2), "d_cohen": round(cohen_d(h, f), 3),
                "p_mannwhitney": stats.mannwhitneyu(h, f).pvalue})
f, h = d.loc[d.genero == "Mujer", OBJ], d.loc[d.genero == "Hombre", OBJ]
dif = h.mean() - f.mean()
se = np.sqrt(h.var() / len(h) + f.var() / len(f))
fil.append({"modulo": "Puntaje global", "media_mujeres": round(f.mean(), 2), "media_hombres": round(h.mean(), 2),
            "dif_hombres_menos_mujeres": round(dif, 2), "ic95_inf": round(dif - 1.96 * se, 2),
            "ic95_sup": round(dif + 1.96 * se, 2), "d_cohen": round(cohen_d(h, f), 3),
            "p_mannwhitney": stats.mannwhitneyu(h, f).pvalue})
gen = pd.DataFrame(fil)
gen.to_csv(C.TABLAS / "eda_genero_modulos.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 4))
g2 = gen.iloc[:-1]
ax.barh(g2["modulo"], g2["dif_hombres_menos_mujeres"], xerr=[g2["dif_hombres_menos_mujeres"] - g2["ic95_inf"],
        g2["ic95_sup"] - g2["dif_hombres_menos_mujeres"]], color=AZUL, ecolor="black", capsize=3)
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="Diferencia de medias, hombres − mujeres (puntos, IC 95 %)", title="Brecha de género por módulo")
guardar(fig, "fig_06_brecha_genero_modulos.png")

# ======================================== 7. Heatmaps multivariados
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
h1 = d[d.estrato.isin(ORD_ESTRATO[:6])].pivot_table(index="region_inst", columns="estrato", values=OBJ, aggfunc="mean")
sns.heatmap(h1[ORD_ESTRATO[:6]], annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Puntaje global"}, ax=ax[0])
ax[0].set(xlabel="", ylabel="", title="Región × estrato (media)")
h2 = d.pivot_table(index="sector", columns="modalidad", values=OBJ, aggfunc="mean")
sns.heatmap(h2, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Puntaje global"}, ax=ax[1])
ax[1].set(xlabel="", ylabel="", title="Sector × modalidad (media)")
guardar(fig, "fig_07_heatmaps_region_estrato_sector_modalidad.png")
h1.round(2).to_csv(C.TABLAS / "eda_region_x_estrato.csv", encoding="utf-8-sig")

# ================================================= 8. Correlaciones
cols = ["puntaje_global", *C.PUNT, "estrato_num", "educ_madre_ord", "educ_padre_ord", "matricula_ord",
        "horas_trabajo_ord", "tiene_internet", "tiene_computador", "tiene_automovil", "edad"]
etq = {**C.MODULOS, "puntaje_global": "Puntaje global", "estrato_num": "Estrato", "educ_madre_ord": "Educ. madre",
       "educ_padre_ord": "Educ. padre", "matricula_ord": "Matrícula", "horas_trabajo_ord": "Horas trabajo",
       "tiene_internet": "Internet", "tiene_computador": "Computador", "tiene_automovil": "Automóvil", "edad": "Edad"}
corr = d[cols].corr(method="spearman").rename(index=etq, columns=etq)
corr.round(3).to_csv(C.TABLAS / "eda_correlaciones_spearman.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 7},
            cbar_kws={"label": "ρ de Spearman"}, ax=ax)
ax.set_title("Correlaciones de Spearman")
guardar(fig, "fig_08_correlaciones.png")

# ======================================== 9. Instituciones y departamentos
inst = (d.groupby(["inst_nombre", "sector", "inst_caracter", "depto_inst"])[OBJ]
        .agg(["count", "mean"]).reset_index().query("count >= 300").sort_values("mean", ascending=False))
inst["mean"] = inst["mean"].round(1)
tb = pd.concat([inst.head(10).assign(grupo="10 más altos"), inst.tail(10).assign(grupo="10 más bajos")])
tb.to_csv(C.TABLAS / "eda_instituciones_top_bottom.csv", index=False, encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(10, 6.5))
def nombre_corto(n):
    """Quita la ciudad tras el ultimo guion y pasa a formato titulo con preposiciones en minuscula."""
    base = n.rsplit("-", 1)[0].strip().replace('"', "").title()
    for w in (" De ", " Del ", " La ", " Los ", " Y ", " Las "):
        base = base.replace(w, w.lower())
    return base if len(base) <= 52 else base[:51] + "…"


tb2 = tb.assign(nombre=tb["inst_nombre"].map(nombre_corto))
colores = [AZUL if g == "10 más altos" else NARANJA for g in tb2["grupo"]]
ax.barh(tb2["nombre"][::-1], tb2["mean"][::-1], color=colores[::-1])
ax.axvline(d[OBJ].mean(), color="black", ls="--", lw=0.8, label=f"Media nacional {d[OBJ].mean():.0f}")
ax.set(xlabel="Puntaje global promedio", title="Instituciones con mayor y menor puntaje (n ≥ 300)")
ax.legend(loc="lower right")
guardar(fig, "fig_09_instituciones_top_bottom.png")
dep = (d.groupby(["depto_inst", "region_inst"])[OBJ].agg(["count", "mean"]).reset_index()
       .query("count >= 500").sort_values("mean"))
dep["mean"] = dep["mean"].round(1)
dep.to_csv(C.TABLAS / "eda_departamentos.csv", index=False, encoding="utf-8-sig")

# ================================= 10. Modelo lineal multivariado (efectos ajustados)
m = d.dropna(subset=["estrato", "educ_madre_ord", "educ_padre_ord", "tiene_internet", "tiene_computador",
                     "horas_trabajo_ord", "matricula_ord", "edad", "genero"]).copy()
m["anio"] = m["anio"].astype(str)
# Categoria de referencia = primer nivel (patsy usa codificacion de tratamiento)
REFERENCIAS = {"estrato": "Estrato 1", "genero": "Mujer", "sector": "Oficial", "inst_caracter": "UNIVERSIDAD",
               "region_inst": "Andina", "modalidad": "PRESENCIAL", "anio": "2020"}
for col, ref in REFERENCIAS.items():
    otros = [v for v in sorted(m[col].dropna().unique()) if v != ref]
    m[col] = pd.Categorical(m[col], categories=[ref, *otros])
bloques = {
    "Estrato": "estrato",
    "Educación de los padres": "educ_madre_ord + educ_padre_ord",
    "Conectividad": "tiene_internet + tiene_computador",
    "Género": "genero",
    "Sector": "sector",
    "Carácter académico": "inst_caracter",
    "Región": "region_inst",
    "Modalidad": "modalidad",
    "Trabajo y matrícula": "horas_trabajo_ord + matricula_ord",
    "Edad": "edad",
    "Año": "anio",
}
formula = f"{OBJ} ~ " + " + ".join(bloques.values())
full = smf.ols(formula, data=m).fit(cov_type="HC1")
print(f"R2 modelo completo: {full.rsquared:.3f}  (n={int(full.nobs)})")
ci = full.conf_int()
coef = pd.DataFrame({"termino": full.params.index, "coef": full.params.values, "ic95_inf": ci[0].values,
                     "ic95_sup": ci[1].values, "p_valor": full.pvalues.values}).round(4)
coef.to_csv(C.TABLAS / "eda_ols_coeficientes.csv", index=False, encoding="utf-8-sig")
fil = []
for nombre in bloques:
    resto = " + ".join(v for k, v in bloques.items() if k != nombre)
    r2 = smf.ols(f"{OBJ} ~ {resto}", data=m).fit().rsquared
    fil.append({"bloque": nombre, "delta_R2_al_quitarlo": round(full.rsquared - r2, 4)})
aporte = pd.DataFrame(fil).sort_values("delta_R2_al_quitarlo", ascending=False)
aporte.loc[len(aporte)] = ["R² del modelo completo", round(full.rsquared, 4)]
aporte.to_csv(C.TABLAS / "eda_ols_aporte_bloques.csv", index=False, encoding="utf-8-sig")
print(aporte.to_string())

# Cuanto de la varianza se explica solo por la institucion (eta2)
gm = d[OBJ].mean()
ss_b = d.groupby("inst_cod")[OBJ].apply(lambda s: len(s) * (s.mean() - gm) ** 2).sum()
ss_t = ((d[OBJ] - gm) ** 2).sum()
pd.DataFrame([{"medida": "eta2 por institución (varianza explicada por la institución)", "valor": round(ss_b / ss_t, 4)}]
             ).to_csv(C.TABLAS / "eda_eta2_institucion.csv", index=False, encoding="utf-8-sig")
print("eta2 institucion:", round(ss_b / ss_t, 4))

fig, ax = plt.subplots(figsize=(8, 5.5))
sel = coef[coef["termino"].str.contains(r"^(?:estrato|region_inst|genero|sector|inst_caracter|modalidad)\[", regex=True)].copy()
sel["etq"] = sel["termino"].str.replace("[T.", ": ", regex=False).str.replace("]", "", regex=False)
sel = sel.sort_values("coef")
ax.errorbar(sel["coef"], range(len(sel)), xerr=[sel["coef"] - sel["ic95_inf"], sel["ic95_sup"] - sel["coef"]],
            fmt="o", color=AZUL, ecolor=GRIS, capsize=2)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(sel["etq"], fontsize=7)
ax.axvline(0, color="black", lw=0.8)
ax.set(xlabel="Diferencia en puntos del puntaje global, ajustada por las demás variables (IC 95 %)",
       title="Efectos ajustados (regresión lineal múltiple)")
guardar(fig, "fig_10_efectos_ajustados.png")

# ===================================================== 11. Sector / H1 resumen
h1 = []
for col in ["sector", "inst_origen", "inst_caracter"]:
    h1.append(d.groupby(col)[OBJ].agg(["count", "mean", "std"]).round(2).reset_index().rename(columns={col: "grupo"})
              .assign(variable=col))
pd.concat(h1).to_csv(C.TABLAS / "eda_h1_institucional.csv", index=False, encoding="utf-8-sig")
print("EDA terminado.")
