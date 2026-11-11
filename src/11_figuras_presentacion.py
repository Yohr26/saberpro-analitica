"""11 - Figuras de la presentacion: arquitectura tecnica y rendimiento (curva ROC y RMSE frente a las lineas base).

Entradas : data/processed/pred_test_2022.parquet (script 08), reports/tables/mod_test_2022_rmse_ic.csv (script 09)
Salidas  : reports/figures/pres_arquitectura_tecnica.png, reports/figures/pres_roc_rmse.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

import config as C

AZUL, GRIS, NARANJA, VERDE, TXT = "#2E5C8A", "#E9EEF4", "#D9822B", "#3C7A5A", "#222222"
coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",") if abs(v) < 10 else f"{v:.0f}")

# ----------------------------------------------------------------------------- 1. Arquitectura tecnica
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.6)
ax.axis("off")


def caja(x, y, w, h, titulo, detalle, color=AZUL, fc=GRIS, fs_t=11.5, fs_d=9.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=color, lw=1.9))
    ax.text(x + w / 2, y + h - 0.27, titulo, ha="center", va="center", fontsize=fs_t, fontweight="bold", color=color)
    ax.text(x + w / 2, y + (h - 0.4) / 2, detalle, ha="center", va="center", fontsize=fs_d, color=TXT, linespacing=1.35)


def flecha(a, b, color="#444"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=1.7, color=color))


ax.text(0.1, 5.38, "Recorrido de los datos", fontsize=11, fontweight="bold", color="#333")
W, H, Y = 2.15, 1.75, 3.35
xs = [0.1, 2.5, 4.9, 7.3, 9.7]
cajas = [
    ("1. Fuente", "ICFES · datos.gov.co\nconjunto u37r-hjmu\nAPI filtrada 2020–2022\n721.129 registros"),
    ("2. Extracción y ETL", "Python 3.12 + pandas\nscripts 01 a 03\nlimpieza, nulos, imputación\n660.712 con 5 módulos"),
    ("3. Datos analíticos", "Archivos CSV y Parquet\nlocales (no se versionan)\nse regeneran con el\nscript 01"),
    ("4. Análisis y modelado", "statsmodels · scikit-learn\nLightGBM · XGBoost · SHAP\nscripts 04 a 09\nPipeline sin fuga"),
    ("5. Salidas", "Tablas CSV y figuras\nmétricas por pliegue\nmodelos entrenados\ndocumento y diapositivas"),
]
for x, (t, d) in zip(xs, cajas):
    caja(x, Y, W, H, t, d)
for x in xs[:-1]:
    flecha((x + W + 0.02, Y + H / 2), (x + 2.4 - 0.02, Y + H / 2))
ax.text(2.45, Y - 0.2, "Entorno de ejecución: equipo local (Windows) o Google Colab", ha="center", fontsize=8.8, style="italic", color=VERDE)

caja(2.5, 1.65, 3.6, 1.15, "Power BI", "Exploración visual interactiva de los datos\n(carga y medidas: Figuras 6 y 7 del documento)", color=NARANJA, fc="#FBF1E6")
caja(6.5, 1.65, 3.9, 1.15, "GitHub + Google Colab", "Código, bitácora de decisiones y cuaderno\npara reproducir el análisis y el modelado", color=VERDE, fc="#EAF3EE")
flecha((6.0, Y - 0.02), (4.3, 2.82), color=NARANJA)
flecha((10.77, Y - 0.02), (9.2, 2.82), color=VERDE)

ax.add_patch(FancyBboxPatch((0.1, 0.1), 11.8, 1.25, boxstyle="round,pad=0.03,rounding_size=0.1", fc="white", ec="#999", lw=1.2, ls="--"))
ax.text(0.3, 1.13, "¿Por qué estas tecnologías para más de 700 mil registros?", fontsize=10, fontweight="bold", color="#333", va="center")
razones = [("Python + pandas", "procesa 700 mil filas y 36 variables\nen memoria con scripts repetibles;\nuna hoja de cálculo no es reproducible"),
           ("scikit-learn, LightGBM, XGBoost", "modelos y validación cruzada sobre\n467 mil registros; el pipeline evita\nla fuga de información"),
           ("Power BI", "exploración interactiva sin código\npara una audiencia no técnica"),
           ("GitHub + Colab", "trazabilidad y reproducción\nde extremo a extremo por terceros")]
for x, (t, d) in zip([0.3, 3.2, 6.6, 9.1], razones):
    ax.text(x, 0.82, t, fontsize=9.4, fontweight="bold", color=AZUL, va="center")
    ax.text(x, 0.4, d, fontsize=8.6, color=TXT, va="center", linespacing=1.3)
fig.tight_layout()
fig.savefig(C.FIGURAS / "pres_arquitectura_tecnica.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------- 2. Curva ROC y RMSE frente a las lineas base
pred = pd.read_parquet(C.RAIZ / "data" / "processed" / "pred_test_2022.parquet")
bajo = pred["bajo"].values
curvas = [  # (etiqueta, puntaje de riesgo, color, estilo)
    ("XGBoost (clasificador)", pred["XGBoost (clasif.)"].values, AZUL, "-"),
    ("XGBoost (regresión)", -pred["XGBoost"].values, "#6FA3D0", "-"),
    ("Regresión logística", pred["Regresión logística"].values, VERDE, "-"),
    ("Línea base B3 (media por institución)", -pred["B3. Media por institución"].values, NARANJA, "--"),
    ("Línea base B1 (media por estrato)", -pred["B1. Media por estrato"].values, "#999999", "--"),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1.05]})
for et, sc, col, ls in curvas:
    fpr, tpr, _ = roc_curve(bajo, sc)
    a1.plot(fpr, tpr, color=col, ls=ls, lw=2.2 if ls == "-" else 1.8, label=f"{et} (AUC {roc_auc_score(bajo, sc):.3f})".replace(".", ","))
a1.plot([0, 1], [0, 1], color="#bbb", lw=1, ls=":")
a1.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos (recall)", title="Curva ROC: detección de bajo desempeño (2022)", xlim=(0, 1), ylim=(0, 1.01))
a1.xaxis.set_major_formatter(coma)
a1.yaxis.set_major_formatter(coma)
a1.legend(loc="lower right", fontsize=8, frameon=False)
a1.grid(alpha=0.25)

ic = pd.read_csv(C.TABLAS / "mod_test_2022_rmse_ic.csv").set_index("modelo")
orden = ["B0. Media global", "B1. Media por estrato", "B3. Media por institución", "Ridge", "RandomForest", "HistGradientBoosting", "LightGBM", "XGBoost"]
etq = {"B0. Media global": "B0 · Media global", "B1. Media por estrato": "B1 · Media por estrato", "B3. Media por institución": "B3 · Media por institución",
       "RandomForest": "Random Forest"}
y = np.arange(len(orden))[::-1]
vals = ic.loc[orden, "RMSE"].values
err = np.vstack([vals - ic.loc[orden, "IC95_inf"].values, ic.loc[orden, "IC95_sup"].values - vals])
cols = [("#B8B8B8" if m.startswith("B") else AZUL) for m in orden]
a2.barh(y, vals, color=cols, height=0.62, xerr=err, error_kw=dict(ecolor="#333", lw=1.2, capsize=3))
for yi, v, m in zip(y, vals, orden):
    a2.text(0.35, yi, f"{v:.2f}".replace(".", ","), va="center", ha="left", fontsize=9, color="white" if not m.startswith("B") else "#222", fontweight="bold")
a2.axvline(19.4, color=NARANJA, ls="--", lw=1.8)
a2.text(19.65, -0.62, "criterio de éxito: RMSE ≤ 19,4", color=NARANJA, fontsize=8.5, va="center")
a2.set_ylim(-0.95, len(orden) - 0.4)
a2.set_yticks(y)
a2.set_yticklabels([etq.get(m, m) for m in orden], fontsize=9)
a2.set(xlabel="RMSE en la prueba 2022 (puntos; barras = IC 95 %)", title="Error de los modelos frente a las líneas base", xlim=(0, 27.5))
a2.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(C.FIGURAS / "pres_roc_rmse.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("ok")
