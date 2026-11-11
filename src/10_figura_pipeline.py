"""10 - Diagrama del Pipeline sin fuga de informacion y de la particion de datos (figura del documento)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config as C

AZUL, GRIS, NARANJA, VERDE, ROJO = "#2E5C8A", "#E9EEF4", "#D9822B", "#3C7A5A", "#B5473A"

fig, ax = plt.subplots(figsize=(11.5, 6.0))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.4)
ax.axis("off")


def caja(x, y, w, h, titulo, detalle="", fc=GRIS, ec=AZUL, tc=AZUL, lw=1.8, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 0.27, titulo, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
    if detalle:
        ax.text(x + w / 2, y + (h - 0.27) / 2, detalle, ha="center", va="center", fontsize=8.4, color="#222")


def flecha(a, b, color="#444", estilo="-", lw=1.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15, lw=lw, color=color, linestyle=estilo))


# 1. Particion temporal
caja(0.4, 5.15, 7.0, 0.95, "Entrenamiento 2020–2021 · 467.356 registros", "", fc="#DCE8F5", ec=AZUL)
caja(7.7, 5.15, 3.9, 0.95, "Prueba 2022 · 193.356 registros", "", fc="#F6E4D0", ec=NARANJA, tc=NARANJA)
ax.text(3.9, 5.33, "ajusta imputación, codificaciones, selección y modelo", ha="center", fontsize=8.6, color="#222")
ax.text(9.65, 5.33, "no se usa para ajustar nada", ha="center", fontsize=8.6, color="#222")

# 2. Pipeline
ax.text(6.1, 4.5, "Pipeline de scikit-learn: se ajusta solo con el entrenamiento", ha="center",
        fontsize=9.6, fontweight="bold", color="#333")
pasos = [("Imputación C1", "Estrato y educación\nde los padres\n(HistGradientBoosting)"),
         ("Codificación", "One-hot · target encoding\n(5 pliegues internos) ·\nmediana + estandarización"),
         ("Selección de variables", "Lasso suave, Lasso fuerte\no ninguna (compite\ncomo hiperparámetro)"),
         ("Estimador", "Ridge · Random Forest · HGB\nLightGBM · XGBoost (reg.)\n+ 5 clasificadores")]
xs = [0.3, 3.2, 6.1, 9.0]
for (t, d), x in zip(pasos, xs):
    caja(x, 2.55, 2.6, 1.65, t, d)
for x in xs[:-1]:
    flecha((x + 2.62, 3.37), (x + 2.88, 3.37))

# flechas desde entrenamiento (ajusta y transforma) y desde prueba (solo transforma)
flecha((1.6, 5.1), (1.6, 4.25), color=AZUL)
ax.text(1.7, 4.72, "fit + transform", fontsize=8, color=AZUL, style="italic")
flecha((10.3, 5.1), (10.3, 4.25), color=NARANJA, estilo="--")
ax.text(10.4, 4.72, "solo transform", fontsize=8, color=NARANJA, style="italic")

# 3. Evaluacion
caja(0.4, 0.35, 3.6, 1.55, "Validación cruzada (5 pliegues)", "“Conocidas”: estratificada por institución\n“Nuevas”: agrupada por institución\nElige hiperparámetros y mejor modelo", fs=10)
caja(4.4, 0.35, 3.6, 1.55, "Inferencia", "t pareada, Nadeau–Bengio, Wilcoxon,\nHolm · errores agrupados por\ninstitución · bootstrap de instituciones", fs=10)
caja(8.4, 0.35, 3.2, 1.55, "Prueba 2022 (una vez)", "RMSE, AUC, recall · SHAP\nablación · equidad", fc="#F6E4D0", ec=NARANJA, tc=NARANJA, fs=10)
flecha((1.8, 2.5), (1.8, 1.95))
flecha((4.05, 1.1), (4.35, 1.1))
flecha((8.05, 1.1), (8.35, 1.1))
flecha((10.3, 2.5), (10.0, 1.95), color=NARANJA, estilo="--")

fig.tight_layout()
fig.savefig(C.FIGURAS / "mod_pipeline_sin_fuga.png", dpi=170, bbox_inches="tight")
print("ok")
