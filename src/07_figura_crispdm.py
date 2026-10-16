"""07 - Diagrama de la metodologia CRISP-DM adaptada al proyecto (Figura 2 del documento)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config as C

AZUL, GRIS, NARANJA, VERDE = "#2E5C8A", "#E9EEF4", "#D9822B", "#3C7A5A"
fases = [
    ("1. Comprensión\ndel negocio", "Problema, hipótesis y objetivos;\nrevisión documental", VERDE, "Entrega 1"),
    ("2. Comprensión\nde los datos", "Ficha técnica, diccionario,\nEDA multivariado", VERDE, "Entrega 1"),
    ("3. Preparación\nde los datos", "ETL, nulos y atípicos,\nimputación comparada", VERDE, "Entrega 1"),
    ("4. Modelado", "Regresión y clasificación\nde bajo desempeño", NARANJA, "Entrega final"),
    ("5. Evaluación", "Protocolo: métricas, partición,\nlíneas base, criterios de éxito", NARANJA, "Protocolo en entrega 1"),
    ("6. Despliegue", "Tres recomendaciones de\ninnovación pedagógica", VERDE, "Entrega 1 (se refinan en la final)"),
]
pos = [(0.8, 3.2), (4.6, 3.2), (8.4, 3.2), (8.4, 0.6), (4.6, 0.6), (0.8, 0.6)]
fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.set_xlim(0, 12.2)
ax.set_ylim(-0.4, 5.4)
ax.axis("off")
W, H = 3.0, 1.75
for (titulo, desc, col, etiqueta), (x, y) in zip(fases, pos):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.04,rounding_size=0.12", fc=GRIS, ec=col, lw=2.2))
    ax.text(x + W / 2, y + H - 0.38, titulo, ha="center", va="center", fontsize=11.5, fontweight="bold", color=col)
    ax.text(x + W / 2, y + 0.55, desc, ha="center", va="center", fontsize=8.8, color="#222")
    ax.text(x + W / 2, y + 0.1, etiqueta, ha="center", va="center", fontsize=8, style="italic", color=col)
flechas = [((3.85, 4.05), (4.55, 4.05)), ((7.65, 4.05), (8.35, 4.05)), ((9.9, 3.15), (9.9, 2.4)),
           ((8.35, 1.5), (7.65, 1.5)), ((4.55, 1.5), (3.85, 1.5))]
for a, b in flechas:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444"))
ax.add_patch(FancyArrowPatch((2.3, 2.4), (2.3, 3.15), arrowstyle="-|>", mutation_scale=16, lw=1.8, color="#444",
                             connectionstyle="arc3,rad=0"))
ax.text(6.1, 2.8, "Iteraciones: los hallazgos del EDA\nretroalimentan la preparación y la evaluación",
        ha="center", va="center", fontsize=8.5, style="italic", color="#555")
ax.text(0.2, 5.15, "Verde: cubierto en esta entrega   ·   Naranja: siguiente entrega", fontsize=9, color="#333")
fig.tight_layout()
fig.savefig(C.FIGURAS / "fig_00_crispdm.png", dpi=170, bbox_inches="tight")
print("ok")
