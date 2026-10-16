"""06 - Genera docs/ficha_tecnica.md a partir de las tablas de reports/tables."""
import json

import pandas as pd

import config as C

f = json.loads((C.TABLAS / "ficha_tecnica.json").read_text(encoding="utf-8"))
dic = pd.read_csv(C.TABLAS / "diccionario_variables.csv")
pasos = pd.read_csv(C.TABLAS / "limpieza_pasos.csv")

L = [
    "# Ficha técnica del conjunto de datos", "",
    "| Campo | Detalle |", "|---|---|",
    f"| Nombre | {f['nombre']} |",
    f"| Entidad | {f['entidad']} |",
    f"| Portal | {f['portal']} (identificador `{f['identificador']}`) |",
    f"| URL | {f['url']} |",
    f"| Fecha de descarga | {f['fecha_descarga']} |",
    f"| Método | {f['metodo_descarga']} |",
    f"| Periodos | {', '.join(f['periodos_incluidos'])} (años 2020–2022) |",
    f"| Registros descargados | {f['registros_descargados']:,} |",
    f"| Registros tras limpieza | {f['registros_tras_limpieza']:,} |",
    f"| Variables | {f['variables_descargadas']} descargadas → {f['variables_finales']} finales |",
    f"| Unidad de análisis | {f['unidad_de_analisis']} |",
    f"| Variable objetivo | {f['variable_objetivo']} |", "",
    "## Consideraciones éticas", "", *[f"* {x}" for x in f["consideraciones_eticas"]], "",
    "## Limitaciones", "", *[f"* {x}" for x in f["limitaciones"]], "",
    "## Trazabilidad de la limpieza", "",
    "| Paso | Descripción | Registros antes | Registros después | Afectados |", "|---|---|---|---|---|",
]
for r in pasos.itertuples():
    L.append(f"| {r.paso} | {r.descripcion} | {r.registros_antes:,} | {r.registros_despues:,} | {r.afectados:,} |")
L += ["", "## Diccionario de variables (dataset limpio)", "",
      "| Variable | Tipo | Descripción | Rango / categorías | % nulos crudo | % nulos limpio |", "|---|---|---|---|---|---|"]
for r in dic.itertuples():
    crudo = "—" if pd.isna(r.pct_nulos_crudo) else f"{r.pct_nulos_crudo:.2f}"
    L.append(f"| `{r.variable}` | {r.tipo} | {r.descripcion} | {r.rango_o_categorias} | {crudo} | {r.pct_nulos_limpio:.2f} |")
(C.RAIZ / "docs" / "ficha_tecnica.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("docs/ficha_tecnica.md generado")
