"""01 - Descarga de datos (CRISP-DM: Comprension de los datos).

Fuente: ICFES - "Resultados unicos Saber Pro", portal Datos Abiertos Colombia.
https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-Pro/u37r-hjmu
Se descargan solo los periodos 2020-2022 (codigos 20201..20229) y las columnas
necesarias para el proyecto, paginando la API SODA (CSV).
"""
import csv
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.datos.gov.co/resource/u37r-hjmu.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "saberpro_2020_2022.csv"
PAGE = 50_000

COLUMNAS = [
    "periodo", "estu_consecutivo", "estu_genero", "estu_fechanacimiento",
    "estu_depto_reside", "estu_mcpio_reside", "estu_depto_presentacion",
    "inst_cod_institucion", "inst_nombre_institucion", "inst_caracter_academico",
    "inst_origen", "estu_nucleo_pregrado", "estu_prgm_academico",
    "estu_nivel_prgm_academico", "estu_metodo_prgm", "estu_inst_departamento",
    "estu_valormatriculauniversidad", "estu_horassemanatrabaja",
    "estu_pagomatriculabeca", "estu_pagomatriculacredito",
    "estu_pagomatriculapadres", "estu_pagomatriculapropio",
    "fami_estratovivienda", "fami_educacionpadre", "fami_educacionmadre",
    "fami_tieneautomovil", "fami_tienelavadora", "fami_tienecomputador",
    "fami_tieneinternet",
    "mod_razona_cuantitat_punt", "mod_lectura_critica_punt",
    "mod_competen_ciudada_punt", "mod_ingles_punt", "mod_comuni_escrita_punt",
    "mod_ingles_desem", "mod_comuni_escrita_desem",
]


def pedir(offset: int) -> str:
    params = {
        "$select": ",".join(COLUMNAS),
        "$where": "periodo >= '20201' AND periodo <= '20229'",
        "$order": "estu_consecutivo",
        "$limit": PAGE,
        "$offset": offset,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    for intento in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode("utf-8")
        except Exception as e:  # reintento simple ante fallas de red
            print(f"  reintento {intento + 1}: {e}")
            time.sleep(5 * (intento + 1))
    raise RuntimeError(f"No fue posible descargar offset={offset}")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total, offset = 0, 0
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS)
        while True:
            txt = pedir(offset)
            filas = list(csv.reader(io.StringIO(txt)))
            header, datos = filas[0], filas[1:]
            if not datos:
                break
            # Socrata omite columnas vacias en el CSV: se alinea por nombre
            idx = {c: i for i, c in enumerate(header)}
            for fila in datos:
                w.writerow([fila[idx[c]] if c in idx and idx[c] < len(fila) else "" for c in COLUMNAS])
            total += len(datos)
            offset += PAGE
            print(f"{total:,} filas descargadas")
            if len(datos) < PAGE:
                break
    print(f"Listo: {total:,} filas -> {OUT}")


if __name__ == "__main__":
    main()
