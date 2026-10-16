"""02 - ETL y limpieza (CRISP-DM: Preparacion de los datos).

Entrada : data/raw/saberpro_2020_2022.csv
Salidas : data/processed/saberpro_limpio.parquet
          reports/tables/{ficha_tecnica.json, limpieza_pasos.csv, nulos_antes_despues.csv,
                          atipicos_antes_despues.csv, diccionario_variables.csv}

Cada decision de limpieza queda registrada en docs/bitacora_decisiones.md (D-xx).
"""
import json
from datetime import date

import numpy as np
import pandas as pd

import config as C

C.TABLAS.mkdir(parents=True, exist_ok=True)
C.LIMPIO.parent.mkdir(parents=True, exist_ok=True)

pasos = []  # registro de la limpieza: registros antes/despues de cada paso


def registrar(paso, descripcion, antes, despues, afectados=None):
    pasos.append({
        "paso": paso, "descripcion": descripcion, "registros_antes": antes,
        "registros_despues": despues,
        "afectados": antes - despues if afectados is None else afectados,
    })


# ---------------------------------------------------------------- 1. Lectura
raw = pd.read_csv(C.RAW, dtype=str, keep_default_na=False, na_values=[""])
n_raw = len(raw)
for c in C.RENOMBRAR_MODULOS:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
nulos_raw = raw.isna().sum()
df = raw.copy()

# ------------------------------------------------- 2. Duplicados (D-01)
# Los 5.026 consecutivos repetidos tienen puntajes y atributos identicos; solo cambia
# la sede de presentacion (estu_depto_presentacion), que no se usa en el analisis.
n0 = len(df)
df = df.sort_values(["estu_consecutivo", "estu_depto_presentacion"]).drop_duplicates("estu_consecutivo", keep="first")
registrar("D-01", "Eliminar registros duplicados por estu_consecutivo", n0, len(df))

# -------------------------- 3. Registros sin contexto familiar (D-02)
ctx = ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre", "fami_tieneautomovil",
       "fami_tienelavadora", "fami_tienecomputador", "fami_tieneinternet"]
sin_ctx = df[ctx].isna().all(axis=1)
detalle_sin_ctx = df.loc[sin_ctx, "periodo"].value_counts().to_dict()
n0 = len(df)
df = df.loc[~sin_ctx].copy()
registrar("D-02", f"Eliminar registros sin NINGUN dato de contexto familiar {detalle_sin_ctx}", n0, len(df))

# ----------------------------- 4. Estandarizacion de texto (D-03)
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].str.strip()
df["estu_metodo_prgm"] = df["estu_metodo_prgm"].replace({"DISTANCIA VITUAL": "DISTANCIA VIRTUAL"})
dept_norm = df["estu_depto_reside"].map(C.normalizar)
validos = set(C.REGION_POR_DEPTO) | {"BOGOTA", "EXTRANJERO"}
fuera = (~dept_norm.isin(validos)) & dept_norm.notna()
n_exterior = int(fuera.sum())
df["depto_reside"] = np.where(fuera, "EXTRANJERO", df["estu_depto_reside"])
registrar("D-03", "Agrupar ciudades del exterior de estu_depto_reside en 'EXTRANJERO'; corregir 'DISTANCIA VITUAL'",
          len(df), len(df), afectados=n_exterior)

# ---------------------------- 5. Puntajes: ceros y rango (D-04)
antes_pt = df[list(C.RENOMBRAR_MODULOS)].copy()  # para tabla de atipicos
fuera_rango = 0
ceros = {}
for orig, nuevo in C.RENOMBRAR_MODULOS.items():
    s = df[orig]
    fuera_rango += int(((s < 0) | (s > 300)).sum())
    ceros[nuevo] = int((s == 0).sum())
    s = s.mask((s < 0) | (s > 300))
    df[nuevo] = s.mask(s == 0)  # 0 = modulo no calificado / invalido -> NaN
df = df.drop(columns=list(C.RENOMBRAR_MODULOS))
registrar("D-04", f"Puntaje 0 -> NaN (no calificado). Ceros por modulo: {ceros}. Fuera de [0,300]: {fuera_rango}",
          len(df), len(df), afectados=sum(ceros.values()) + fuera_rango)

# --------------------------------- 6. Edad (D-05)
nac = pd.to_datetime(df["estu_fechanacimiento"], format="%d/%m/%Y", errors="coerce")
edad = df["periodo"].str[:4].astype(int) - nac.dt.year
inval = (edad < 16) | (edad > 70) | edad.isna()
df["edad"] = edad.mask(inval)
registrar("D-05", "Edad fuera de [16, 70] o fecha de nacimiento invalida -> NaN (se descarta la fecha)",
          len(df), len(df), afectados=int(inval.sum()))

# ----------------------------- 7. Variables derivadas (D-06 a D-09)
df["anio"] = df["periodo"].str[:4].astype(int)
df["region_inst"] = df["estu_inst_departamento"].map(C.normalizar).map(C.REGION_POR_DEPTO)
df["sector"] = np.where(df["inst_origen"].str.startswith("OFICIAL"), "Oficial",
                        np.where(df["inst_origen"].str.startswith("NO OFICIAL"), "No oficial", "Régimen especial"))
df["estrato_num"] = df["fami_estratovivienda"].map(C.ORD_ESTRATO)
df["educ_madre_ord"] = df["fami_educacionmadre"].map(C.ORD_EDUCACION)
df["educ_padre_ord"] = df["fami_educacionpadre"].map(C.ORD_EDUCACION)
df["matricula_ord"] = df["estu_valormatriculauniversidad"].map(C.ORD_MATRICULA)
df["horas_trabajo_ord"] = df["estu_horassemanatrabaja"].map(C.ORD_HORAS_TRABAJO)
for c in ["fami_tieneautomovil", "fami_tienelavadora", "fami_tienecomputador", "fami_tieneinternet"]:
    df[c.replace("fami_tiene", "tiene_")] = df[c].map({"Si": 1, "No": 0})

df["n_modulos_validos"] = df[C.PUNT].notna().sum(axis=1)
# Puntaje global propio (el dataset no lo trae): promedio simple de los 5 modulos genericos,
# solo cuando los 5 son validos (D-07).
df["puntaje_global"] = df[C.PUNT].mean(axis=1).where(df["n_modulos_validos"] == 5)
df["nivel_global"] = pd.qcut(df["puntaje_global"], 4, labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"])

df = df.rename(columns={
    "estu_consecutivo": "id_anonimo", "estu_genero": "genero", "inst_cod_institucion": "inst_cod",
    "inst_nombre_institucion": "inst_nombre", "inst_caracter_academico": "inst_caracter",
    "estu_nucleo_pregrado": "nucleo", "estu_prgm_academico": "programa", "estu_metodo_prgm": "modalidad",
    "estu_inst_departamento": "depto_inst", "estu_valormatriculauniversidad": "matricula",
    "estu_horassemanatrabaja": "horas_trabajo", "estu_pagomatriculabeca": "pago_beca",
    "estu_pagomatriculacredito": "pago_credito", "estu_pagomatriculapadres": "pago_padres",
    "estu_pagomatriculapropio": "pago_propio", "fami_estratovivienda": "estrato",
    "fami_educacionpadre": "educ_padre", "fami_educacionmadre": "educ_madre",
    "mod_ingles_desem": "ingles_nivel", "mod_comuni_escrita_desem": "escrita_nivel",
})
final_cols = [
    "id_anonimo", "periodo", "anio", "genero", "edad", "depto_reside", "inst_cod", "inst_nombre",
    "inst_caracter", "inst_origen", "sector", "depto_inst", "region_inst", "nucleo", "programa",
    "modalidad", "matricula", "matricula_ord", "horas_trabajo", "horas_trabajo_ord", "pago_beca",
    "pago_credito", "pago_padres", "pago_propio", "estrato", "estrato_num", "educ_madre",
    "educ_madre_ord", "educ_padre", "educ_padre_ord", "tiene_automovil", "tiene_lavadora",
    "tiene_computador", "tiene_internet", *C.PUNT, "ingles_nivel", "escrita_nivel",
    "n_modulos_validos", "puntaje_global", "nivel_global",
]
df = df[final_cols].reset_index(drop=True)
registrar("FIN", "Dataset limpio (sin imputar)", n_raw, len(df))
df.to_parquet(C.LIMPIO, index=False)
pd.DataFrame(pasos).to_csv(C.TABLAS / "limpieza_pasos.csv", index=False, encoding="utf-8-sig")

# ---------------------------- Tabla de nulos antes / despues
cmp_cols = {  # columna original -> columna limpia
    "estu_genero": "genero", "estu_depto_reside": "depto_reside", "inst_origen": "inst_origen",
    "estu_valormatriculauniversidad": "matricula", "estu_horassemanatrabaja": "horas_trabajo",
    "estu_pagomatriculabeca": "pago_beca", "estu_pagomatriculacredito": "pago_credito",
    "estu_pagomatriculapadres": "pago_padres", "estu_pagomatriculapropio": "pago_propio",
    "fami_estratovivienda": "estrato", "fami_educacionpadre": "educ_padre",
    "fami_educacionmadre": "educ_madre", "fami_tieneautomovil": "tiene_automovil",
    "fami_tienelavadora": "tiene_lavadora", "fami_tienecomputador": "tiene_computador",
    "fami_tieneinternet": "tiene_internet", **C.RENOMBRAR_MODULOS,
    "mod_ingles_desem": "ingles_nivel", "mod_comuni_escrita_desem": "escrita_nivel",
    "estu_fechanacimiento": "edad",
}
filas = []
for o, n in cmp_cols.items():
    a = int(nulos_raw[o]) if o != "estu_fechanacimiento" else int(raw[o].isna().sum())
    d = int(df[n].isna().sum())
    filas.append({"variable": n, "nulos_antes": a, "pct_antes": round(100 * a / n_raw, 2),
                  "nulos_despues": d, "pct_despues": round(100 * d / len(df), 2)})
filas.append({"variable": "puntaje_global (derivada)", "nulos_antes": np.nan, "pct_antes": np.nan,
              "nulos_despues": int(df["puntaje_global"].isna().sum()),
              "pct_despues": round(100 * df["puntaje_global"].isna().mean(), 2)})
pd.DataFrame(filas).to_csv(C.TABLAS / "nulos_antes_despues.csv", index=False, encoding="utf-8-sig")

# ---------------------------- Tabla de atipicos antes / despues
def resumen(s):
    v = s.dropna()
    q1, q3 = v.quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {"n_validos": len(v), "media": round(v.mean(), 2), "desv": round(v.std(), 2), "min": v.min(),
            "max": v.max(), "ceros": int((v == 0).sum()), "lim_inf_iqr": round(lo, 1),
            "lim_sup_iqr": round(hi, 1), "atipicos_iqr": int(((v < lo) | (v > hi)).sum())}

filas = []
for o, n in C.RENOMBRAR_MODULOS.items():
    filas.append({"modulo": C.MODULOS[n], "etapa": "antes (crudo)", **resumen(antes_pt[o])})
    filas.append({"modulo": C.MODULOS[n], "etapa": "despues (ceros->NaN)", **resumen(df[n])})
filas.append({"modulo": "Puntaje global", "etapa": "despues (ceros->NaN)", **resumen(df["puntaje_global"])})
pd.DataFrame(filas).to_csv(C.TABLAS / "atipicos_antes_despues.csv", index=False, encoding="utf-8-sig")

# ------------------------------ Ficha tecnica
ficha = {
    "nombre": "Resultados únicos Saber Pro",
    "entidad": "Instituto Colombiano para la Evaluación de la Educación (ICFES)",
    "portal": "Datos Abiertos Colombia (datos.gov.co)",
    "identificador": "u37r-hjmu",
    "url": "https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-Pro/u37r-hjmu",
    "fecha_descarga": date.today().isoformat(),
    "metodo_descarga": "API SODA (CSV) con filtro periodo 20201-20229 y 36 columnas seleccionadas (src/01_descargar_datos.py)",
    "periodos_incluidos": sorted(raw["periodo"].unique().tolist()),
    "registros_descargados": n_raw,
    "registros_tras_limpieza": len(df),
    "unidad_de_analisis": "Una presentación de la prueba Saber Pro por estudiante (pregrado universitario)",
    "variables_descargadas": raw.shape[1],
    "variables_finales": df.shape[1],
    "variable_objetivo": "puntaje_global (derivado: promedio de los 5 módulos genéricos, escala 0-300)",
    "consideraciones_eticas": [
        "Datos abiertos y anonimizados publicados por el ICFES; no contienen nombre ni documento.",
        "Se descarta la fecha de nacimiento y solo se conserva la edad en años; el identificador es el consecutivo anonimizado.",
        "No se intenta reidentificar personas; los resultados se reportan agregados por grupos con n elevado.",
        "Las diferencias entre grupos (género, estrato, región) son asociaciones descriptivas, no causalidad, y no deben usarse para estigmatizar poblaciones o instituciones.",
        "Las variables socioeconómicas son autorreportadas por el estudiante al inscribirse.",
    ],
    "limitaciones": [
        "El conjunto abierto solo llega al periodo 2022; no incluye acreditación institucional ni región oficial del ICFES.",
        "No contiene horas de estudio ni puntaje global oficial.",
        "Estrato y educación de los padres son autorreportados y tienen valores faltantes (ver nulos_antes_despues.csv).",
    ],
}
(C.TABLAS / "ficha_tecnica.json").write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------- Diccionario de variables
DESC = {
    "id_anonimo": ("Identificador", "Consecutivo anonimizado de la presentación de la prueba", "estu_consecutivo"),
    "periodo": ("Categórica ordinal", "Periodo de aplicación (AAAAP)", "periodo"),
    "anio": ("Numérica discreta", "Año de aplicación (derivada de periodo)", "derivada"),
    "genero": ("Categórica binaria", "Género reportado (F/M)", "estu_genero"),
    "edad": ("Numérica continua", "Edad en años al momento de la prueba; fuera de 16-70 = nulo", "derivada de estu_fechanacimiento"),
    "depto_reside": ("Categórica nominal", "Departamento de residencia (exterior agrupado en EXTRANJERO)", "estu_depto_reside"),
    "inst_cod": ("Categórica nominal", "Código de la institución de educación superior", "inst_cod_institucion"),
    "inst_nombre": ("Categórica nominal", "Nombre de la institución", "inst_nombre_institucion"),
    "inst_caracter": ("Categórica nominal", "Carácter académico de la institución", "inst_caracter_academico"),
    "inst_origen": ("Categórica nominal", "Naturaleza jurídica (oficial / no oficial)", "inst_origen"),
    "sector": ("Categórica nominal", "Oficial / No oficial / Régimen especial (derivada de inst_origen)", "derivada"),
    "depto_inst": ("Categórica nominal", "Departamento de la institución", "estu_inst_departamento"),
    "region_inst": ("Categórica nominal", "Región natural de la institución (mapeo propio, D-06)", "derivada"),
    "nucleo": ("Categórica nominal", "Núcleo básico del conocimiento del programa", "estu_nucleo_pregrado"),
    "programa": ("Categórica nominal", "Programa académico", "estu_prgm_academico"),
    "modalidad": ("Categórica nominal", "Modalidad del programa: presencial, distancia, distancia virtual", "estu_metodo_prgm"),
    "matricula": ("Categórica ordinal", "Valor de la matrícula semestral (rangos)", "estu_valormatriculauniversidad"),
    "matricula_ord": ("Numérica ordinal", "Matrícula codificada 0 (no pagó) a 7 (más de 7 millones)", "derivada"),
    "horas_trabajo": ("Categórica ordinal", "Horas semanales que trabaja el estudiante", "estu_horassemanatrabaja"),
    "horas_trabajo_ord": ("Numérica ordinal", "Horas de trabajo codificadas 0 a 4", "derivada"),
    "pago_beca": ("Categórica binaria", "Pagó la matrícula con beca", "estu_pagomatriculabeca"),
    "pago_credito": ("Categórica binaria", "Pagó la matrícula con crédito", "estu_pagomatriculacredito"),
    "pago_padres": ("Categórica binaria", "Pagó la matrícula con recursos de los padres", "estu_pagomatriculapadres"),
    "pago_propio": ("Categórica binaria", "Pagó la matrícula con recursos propios", "estu_pagomatriculapropio"),
    "estrato": ("Categórica ordinal", "Estrato de la vivienda (Estrato 1-6 o Sin Estrato)", "fami_estratovivienda"),
    "estrato_num": ("Numérica ordinal", "Estrato 1-6 (Sin Estrato = nulo)", "derivada"),
    "educ_madre": ("Categórica ordinal", "Nivel educativo de la madre", "fami_educacionmadre"),
    "educ_madre_ord": ("Numérica ordinal", "Educación de la madre codificada 0-9 (No sabe / No aplica = nulo)", "derivada"),
    "educ_padre": ("Categórica ordinal", "Nivel educativo del padre", "fami_educacionpadre"),
    "educ_padre_ord": ("Numérica ordinal", "Educación del padre codificada 0-9", "derivada"),
    "tiene_automovil": ("Binaria 0/1", "El hogar tiene automóvil", "fami_tieneautomovil"),
    "tiene_lavadora": ("Binaria 0/1", "El hogar tiene lavadora", "fami_tienelavadora"),
    "tiene_computador": ("Binaria 0/1", "El hogar tiene computador", "fami_tienecomputador"),
    "tiene_internet": ("Binaria 0/1", "El hogar tiene internet", "fami_tieneinternet"),
    "pt_razonamiento": ("Numérica continua", "Puntaje módulo Razonamiento cuantitativo (0 = no calificado -> nulo)", "mod_razona_cuantitat_punt"),
    "pt_lectura": ("Numérica continua", "Puntaje módulo Lectura crítica", "mod_lectura_critica_punt"),
    "pt_ciudadanas": ("Numérica continua", "Puntaje módulo Competencias ciudadanas", "mod_competen_ciudada_punt"),
    "pt_ingles": ("Numérica continua", "Puntaje módulo Inglés", "mod_ingles_punt"),
    "pt_escrita": ("Numérica continua", "Puntaje módulo Comunicación escrita (0 = no calificado -> nulo)", "mod_comuni_escrita_punt"),
    "ingles_nivel": ("Categórica ordinal", "Nivel de desempeño en inglés (-A1, A1, A2, B1, B2)", "mod_ingles_desem"),
    "escrita_nivel": ("Categórica ordinal", "Nivel de desempeño en comunicación escrita (1-4)", "mod_comuni_escrita_desem"),
    "n_modulos_validos": ("Numérica discreta", "Cantidad de módulos genéricos con puntaje válido (0-5)", "derivada"),
    "puntaje_global": ("Numérica continua", "VARIABLE OBJETIVO: promedio de los 5 módulos (solo si los 5 son válidos)", "derivada"),
    "nivel_global": ("Categórica ordinal", "Cuartil del puntaje global (Q1 bajo ... Q4 alto)", "derivada"),
}
tabla_nulos = pd.read_csv(C.TABLAS / "nulos_antes_despues.csv")
raw_null = dict(zip(tabla_nulos["variable"], tabla_nulos["pct_antes"]))
dic = []
for c in df.columns:
    s = df[c]
    tipo, desc, origen = DESC[c]
    if pd.api.types.is_numeric_dtype(s):
        u = s.dropna().unique()
        if len(u) <= 2:
            rango = ", ".join(f"{v:g}" for v in sorted(u))
        elif len(u) <= 12 and (s.dropna() % 1 == 0).all():
            rango = f"{s.min():g} – {s.max():g}"
        else:
            rango = f"{s.min():g} – {s.max():g}  (media {s.mean():.1f})".replace(".", ",")
    else:
        vals = sorted(s.dropna().astype(str).unique())
        texto = ", ".join(vals)
        rango = texto if (len(vals) <= 5 and len(texto) <= 90) else f"{len(vals)} categorías"
    dic.append({"variable": c, "tipo": tipo, "descripcion": desc, "origen": origen, "rango_o_categorias": rango,
                "pct_nulos_crudo": raw_null.get(c, np.nan), "pct_nulos_limpio": round(100 * s.isna().mean(), 2)})
pd.DataFrame(dic).to_csv(C.TABLAS / "diccionario_variables.csv", index=False, encoding="utf-8-sig")

print(pd.DataFrame(pasos).to_string())
print("\nRegistros finales:", len(df), "| columnas:", df.shape[1])
