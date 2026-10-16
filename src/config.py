"""Configuracion comun: rutas, constantes y mapeos documentados en docs/bitacora_decisiones.md."""
from pathlib import Path
import unicodedata

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw" / "saberpro_2020_2022.csv"
LIMPIO = RAIZ / "data" / "processed" / "saberpro_limpio.parquet"
IMPUTADO = RAIZ / "data" / "processed" / "saberpro_imputado.parquet"
TABLAS = RAIZ / "reports" / "tables"
FIGURAS = RAIZ / "reports" / "figures"
SEMILLA = 42

# nombre original en datos.gov.co -> nombre en el dataset limpio
RENOMBRAR_MODULOS = {
    "mod_razona_cuantitat_punt": "pt_razonamiento",
    "mod_lectura_critica_punt": "pt_lectura",
    "mod_competen_ciudada_punt": "pt_ciudadanas",
    "mod_ingles_punt": "pt_ingles",
    "mod_comuni_escrita_punt": "pt_escrita",
}
MODULOS = {  # nombre limpio -> etiqueta para tablas y graficos
    "pt_razonamiento": "Razonamiento cuantitativo",
    "pt_lectura": "Lectura crítica",
    "pt_ciudadanas": "Competencias ciudadanas",
    "pt_ingles": "Inglés",
    "pt_escrita": "Comunicación escrita",
}
PUNT = list(MODULOS)


def normalizar(txt: str) -> str:
    """Mayusculas sin tildes (para unir nombres de departamentos)."""
    if not isinstance(txt, str):
        return txt
    t = unicodedata.normalize("NFKD", txt.strip().upper())
    return "".join(c for c in t if not unicodedata.combining(c))


# Regiones naturales de Colombia (decision documentada en la bitacora, D-06).
# Se asigna segun el departamento de la INSTITUCION (estu_inst_departamento).
REGION_POR_DEPTO = {
    "BOGOTA": "Andina", "ANTIOQUIA": "Andina", "BOYACA": "Andina", "CALDAS": "Andina",
    "CUNDINAMARCA": "Andina", "HUILA": "Andina", "NORTE SANTANDER": "Andina",
    "QUINDIO": "Andina", "RISARALDA": "Andina", "SANTANDER": "Andina", "TOLIMA": "Andina",
    "ATLANTICO": "Caribe", "BOLIVAR": "Caribe", "CESAR": "Caribe", "CORDOBA": "Caribe",
    "LA GUAJIRA": "Caribe", "MAGDALENA": "Caribe", "SUCRE": "Caribe", "SAN ANDRES": "Caribe",
    "CAUCA": "Pacífica", "CHOCO": "Pacífica", "NARINO": "Pacífica", "VALLE": "Pacífica",
    "META": "Orinoquía", "CASANARE": "Orinoquía", "ARAUCA": "Orinoquía", "VICHADA": "Orinoquía",
    "CAQUETA": "Amazonía", "PUTUMAYO": "Amazonía", "AMAZONAS": "Amazonía",
    "GUAVIARE": "Amazonía", "GUAINIA": "Amazonía", "VAUPES": "Amazonía",
}

ORD_ESTRATO = {f"Estrato {i}": i for i in range(1, 7)}
ORD_EDUCACION = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
ORD_MATRICULA = {
    "No pagó matrícula": 0, "Menos de 500 mil": 1, "Entre 500 mil y menos de 1 millón": 2,
    "Entre 1 millón y menos de 2.5 millones": 3, "Entre 2.5 millones y menos de 4 millones": 4,
    "Entre 4 millones y menos de 5.5 millones": 5, "Entre 5.5 millones y menos de 7 millones": 6,
    "Más de 7 millones": 7,
}
ORD_HORAS_TRABAJO = {
    "0": 0, "Menos de 10 horas": 1, "Entre 11 y 20 horas": 2,
    "Entre 21 y 30 horas": 3, "Más de 30 horas": 4,
}
