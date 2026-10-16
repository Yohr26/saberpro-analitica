# Saber Pro 2020–2022: desempeño académico e innovación pedagógica

Proyecto de grado — Especialización en Analítica de Datos (CUN). Línea de profundización: Innovación pedagógica.
Autores: Christian Andrés Murillo Viellar y Yohr Dano Cabrales Montoya · Director: Andrés Felipe Solis Pino.

Repositorio con el **ETL, la limpieza documentada, la comparación de métodos de imputación, el análisis exploratorio multivariado y el protocolo de evaluación** del proyecto. Todo el análisis es reproducible con los scripts de `src/`.

## Metodología: CRISP-DM

| Fase CRISP-DM | Qué se hizo | Evidencia |
|---------------|-------------|-----------|
| 1. Comprensión del negocio | Brechas de calidad en educación superior; pregunta de investigación; objetivos | Documento del proyecto |
| 2. Comprensión de los datos | Ficha técnica, diccionario, calidad de datos, EDA multivariado | `docs/ficha_tecnica.md`, `reports/tables/diccionario_variables.csv`, `src/04_eda_multivariado.py` |
| 3. Preparación de los datos | Duplicados, ceros "no calificado", edad, regiones, imputación comparada | `src/02_etl_limpieza.py`, `src/03_imputacion.py`, `docs/bitacora_decisiones.md` |
| 4. Modelado | **Siguiente entrega** (modelos predictivos y explicativos) | — |
| 5. Evaluación | Protocolo definido y líneas base calculadas | `docs/protocolo_evaluacion.md`, `src/05_protocolo_evaluacion.py` |
| 6. Despliegue | Tres recomendaciones de innovación pedagógica | `docs/hallazgos_y_recomendaciones.md` |

## Cómo reproducirlo

Requiere Python 3.12.

```bash
pip install -r requirements.txt
python src/01_descargar_datos.py      # descarga ~721 mil registros de datos.gov.co (~150 MB, tarda unos minutos)
python src/02_etl_limpieza.py         # limpieza + ficha técnica + diccionario + tablas antes/después
python src/03_imputacion.py           # comparación de 4 métodos de imputación
python src/04_eda_multivariado.py     # figuras y tablas del EDA
python src/05_protocolo_evaluacion.py # partición temporal y líneas base
```

Los datos (`data/`) no se versionan: se regeneran con el paso 1. El notebook `notebooks/01_etl_eda.ipynb` presenta los resultados.

## Estructura

```
src/            scripts numerados del pipeline (01 a 05) y config.py
notebooks/      01_etl_eda.ipynb (resultados y figuras)
docs/           bitácora de decisiones, protocolo de evaluación, ficha técnica, hallazgos
reports/tables  tablas CSV (diccionario, limpieza antes/después, EDA, imputación, líneas base)
reports/figures figuras PNG del EDA
data/           raw/ y processed/ (no versionado)
```

## Resultados principales (n = 660.712 estudiantes con los 5 módulos válidos)

* Puntaje global promedio **148,8** sobre 300; cae de 151,2 (2020) a 146,6 (2021) y se recupera a 148,8 (2022).
* La **institución** explica por sí sola el 31 % de la varianza del puntaje (η² = 0,314). El carácter académico pesa más que el sector: universidades 154,9 vs instituciones universitarias 140,1, tecnológicas 136,0 y técnicas profesionales 135,2; oficial vs no oficial solo difiere 1,9 puntos.
* **Brecha de género concentrada en Razonamiento cuantitativo:** hombres +14,6 puntos (d = 0,46); en Comunicación escrita la diferencia es nula.
* **Estrato:** de 139,3 (estrato 1) a 172,9 (estrato 6). Modalidad presencial 152,4 vs distancia 134,8 y distancia virtual 138,9.
* Departamentos con menor promedio: Chocó (120,4), La Guajira (131,0), Putumayo (132,6) y Sucre (138,1).
* Regresión múltiple (R² = 0,28): estrato, tipo de institución, región, modalidad y conectividad mantienen efectos propios tras ajustar por las demás variables.

Detalle, interpretación y límites en `docs/hallazgos_y_recomendaciones.md`.

## Fuente de datos y ética

ICFES, *Resultados únicos Saber Pro*, [datos.gov.co](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-Pro/u37r-hjmu) (identificador `u37r-hjmu`). Datos abiertos y anonimizados. No se intenta reidentificar a nadie y se descarta la fecha de nacimiento. Los resultados son asociaciones descriptivas, no causales. Ver `docs/ficha_tecnica.md`.
