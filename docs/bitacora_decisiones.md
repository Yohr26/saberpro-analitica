# Bitácora de decisiones metodológicas

Proyecto: *Análisis de las pruebas Saber Pro 2020–2022 para identificar áreas menos atendidas y proponer innovación pedagógica*
Metodología: CRISP-DM (ver `README.md`). Cada decisión indica la evidencia que la motivó y el archivo que la reproduce.

## Fase 1 — Comprensión del negocio y de los datos

| ID | Decisión | Alternativas consideradas | Evidencia / justificación | Dónde se reproduce |
|----|----------|---------------------------|---------------------------|--------------------|
| D-00 | Filtrar el dataset `u37r-hjmu` a los periodos 2020–2022 (códigos 20201–20229) **desde la API**, con 36 de 50 columnas. | Descargar todo y filtrar en Power BI. | El dataset completo tiene 1.217.482 registros (2018–2022). Con el filtro 2020–2022 hay **721.129**. La cifra "1'217.482" del documento v1 correspondía al total sin filtrar. | `src/01_descargar_datos.py` |
| D-01 | Eliminar duplicados por `estu_consecutivo` (−5.026). | Conservar ambos. | Los 5.026 consecutivos repetidos tienen puntajes y atributos idénticos; solo cambia la sede de presentación. Contarlos duplicaría estudiantes. | `src/02_etl_limpieza.py` |
| D-02 | Eliminar registros sin **ningún** dato de contexto familiar (−16.260; 2,3 %). | Conservarlos e imputar todo el bloque. | Imputar siete variables sin ningún ancla sería inventar información. Se verificó que el descarte no sesga mucho: media de 4 módulos 146,4 (descartados) vs 149,8 (conservados); composición por naturaleza jurídica y modalidad similar. | `src/02_etl_limpieza.py` |
| D-03 | Estandarizar texto; agrupar ciudades del exterior en `EXTRANJERO`; corregir `DISTANCIA VITUAL`. | — | 19 registros con ciudades extranjeras como "departamento"; error ortográfico en la fuente. | `src/02_etl_limpieza.py` |
| D-04 | Puntaje **0 → nulo** en los módulos. | Dejar el 0 como valor real; eliminar la fila. | En Comunicación escrita hay 38.352 ceros (5,5 %) y **todos** tienen el nivel de desempeño nulo: son "no calificado", no puntajes reales. En los otros módulos los ceros son 62–441 y quedan a más de 4 desviaciones de la media (percentil 0,1 de Razonamiento = 68). Dejarlos sesgaría a la baja el promedio de Comunicación escrita (134,1 → 141,9). | `reports/tables/atipicos_antes_despues.csv` |
| D-05 | Edad fuera de 16–70 años o fecha inválida → nula (329 casos); se descarta la fecha de nacimiento. | Winsorizar. | El crudo contiene edades de −73 y 2.020 años (errores de captura). Además es un dato personal que no se necesita. | `src/02_etl_limpieza.py` |
| D-06 | **Región** = región natural según el **departamento de la institución** (`estu_inst_departamento`). | Región del ICFES; departamento de residencia o de presentación. | El dataset no trae región. El departamento de presentación no sirve: el 82 % se presentó en Bogotá. La institución es la unidad sobre la que se proponen intervenciones. Residencia queda como análisis de sensibilidad futuro. | `src/config.py` |
| D-07 | `puntaje_global` = promedio simple de los 5 módulos genéricos, **solo si los 5 son válidos** (n = 660.712). | Ponderado como el ICFES; promedio de los módulos disponibles. | El dataset abierto no trae puntaje global. Exigir los 5 evita mezclar promedios de distintos módulos (Comunicación escrita es el más bajo). No es el puntaje oficial. | `src/02_etl_limpieza.py` |
| D-08 | "Sin Estrato" se mantiene como categoría propia; en `estrato_num` es nulo. | Imputarlo. | Es una respuesta válida (2.488 casos con puntaje medio de 139,6), no un faltante. | `src/config.py` |
| D-09 | "No sabe" / "No aplica" en educación de los padres son categorías válidas; en la variable ordinal son nulas. | Tratarlas como faltantes a imputar. | Informan algo distinto de "dato no capturado". | `src/config.py` |
| D-11 | Los valores atípicos por IQR **se marcan pero no se eliminan**. | Eliminarlos o winsorizar. | Tras D-04, ningún puntaje está fuera del rango válido 0–300. Los extremos (0,06–1,4 % por módulo) son desempeños reales y por la escala acotada hay efecto techo en 300. Sensibilidad: al excluirlos, la media de cada módulo cambia entre 0,1 y 2,0 puntos (0,5 en el puntaje global), sin alterar ningún ordenamiento. | `reports/tables/atipicos_antes_despues.csv` |

## Fase 2 — Preparación de los datos: imputación

| ID | Decisión | Evidencia | Dónde |
|----|----------|-----------|-------|
| D-10 | **Protocolo de comparación declarado antes de ver resultados.** Se enmascara al azar el 10 % de los valores conocidos de `estrato`, `educ_madre` y `educ_padre` (3 repeticiones, semilla 42), se imputan con 4 métodos (A moda global, B moda por institución, C1 modelo HistGradientBoosting con la clase más probable, C2 el mismo modelo con muestreo estocástico) y se mide: accuracy, F1 macro, divergencia de Jensen–Shannon con la distribución original y variación del ρ de Spearman con el puntaje. **Criterio:** menor JSD; si dos métodos difieren < 0,0005, gana el de mayor accuracy. | Ganó **C1** en las tres variables (JSD 0,0001–0,0004; accuracy 0,49 en estrato frente a 0,36 de la moda global; |Δρ| 0,002–0,004). Se imputaron 5.607 estratos, 1.575 educ. madre y 845 educ. padre. | `src/03_imputacion.py`, `reports/tables/imputacion_*.csv` |
| D-12 | El puntaje **no** se usa como predictor al imputar. | Usarlo mejoraría la exactitud, pero fabricaría la asociación estrato-puntaje que luego se analiza (circularidad). | `src/03_imputacion.py` |
| D-13 | Las variables con nulos < 5 % que no son el foco (automóvil, lavadora, computador, internet, matrícula, horas de trabajo) **no se imputan**; los modelos de la fase siguiente usarán "faltante" como categoría o eliminación por pares. | Imputarlas todas. | Mantiene el alcance de la comparación de métodos en las variables que pidió la retroalimentación. |

## Fase 3 — EDA (comprensión de los datos)

| ID | Decisión | Justificación |
|----|----------|---------------|
| D-14 | Con n ≈ 660 mil **todo valor-p sale ≈ 0**; se reportan tamaños de efecto (ε² de Kruskal–Wallis, d de Cohen, ΔR²) y se interpreta la magnitud, no la significancia. | Evita concluir "relación fuerte" a partir de p < 0,001. |
| D-15 | Regresión lineal múltiple con errores robustos (HC1) para separar efectos ajustados (R² = 0,278) y ΔR² por bloque. Es **descriptiva**, no causal. | Estrato, matrícula, tipo de institución y región están correlacionados entre sí. |
| D-16 | Se reporta η² por institución (0,314): la institución sola explica el 31 % de la varianza del puntaje. | Mide cuánto del puntaje se asocia al "dónde" frente al "quién". |

## Fase 4 — Diseño de la evaluación (puente a modelado)

| ID | Decisión | Justificación |
|----|----------|---------------|
| D-17 | Partición **temporal**: entrenamiento 2020–2021 (467.356) y prueba 2022 (193.356). Validación interna con 5 pliegues agrupados por institución. | Simula uso real (predecir cohortes futuras) y evita fuga entre estudiantes de una misma institución. |
| D-18 | Líneas base B0–B4 y criterios de éxito en `docs/protocolo_evaluacion.md`. | La retroalimentación exige métrica principal, secundarias, partición, línea base y criterios de éxito. |

## Correcciones al documento v1 (hallazgos al reproducir el análisis)

| Texto v1 | Lo que muestran los datos | Acción |
|----------|---------------------------|--------|
| 1'217.482 / 800.000 / "más de 1 millón" / "700 mil" registros | 721.129 descargados; 699.843 tras limpieza; 660.712 con los 5 módulos | Unificar cifras |
| Puntajes "sobre 200" y "sobre 300" | Escala 0–300 | Unificar |
| Cesar = región Pacífica | Cesar es Caribe; el menor promedio departamental es Chocó (120,4), luego La Guajira (131,0) | Corregir |
| "Los resultados decrecen cada año (de 160 a 143)" | Global: 151,2 (2020) → 146,6 (2021) → 148,8 (2022): cae en 2021 y se recupera parcialmente | Corregir |
| "Razonamiento es el módulo más bajo" | Es el segundo más bajo (145,9); el más bajo es Comunicación escrita (141,9) | Matizar |
| "Hombres superan a mujeres en 3–4 puntos / 4–7 %" | 5,7 puntos globales (3,9 %); 14,6 puntos en Razonamiento | Corregir |
| "Correlación de 68 % entre estrato y horas de estudio" | El dataset **no tiene horas de estudio** (solo horas de trabajo). ρ(estrato, puntaje) = 0,28 | Eliminar y reemplazar |
| Estrato 0: 145 vs estrato 6: 171 | No existe estrato 0. Estrato 1: 139,3; Estrato 6: 172,9 | Corregir |
| Presencial ≈150, a distancia 133, virtual 135 | 152,4 / 134,8 / 138,9. "Distancia" y "distancia virtual" son las dos categorías del dato | Corregir |
