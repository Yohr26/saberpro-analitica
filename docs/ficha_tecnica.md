# Ficha técnica del conjunto de datos

| Campo | Detalle |
|---|---|
| Nombre | Resultados únicos Saber Pro |
| Entidad | Instituto Colombiano para la Evaluación de la Educación (ICFES) |
| Portal | Datos Abiertos Colombia (datos.gov.co) (identificador `u37r-hjmu`) |
| URL | https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-Pro/u37r-hjmu |
| Fecha de descarga | 2026-10-14 |
| Método | API SODA (CSV) con filtro periodo 20201-20229 y 36 columnas seleccionadas (src/01_descargar_datos.py) |
| Periodos | 20202, 20203, 20212, 20213, 20222, 20223, 20225, 20226 (años 2020–2022) |
| Registros descargados | 721,129 |
| Registros tras limpieza | 699,843 |
| Variables | 36 descargadas → 44 finales |
| Unidad de análisis | Una presentación de la prueba Saber Pro por estudiante (pregrado universitario) |
| Variable objetivo | puntaje_global (derivado: promedio de los 5 módulos genéricos, escala 0-300) |

## Consideraciones éticas

* Datos abiertos y anonimizados publicados por el ICFES; no contienen nombre ni documento.
* Se descarta la fecha de nacimiento y solo se conserva la edad en años; el identificador es el consecutivo anonimizado.
* No se intenta reidentificar personas; los resultados se reportan agregados por grupos con n elevado.
* Las diferencias entre grupos (género, estrato, región) son asociaciones descriptivas, no causalidad, y no deben usarse para estigmatizar poblaciones o instituciones.
* Las variables socioeconómicas son autorreportadas por el estudiante al inscribirse.

## Limitaciones

* El conjunto abierto solo llega al periodo 2022; no incluye acreditación institucional ni región oficial del ICFES.
* No contiene horas de estudio ni puntaje global oficial.
* Estrato y educación de los padres son autorreportados y tienen valores faltantes (ver nulos_antes_despues.csv).

## Trazabilidad de la limpieza

| Paso | Descripción | Registros antes | Registros después | Afectados |
|---|---|---|---|---|
| D-01 | Eliminar registros duplicados por estu_consecutivo | 721,129 | 716,103 | 5,026 |
| D-02 | Eliminar registros sin NINGUN dato de contexto familiar {'20212': 5235, '20225': 4373, '20203': 3557, '20222': 2281, '20223': 673, '20213': 94, '20202': 24, '20226': 23} | 716,103 | 699,843 | 16,260 |
| D-03 | Agrupar ciudades del exterior de estu_depto_reside en 'EXTRANJERO'; corregir 'DISTANCIA VITUAL' | 699,843 | 699,843 | 19 |
| D-04 | Puntaje 0 -> NaN (no calificado). Ceros por modulo: {'pt_razonamiento': 62, 'pt_lectura': 120, 'pt_ciudadanas': 313, 'pt_ingles': 441, 'pt_escrita': 38352}. Fuera de [0,300]: 0 | 699,843 | 699,843 | 39,288 |
| D-05 | Edad fuera de [16, 70] o fecha de nacimiento invalida -> NaN (se descarta la fecha) | 699,843 | 699,843 | 329 |
| FIN | Dataset limpio (sin imputar) | 721,129 | 699,843 | 21,286 |

## Diccionario de variables (dataset limpio)

| Variable | Tipo | Descripción | Rango / categorías | % nulos crudo | % nulos limpio |
|---|---|---|---|---|---|
| `id_anonimo` | Identificador | Consecutivo anonimizado de la presentación de la prueba | 699843 categorías | — | 0.00 |
| `periodo` | Categórica ordinal | Periodo de aplicación (AAAAP) | 8 categorías | — | 0.00 |
| `anio` | Numérica discreta | Año de aplicación (derivada de periodo) | 2020 – 2022 | — | 0.00 |
| `genero` | Categórica binaria | Género reportado (F/M) | F, M | 0.01 | 0.01 |
| `edad` | Numérica continua | Edad en años al momento de la prueba; fuera de 16-70 = nulo | 16 – 70  (media 26,5) | 0.00 | 0.05 |
| `depto_reside` | Categórica nominal | Departamento de residencia (exterior agrupado en EXTRANJERO) | 34 categorías | 0.10 | 0.00 |
| `inst_cod` | Categórica nominal | Código de la institución de educación superior | 266 categorías | — | 0.00 |
| `inst_nombre` | Categórica nominal | Nombre de la institución | 269 categorías | — | 0.00 |
| `inst_caracter` | Categórica nominal | Carácter académico de la institución | INSTITUCIÓN TECNOLÓGICA, INSTITUCIÓN UNIVERSITARIA, TÉCNICA PROFESIONAL, UNIVERSIDAD | — | 0.00 |
| `inst_origen` | Categórica nominal | Naturaleza jurídica (oficial / no oficial) | 6 categorías | 0.00 | 0.00 |
| `sector` | Categórica nominal | Oficial / No oficial / Régimen especial (derivada de inst_origen) | No oficial, Oficial, Régimen especial | — | 0.00 |
| `depto_inst` | Categórica nominal | Departamento de la institución | 26 categorías | — | 0.00 |
| `region_inst` | Categórica nominal | Región natural de la institución (mapeo propio, D-06) | Amazonía, Andina, Caribe, Orinoquía, Pacífica | — | 0.00 |
| `nucleo` | Categórica nominal | Núcleo básico del conocimiento del programa | 56 categorías | — | 0.00 |
| `programa` | Categórica nominal | Programa académico | 988 categorías | — | 0.00 |
| `modalidad` | Categórica nominal | Modalidad del programa: presencial, distancia, distancia virtual | DISTANCIA, DISTANCIA VIRTUAL, PRESENCIAL | — | 0.00 |
| `matricula` | Categórica ordinal | Valor de la matrícula semestral (rangos) | 8 categorías | 0.96 | 0.85 |
| `matricula_ord` | Numérica ordinal | Matrícula codificada 0 (no pagó) a 7 (más de 7 millones) | 0 – 7 | — | 0.85 |
| `horas_trabajo` | Categórica ordinal | Horas semanales que trabaja el estudiante | 0, Entre 11 y 20 horas, Entre 21 y 30 horas, Menos de 10 horas, Más de 30 horas | 3.93 | 1.73 |
| `horas_trabajo_ord` | Numérica ordinal | Horas de trabajo codificadas 0 a 4 | 0 – 4 | — | 1.73 |
| `pago_beca` | Categórica binaria | Pagó la matrícula con beca | No, Si | 0.98 | 0.87 |
| `pago_credito` | Categórica binaria | Pagó la matrícula con crédito | No, Si | 0.98 | 0.87 |
| `pago_padres` | Categórica binaria | Pagó la matrícula con recursos de los padres | No, Si | 0.97 | 0.86 |
| `pago_propio` | Categórica binaria | Pagó la matrícula con recursos propios | No, Si | 0.98 | 0.87 |
| `estrato` | Categórica ordinal | Estrato de la vivienda (Estrato 1-6 o Sin Estrato) | 7 categorías | 3.06 | 0.80 |
| `estrato_num` | Numérica ordinal | Estrato 1-6 (Sin Estrato = nulo) | 1 – 6 | — | 1.19 |
| `educ_madre` | Categórica ordinal | Nivel educativo de la madre | 12 categorías | 2.50 | 0.23 |
| `educ_madre_ord` | Numérica ordinal | Educación de la madre codificada 0-9 (No sabe / No aplica = nulo) | 0 – 9 | — | 0.83 |
| `educ_padre` | Categórica ordinal | Nivel educativo del padre | 12 categorías | 2.39 | 0.12 |
| `educ_padre_ord` | Numérica ordinal | Educación del padre codificada 0-9 | 0 – 9 | — | 4.27 |
| `tiene_automovil` | Binaria 0/1 | El hogar tiene automóvil | 0, 1 | 6.85 | 4.68 |
| `tiene_lavadora` | Binaria 0/1 | El hogar tiene lavadora | 0, 1 | 5.74 | 3.54 |
| `tiene_computador` | Binaria 0/1 | El hogar tiene computador | 0, 1 | 5.35 | 3.14 |
| `tiene_internet` | Binaria 0/1 | El hogar tiene internet | 0, 1 | 3.14 | 0.88 |
| `pt_razonamiento` | Numérica continua | Puntaje módulo Razonamiento cuantitativo (0 = no calificado -> nulo) | 61 – 300  (media 145,8) | 0.00 | 0.01 |
| `pt_lectura` | Numérica continua | Puntaje módulo Lectura crítica | 54 – 300  (media 149,5) | 0.00 | 0.02 |
| `pt_ciudadanas` | Numérica continua | Puntaje módulo Competencias ciudadanas | 57 – 300  (media 147,4) | 0.00 | 0.04 |
| `pt_ingles` | Numérica continua | Puntaje módulo Inglés | 76 – 300  (media 156,5) | 0.01 | 0.07 |
| `pt_escrita` | Numérica continua | Puntaje módulo Comunicación escrita (0 = no calificado -> nulo) | 47 – 300  (media 141,9) | 0.00 | 5.48 |
| `ingles_nivel` | Categórica ordinal | Nivel de desempeño en inglés (-A1, A1, A2, B1, B2) | -A1, A1, A2, B1, B2 | 0.01 | 0.01 |
| `escrita_nivel` | Categórica ordinal | Nivel de desempeño en comunicación escrita (1-4) | 1, 2, 3, 4 | 5.50 | 5.48 |
| `n_modulos_validos` | Numérica discreta | Cantidad de módulos genéricos con puntaje válido (0-5) | 0 – 5 | — | 0.00 |
| `puntaje_global` | Numérica continua | VARIABLE OBJETIVO: promedio de los 5 módulos (solo si los 5 son válidos) | 73,2 – 278,2  (media 148,8) | — | 5.59 |
| `nivel_global` | Categórica ordinal | Cuartil del puntaje global (Q1 bajo ... Q4 alto) | Q1 (bajo), Q2, Q3, Q4 (alto) | — | 5.59 |
