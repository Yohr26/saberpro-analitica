# Hallazgos del EDA y recomendaciones de innovación pedagógica

n = 660.712 estudiantes con los 5 módulos válidos (2020–2022). Puntaje global = promedio de los 5 módulos genéricos (0–300). Todas las cifras salen de `reports/tables/`.

## 1. Decisión sobre las hipótesis

| Hipótesis | Veredicto | Evidencia |
|-----------|-----------|-----------|
| **H1.** Las instituciones de mayor tradición y acreditación obtienen los puntajes más altos, con una brecha fuerte frente a las demás. | **Se confirma en parte.** Lo que separa a las instituciones es el *carácter académico* y la selectividad, no que sean oficiales o privadas. La acreditación no está en el dataset, así que no se puede probar directamente. | Universidad 154,9 vs inst. universitaria 140,1, tecnológica 136,0 y técnica profesional 135,2 (ε² = 0,094, efecto mediano). Oficial 150,1 vs no oficial 148,2 (ε² = 0,002, insignificante). Los 10 puntajes más altos son Andes (191,7), Nacional-Bogotá (186,0), CESA, EIA, Sabana, etc.; la institución sola explica el 31,4 % de la varianza. |
| **H2a.** Razonamiento cuantitativo es el módulo que más baja el promedio. | **Se confirma en parte.** Es el segundo más bajo; el más bajo es Comunicación escrita. | Medias: Escrita 141,9 · Razonamiento 145,9 · Ciudadanas 147,4 · Lectura 149,5 · Inglés 156,5. Razonamiento < Lectura (−3,6), < Ciudadanas (−1,6), < Inglés (−10,5), > Escrita (+4,8). |
| **H2b.** Las mujeres tienen puntajes más bajos que los hombres en Razonamiento cuantitativo. | **Se confirma, y es donde la brecha es mayor.** | Razonamiento: mujeres 140,7 vs hombres 155,3 (−14,6; d = 0,46). Inglés −6,1, Ciudadanas −5,4, Lectura −2,7; Comunicación escrita ≈ 0 (−0,14). Global: −5,7 (d = 0,23). |

## 2. Otros hallazgos

* **Tendencia anual:** global 151,2 (2020) → 146,6 (2021) → 148,8 (2022). Cae en 2021 (año de pandemia) y se recupera parcialmente; no es un descenso continuo.
* **Estrato:** 139,3 (1) · 145,8 (2) · 151,4 (3) · 161,1 (4) · 167,2 (5) · 172,9 (6). Ajustando por las demás variables, el estrato 6 supera al 1 en +16,3 puntos. El módulo más sensible al estrato es Inglés (ρ = 0,38).
* **Modalidad:** presencial 152,4 · distancia virtual 138,9 · distancia 134,8. Ajustado: distancia −7,5 vs presencial.
* **Territorio:** departamentos más bajos: Chocó 120,4 · La Guajira 131,0 · Putumayo 132,6 · Sucre 138,1 · Caquetá 139,4 · Cesar 140,1. Por región, Andina 150,1 · Pacífica 147,3 · Orinoquía 147,2 · Caribe 144,1 · Amazonía 138,0. Las diferencias *entre* regiones son pequeñas (ε² = 0,009) porque dentro de cada región conviven instituciones muy distintas.
* **Familia y recursos:** matrícula (ε² = 0,135), educación de la madre (0,099) y del padre (0,091) son los factores con mayor efecto. Sin internet: 136,7 vs con internet 150,1.
* **Edad y trabajo:** a mayor edad, menor puntaje (ρ = −0,32; −0,5 puntos por año ajustado). Quienes no trabajan promedian 158,3; quienes trabajan, ~147 sin importar cuántas horas.
* **Correlación de estrato con puntaje:** ρ de Spearman = 0,28 (moderada-baja). El documento v1 citaba 68 % con "horas de estudio", variable que no existe en el dataset.
* **Modelo multivariado (R² = 0,278):** la mayor caída de R² al retirar un bloque corresponde al carácter académico (−0,016), seguido de estrato (−0,013), edad (−0,012), sector (−0,011) y región (−0,011). Aun así, la mayor parte de la variación (72 %) queda sin explicar con las variables contextuales.

## 3. Tres recomendaciones aplicables a la innovación pedagógica

**R1 — Nivelación temprana en razonamiento cuantitativo con enfoque de género.**
*Dato:* la brecha de 14,6 puntos en razonamiento cuantitativo es 5 veces la de lectura crítica (2,7) y desaparece en comunicación escrita.
*Propuesta:* un diagnóstico en el primer semestre y rutas de nivelación (aprendizaje basado en problemas con contextos aplicados, tutorías entre pares y mentoría) para estudiantes con baja base cuantitativa, con seguimiento de la brecha por género.
*Indicador sugerido:* reducir la brecha de género en razonamiento por debajo de 10 puntos en dos cohortes.

**R2 — Rediseñar el acompañamiento en modalidad a distancia y en instituciones técnicas, tecnológicas y universitarias.**
*Dato:* distancia −7,5 puntos y carácter académico distinto de universidad −7 a −13 puntos, aun ajustando por estrato, educación familiar y región.
*Propuesta:* tutoría sincrónica obligatoria en los primeros semestres, evaluaciones formativas frecuentes y un tablero de alerta temprana basado en las variables del protocolo de evaluación (sin usar género ni estrato para decidir sobre individuos).
*Indicador sugerido:* acercar la media de la modalidad a distancia a la presencial.

**R3 — Cerrar brechas territoriales y de conectividad, empezando por inglés.**
*Dato:* Inglés es el módulo más asociado al estrato (ρ = 0,38) y a la matrícula; los departamentos con menor desempeño (Chocó, La Guajira, Putumayo, Sucre) y los estudiantes sin internet (−13,4 puntos) concentran el rezago.
*Propuesta:* programa de inglés y competencias digitales para estratos 1–2 con recursos digitales descargables, préstamo de dispositivos y alianzas entre instituciones de alto y bajo desempeño de la misma región.
*Indicador sugerido:* reducir la distancia entre los 4 departamentos más bajos y la media nacional.

## 4. Límites que deben declararse

* Son **asociaciones**, no causas: estrato, matrícula, institución y región están correlacionados.
* El puntaje global es una construcción propia (promedio simple), no el puntaje oficial del ICFES.
* La acreditación institucional y las horas de estudio no están en el dataset.
* Estrato y educación de los padres son autorreportados.
* Los datos cubren hasta 2022; no permiten afirmar tendencias posteriores.

## 5. Resultados del modelado predictivo (entrega final)

Entrenamiento 2020–2021 (467.356) y prueba 2022 (193.356); todo el preprocesamiento se ajusta solo con el entrenamiento (`docs/bitacora_decisiones.md`, D-19 a D-31).

* **Un modelo de ensamble supera a la línea base institucional.** XGBoost: RMSE 18,57 frente a 20,46 (B3), AUC 0,777 y recall 0,509 al alertar al 25 %. La diferencia es significativa (IC 95 % −2,11 a −1,67 puntos, con errores agrupados por institución).
* **Gran parte del poder predictivo viene de saber a qué institución pertenece el estudiante.** Con instituciones nuevas el RMSE sube a 20,57. SHAP: institución 19,5 %, edad 10,2 %, núcleo del programa 9,0 %, estrato 5,9 %, educación del padre 4,7 %.
* **La ablación confirma que no hay un solo bloque decisivo:** quitar modalidad y núcleo del programa (+0,42 puntos de RMSE), edad/género/trabajo (+0,41) o la identidad institucional (+0,30) es lo que más empeora el modelo; quitar bienes y conectividad del hogar casi no cambia el resultado (+0,04), probablemente porque estrato y matrícula ya recogen parte de esa información.
* **La imputación (método C1) y la selección por Lasso no cambian el resultado predictivo** (variaciones ≤ 0,01 puntos de RMSE con el Lasso suave).
* **Equidad: el criterio no se cumple.** El modelo recupera más mujeres que hombres entre los estudiantes en riesgo (recall 0,584 frente a 0,383), con precisión igual en ambos grupos; la causa es que alerta a menos hombres, que tienen una tasa real de bajo desempeño menor (21,7 % frente a 26,4 %). Por región, Caribe y Pacífica quedan 10–11 pp por encima del global, y Amazonía y Orinoquía (≈ 930 estudiantes cada una) tienen estimaciones muy inestables.

**Implicación para las recomendaciones.** El tablero de alerta temprana de R2 debe usarse para focalizar apoyos por programa o institución, con un monitoreo periódico del recall por género y región; no para decidir sobre estudiantes individuales.
