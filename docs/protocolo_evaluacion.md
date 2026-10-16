# Protocolo de evaluación analítica

Aplica a la fase de **modelado y evaluación** de CRISP-DM. Se define ahora, antes de entrenar modelos, para evitar elegir métricas después de ver resultados.

## 1. Problema analítico

| Tarea | Definición |
|-------|-----------|
| Regresión (principal) | Predecir `puntaje_global` (0–300) de un estudiante usando solo variables **conocidas antes de la prueba**: institución, carácter, sector, región, modalidad, matrícula, trabajo, estrato, educación de los padres, conectividad, edad, género. **Los puntajes de los módulos no son predictores** (fuga). |
| Clasificación (secundaria) | Detectar estudiantes en **riesgo de bajo desempeño**: cuartil inferior (`puntaje_global` < P25 del entrenamiento = 131,0 puntos). |

## 2. Estrategia de partición

| Conjunto | Datos | Registros | Uso |
|----------|-------|-----------|-----|
| Entrenamiento | 2020 y 2021 | 467.356 | Ajuste de modelos y selección de hiperparámetros |
| Validación interna | Dentro del entrenamiento: 5 pliegues **agrupados por institución** (`inst_cod`) | — | Estimar generalización a instituciones no vistas |
| Prueba (hold-out temporal) | 2022 | 193.356 | Evaluación final, **una sola vez** |

Reglas anti-fuga: (i) el umbral de riesgo y todas las imputaciones/codificaciones se **ajustan solo con entrenamiento**; (ii) no se usan puntajes de módulos como predictores; (iii) no se reajusta nada con la prueba.

## 3. Métricas

| Tipo | Métrica | Por qué |
|------|---------|---------|
| **Principal** | **RMSE** sobre `puntaje_global` en la prueba 2022 (en puntos) | Penaliza errores grandes y se interpreta en la escala del examen |
| Secundaria | MAE, R² | Error típico y varianza explicada |
| Secundaria | AUC-ROC del riesgo de bajo desempeño | Capacidad de ordenar a quienes están en riesgo |
| Secundaria | Recall y precisión al alertar al 25 % con mayor riesgo | Utilidad práctica: una institución solo puede intervenir a una fracción |
| Equidad | Recall por género y por región, comparado con el global | Que el modelo no funcione peor en los grupos que se quiere ayudar |

## 4. Líneas base (calculadas sobre la prueba 2022)

| Línea base | RMSE | MAE | R² | AUC-ROC | Recall @25 % | Precisión @25 % |
|------------|------|-----|----|---------|--------------|-----------------|
| B0 Media global | 24,22 | 19,47 | 0,000 | 0,500 | 0,250 | 0,244 |
| B1 Media por estrato | 23,10 | 18,51 | 0,090 | 0,609 | 0,350 | 0,342 |
| B2 Media por carácter académico | 23,11 | 18,50 | 0,089 | 0,612 | 0,364 | 0,355 |
| **B3 Media por institución** | **20,46** | **16,19** | **0,287** | 0,712 | 0,428 | 0,418 |
| **B4 Regresión lineal múltiple** | 20,91 | 16,58 | 0,255 | **0,719** | **0,455** | **0,444** |

Reproducible con `python src/05_protocolo_evaluacion.py` → `reports/tables/protocolo_lineas_base.csv`.

## 5. Criterios de éxito

El modelo de la fase siguiente se considera exitoso si **cumple todos**:

1. **RMSE ≤ 19,4** en 2022 (≥ 5 % mejor que la mejor línea base, B3 = 20,46).
2. **AUC-ROC ≥ 0,75** (la mejor línea base logra 0,719).
3. **Recall ≥ 0,50 al alertar al 25 %** (la mejor línea base logra 0,455).
4. **Estabilidad:** el RMSE de prueba no difiere más de 10 % del RMSE de validación interna agrupada por institución.
5. **Equidad:** el recall por género y por región queda a ±10 puntos porcentuales del recall global.
6. **Interpretabilidad:** se pueden explicar los factores más influyentes (importancia por permutación o SHAP) y estos son coherentes con el EDA, de modo que respalden recomendaciones de innovación pedagógica.

Si ningún modelo supera a B3/B4 de forma material, el hallazgo se reporta tal cual: significaría que el contexto observable explica poco más allá de la institución, y la recomendación sería de política institucional, no de predicción individual.

## 6. Consideraciones éticas del modelado

Las predicciones **no deben usarse para excluir o clasificar estudiantes**, sino para focalizar apoyos a nivel de programa o institución. Variables como género y estrato se usan para auditar equidad y no para tomar decisiones individuales.
