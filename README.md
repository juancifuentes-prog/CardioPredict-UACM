# CardioPredict-UACM

Aplicación experimental de investigación para estimar la probabilidad de presentar **carga aterogénica elevada (≥2 alteraciones)**.

## Versión 2.1 depurada

- n analizado: 55 participantes.
- Desenlace: ≥2 alteraciones entre triglicéridos, LDL-c, HDL-c bajo por sexo, no-HDL-c, IRC I, IRC II y TyG/HDL-c.
- Predictores: IMC, cintura, cuello, % de grasa por BIA, grasa visceral, masa musculoesquelética, ángulo de fase y fuerza de agarre ajustada.
- Algoritmo: regresión logística penalizada L2.
- Validación interna: validación cruzada anidada estratificada.
- AUC-ROC interno: 0.683.

La edad y la respuesta cardiovascular aguda a 5 minutos de bicicleta fueron evaluadas, pero no mejoraron el desempeño y no se incluyeron en el modelo final.

> Prototipo académico. No es una herramienta diagnóstica ni estima riesgo cardiovascular absoluto. Requiere ampliación de muestra y validación externa.
