# Machine Learning from Scratch

**Regresión lineal y logística implementadas con NumPy**, con descenso de gradiente, regularización y una implementación manual de validación cruzada.

Selección de prácticas de **Daniel Carrasco Jurado**, estudiante de Ingeniería Matemática e Inteligencia Artificial en ICAI (Universidad Pontificia Comillas). El objetivo es mostrar cómo se traducen las ecuaciones de los modelos a código, y contrastar su comportamiento con implementaciones de referencia.

![Regresión lineal y frontera de decisión logística sobre datos sintéticos](demo.png)

## Implementaciones

| Archivo | Contenido |
|---|---|
| [linear_regression.py](linear_regression.py) | Regresión simple y múltiple mediante mínimos cuadrados; R², RMSE y MAE |
| [gradient_descent.py](gradient_descent.py) | Ajuste por descenso de gradiente o ecuación normal; codificación one-hot |
| [logistic_regression.py](logistic_regression.py) | Clasificación binaria, sigmoide, entropía cruzada y penalizaciones L1, L2 y Elastic Net |
| [cross_validation.py](cross_validation.py) | Particiones contiguas k-fold y leave-one-out, media y desviación del score |

Los ajustes de los modelos se calculan con NumPy. Scikit-learn se utiliza para comparar resultados, preparar ejemplos y probar la utilidad de validación cruzada; no sustituye las implementaciones de los modelos.

## Ejecutar

Probado con **Python 3.12**. Desde la carpeta del repositorio:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q -m "not network"
python demo.py
```

La demo genera datos sintéticos con semilla 42, separa entrenamiento y prueba y guarda `demo.png`. No necesita descargar datasets. Resultados de la ejecución verificada:

| Ejemplo | Métrica en prueba |
|---|---:|
| Regresión lineal, mínimos cuadrados | RMSE = 0.5266 |
| Regresión lineal, descenso de gradiente | RMSE = 0.5266 |
| Regresión logística | Accuracy = 0.7467; log-loss = 0.4982 |

Estas cifras ilustran el comportamiento en los datos sintéticos de la demo; no son resultados de un problema real ni una competición de modelos.

## Comprobaciones

**29 pruebas locales superadas**: 24 de las prácticas y 5 comprobaciones adicionales de referencia. Se contrastan los mínimos cuadrados con `LinearRegression`, la convergencia del descenso de gradiente, las probabilidades logísticas sin regularización y los scores de k-fold. Las pruebas originales se conservan con sus importaciones adaptadas a los nombres de archivo.

Hay además una prueba opcional que descarga el cuarteto de Anscombe mediante Seaborn:

```bash
python -m pytest -q -m network
```

Esta prueba requiere acceso a Internet y no forma parte de las 29 comprobaciones locales. Las pruebas validan casos concretos; no garantizan soporte para cualquier entrada.

## Alcance y decisiones matemáticas

- **Mínimos cuadrados:** usa la inversa de `X.T @ X`. Requiere rango completo; columnas colineales pueden producir errores o inestabilidad. El código se conserva como implementación educativa de la ecuación normal.
- **Descenso de gradiente:** la tasa de aprendizaje depende de la escala de las variables. La inicialización usa el generador global de NumPy; fija `np.random.seed(...)` para reproducirla.
- **Regresión logística:** espera etiquetas binarias 0/1. `predict_proba` devuelve un vector con la probabilidad de la clase 1, no una matriz de dos columnas. La sigmoide directa puede emitir avisos con logits extremos.
- **Regularización:** en el código original `C` multiplica la penalización (`C / n`): un valor mayor implica más regularización. Algunos docstrings heredados dicen lo contrario. Aquí `C` **no sigue la convención inversa de scikit-learn**. Se mantiene el algoritmo original y se explicita esta diferencia.
- **One-hot:** es una transformación puntual; no guarda un vocabulario para transformar datos nuevos. No debe ajustarse por separado a entrenamiento y prueba como si fuese un encoder persistente.
- **Validación cruzada:** no baraja ni estratifica; exige arrays de NumPy y `2 <= nFolds <= n_samples`, o `-1` para leave-one-out. Reutiliza el estimador, por lo que su `fit` debe reiniciar el estado. El preprocesamiento aprendido debe ajustarse dentro de cada fold. Los modelos de regresión de este repositorio no implementan `.score()` y no se conectan directamente a esta utilidad.

## Origen y preparación del portfolio

Las prácticas proceden de la asignatura de Aprendizaje Automático, a partir de las plantillas y pruebas docentes de **ICAI-IMAT-ML**. Este repositorio reúne las implementaciones completadas durante el curso; no presenta las plantillas ni los tests docentes como una biblioteca diseñada íntegramente desde cero.

Los cuatro módulos originales se han conservado **sin cambios de contenido**, únicamente con nombres más legibles. La preparación del portfolio añade este README, dependencias, la demo y cinco comprobaciones; adapta los imports de los tests. No se han reescrito los algoritmos. Véase [PROVENANCE.md](PROVENANCE.md) para la correspondencia exacta.

Se publica el núcleo de código comprobado. Los notebooks completos de clase, sus datasets y las salidas antiguas quedan fuera de esta selección.
