# Sesión 3: Regresión Lineal y Regresión Logística

---

## 1. Regresión Lineal (Predicción Numérica Continua)

> **Propósito:** Predecir un valor numérico continuo (ej: el precio de una vivienda, el tiempo de entrega de un envío) combinando linealmente las características de entrada.

### 1.1. Regresión Lineal Simple (Una sola variable)

Buscamos la ecuación de la recta que mejor se ajuste a los datos:

$$\hat{y} = w \cdot x + b$$

- $x$: Variable predictora (feature).
- $\hat{y}$: Valor predicho por el modelo.
- $w$ (Weight / Pendiente): Cuánto cambia $\hat{y}$ por cada unidad que sube $x$.
- $b$ (Bias / Intercepto / Ordenada al origen): Valor predicho cuando $x = 0$.

### 1.2. Función de Costo: Error Cuadrático Medio (MSE)

Para encontrar los mejores valores de $w$ y $b$, necesitamos una función que mida la magnitud del error de predicción. Usamos el **MSE (Mean Squared Error)**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**¿Por qué al cuadrado?**
1. Evita que los errores positivos ($+5$) y negativos ($-5$) se anulen mutuamente al sumarse.
2. Penaliza con mucha más fuerza los errores grandes que los pequeños (un error de 10 cuesta 100 veces más que un error de 1).

### 1.3. Optimización: Descenso de Gradiente (Gradient Descent)

¿Cómo encuentra el algoritmo los valores óptimos de $w$ y $b$?
1. Inicia con pesos $w$ y sesgo $b$ asignados al azar (o en cero).
2. Calcula el error (MSE) sobre los datos.
3. Calcula el gradiente (derivada) de la función de costo con respecto a cada peso. El gradiente apunta a la dirección de máxima subida del error; por ende, nos movemos en la **dirección opuesta**.
4. Actualiza los parámetros dando un paso proporcional a la tasa de aprendizaje (**learning rate**, $\alpha$):
   $$w \leftarrow w - \alpha \frac{\partial \text{MSE}}{\partial w}$$
5. Itera hasta que el error deja de descender significativamente (convergencia).

> [!NOTE]
> **Tasa de aprendizaje (Learning Rate, $\alpha$):**
> - **Demasiado pequeña:** El modelo tardará miles de iteraciones en aprender.
> - **Demasiado grande:** Puede rebotar descontroladamente y no converger nunca.

### 1.4. Regresión Lineal Múltiple

Generalización a múltiples variables de entrada ($x_1, x_2, \dots, x_n$):

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

- Coeficiente positivo grande: la variable influye positivamente en el resultado.
- Coeficiente negativo: disminuye la predicción.
- Coeficiente cercano a cero: la variable apenas influye.
- *Nota:* Para interpretar correctamente los coeficientes, las variables deben estar previamente normalizadas/escaladas.

### 1.5. Métrica de Evaluación: Coeficiente de Determinación ($R^2$)

Mide qué proporción de la varianza total de los datos es explicada por el modelo:
- $R^2 = 1.0$: Ajuste perfecto (cero error).
- $R^2 = 0.0$: El modelo predice igual de bien que simplemente decir el promedio de $y$ para todos los casos.
- $R^2 < 0.0$: El modelo es peor que predecir el promedio.

### 1.6. Implementación en Scikit-Learn

```python
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Coeficientes aprendidos
print("Pendientes (w):", modelo.coef_)
print("Ordenada al origen (b):", modelo.intercept_)

# Predicciones y métrica R2
y_pred = modelo.predict(X_test)
r2 = modelo.score(X_test, y_test)
```

---

## 2. Regresión Logística (Clasificación Binaria)

A pesar de su nombre, la **Regresión Logística es un modelo de Clasificación**, no de regresión. Predice categorías (0 o 1, Sí o No, Maligno o Benigno).

### 2.1. ¿Por qué no usar Regresión Lineal para Clasificar?

Si intentamos trazar una recta para predecir si un correo es spam ($0$ o $1$):
1. La recta puede arrojar valores fuera de rango: $-2.5$, $1.8$, $+100$. No representan una probabilidad válida entre $[0, 1]$.
2. Un único valor atípico (outlier) lejano movería toda la recta drásticamente, alterando todas las decisiones.

```
       Probabilidad ^
                   |
                1.0|               /---------- (con la recta normal se pasa de 1)
                   |             /
                0.5|           / 
                   |         /
                0.0|-------/-----------------> Valor de X
                   |     / (se va a números negativos)
```

### 2.2. La Función Sigmoide (Sigmoid / Logistic Function)

Para solucionar esto, aplicamos una función que comprime cualquier número real de $(-\infty, +\infty)$ exactamente dentro del intervalo $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
                1.0 |                _-------------
                    |             .-'
                0.5 |           .'
                    |        .-'
                0.0 | ______'
                    +-----------------------------
                           z=0 (50% de probabilidad)
```

- Si $z \to +\infty \implies \sigma(z) \to 1$
- Si $z \to -\infty \implies \sigma(z) \to 0$
- Si $z = 0 \implies \sigma(z) = 0.5$

### 2.3. El Modelo de Regresión Logística

Pasamos la combinación lineal por la sigmoide para predecir la probabilidad de que una instancia pertenezca a la clase positiva ($y = 1$):

$$\hat{p} = \sigma(w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b) = P(y = 1 \mid X)$$

### 2.4. Umbral de Decisión (Decision Threshold)

Por defecto, se asigna la clase $1$ si la probabilidad calculada es mayor o igual a $0.5$:
$$\hat{y} = \begin{cases} 1 & \text{si } \hat{p} \ge 0.5 \\ 0 & \text{si } \hat{p} < 0.5 \end{cases}$$

> [!IMPORTANT]
> El umbral $0.5$ es una convención, pero puede modificarse:
> - En diagnóstico médico de cáncer, podemos bajar el umbral a $0.3$ para asegurarnos de no omitir ningún caso sospechoso (priorizando Recall).

### 2.5. Función de Costo: Pérdida Logística (Log Loss / Binary Cross-Entropy)

No se puede usar MSE porque genera una superficie no convexa (llena de mínimos locales). En su lugar se utiliza **Log Loss**:

$$\text{Log Loss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

- Si la clase real es $1$ y el modelo predice $\hat{p} = 0.99$, la pérdida es casi $0$.
- Si la clase real es $1$ y el modelo predice $\hat{p} = 0.01$ (equivocación con certeza), la penalización matemática tiende a infinito.

### 2.6. Clasificación Multiclase
Cuando hay más de 2 clases posibles:
1. **One-vs-Rest (OvR):** Entrena un clasificador binario por cada clase frente a todas las demás juntas.
2. **Softmax (Regresión Logística Multinomial):** Generalización directa de la sigmoide para un vector de $k$ probabilidades que suman exactamente $1$.

### 2.7. Implementación en Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

# 1. Instanciar y entrenar
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train, y_train)

# 2. Predicción de clases (0 o 1)
y_pred = modelo.predict(X_test)

# 3. Predicción de probabilidades exactas [P(clase=0), P(clase=1)]
probabilidades = modelo.predict_proba(X_test)

# 4. Evaluación (Accuracy sobre test)
exactitud = modelo.score(X_test, y_test)
```
