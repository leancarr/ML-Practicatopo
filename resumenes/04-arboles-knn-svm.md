# Sesión 4: Árboles de Decisión, kNN y Support Vector Machines (SVM)

---

## Comparativa Rápida de los Modelos

| Característica | Regresión Logística | Árbol de Decisión | kNN (Vecinos Cercanos) | SVM (Vectores de Soporte) |
| :--- | :--- | :--- | :--- | :--- |
| **Tipo de Frontera** | Línea recta / Hiperplano | Ortogonal en escalones | Muy flexible / irregular | Recta o Curva (según kernel) |
| **¿Requiere Escalado?** | Sí (conviene mucho) | **No** (inmune a escala) | **Estrictamente Obligatorio** | **Estrictamente Obligatorio** |
| **Interpretabilidad** | Alta (pesos $w$) | **Muy Alta** (reglas if/else) | Baja (caja negra local) | Baja (compleja geométricamente) |
| **Velocidad de Predicción**| Inmediata | Inmediata | Lenta (recalcula distancias)| Rápida / Media |
| **Hiperparámetro Clave**| $C$ (regularización) | `max_depth`, `min_samples_leaf` | $k$ (número de vecinos) | $C$, `kernel`, `gamma` |

---

## 1. Árboles de Decisión (Decision Trees)

Un árbol de decisión formula una secuencia de preguntas binarias (sí / no) sobre las variables de entrada. Cada respuesta dirige la instancia hacia una rama, hasta alcanzar un nodo hoja que contiene la predicción final.

### 1.1. Estructura de un Árbol
- **Nodo Raíz (Root Node):** La primera pregunta que divide el dataset completo.
- **Nodos Internos:** Preguntas intermedias.
- **Nodos Hoja (Leaves):** Nodos terminales donde no se divide más y se emite la clase.
- **Profundidad (`max_depth`):** Número máximo de niveles de decisión desde la raíz hasta la hoja más lejana.

### 1.2. Particiones sobre variables numéricas
Para una variable continua (ej: Edad), el árbol busca un umbral óptimo (ej: ¿Edad $\le 30$?). Esto divide el espacio geométrico mediante líneas paralelas a los ejes cartesianos (en escalón).

### 1.3. Criterios de División: Pureza, Gini y Entropía

El objetivo del algoritmo al crear una partición es maximizar la **pureza** de los nodos resultantes (es decir, que los ejemplos de un nodo pertenezcan todos a la misma clase).

#### Criterio 1: Impureza de Gini
Probabilidad de que un elemento seleccionado al azar sea clasificado incorrectamente si se etiquetara aleatoriamente según la distribución de clases del nodo:

$$\text{Gini} = 1 - \sum_{k} p_k^2$$

- **Nodo totalmente puro (todos clase 1):** $\text{Gini} = 1 - (1^2) = 0$
- **Nodo de máxima impureza (50% y 50% en binario):** $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 0.5$

#### Criterio 2: Entropía (Teoría de la Información de Shannon)
Mide el nivel de desorden, sorpresa o incertidumbre del nodo en unidades de **bits**:

$$H = - \sum_{k} p_k \log_2(p_k)$$

- **Nodo puro:** $H = 0$ bits de incertidumbre.
- **Nodo equilibrado (50%/50%):** $H = -(0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1$ bit.

> [!NOTE]
> **Ganancia de Información (Information Gain):** Reducción en la impureza ponderada que se logra al dividir los datos según una pregunta candidata. El árbol evalúa todas las variables y posibles umbrales y selecciona aquel que maximiza dicha ganancia.

### 1.4. Control del Sobreajuste (Overfitting)

Si no se restringe, el árbol continuará dividiéndose hasta que cada hoja tenga un solo dato, memorizando ruido con 100% de exactitud en train y fracasando en test.

**Hiperparámetros de poda y control:**
- `max_depth`: Limita la profundidad máxima del árbol.
- `min_samples_split`: Mínimo de muestras requeridas para permitir otra división.
- `min_samples_leaf`: Mínimo de muestras permitidas en un nodo hoja.

### 1.5. Ventajas y Desventajas
- **Ventajas:** Fácil de interpretar y graficar; no requiere escalado de features; maneja variables numéricas y categóricas; permite extraer la importancia relativa de variables (`feature_importances_`).
- **Desventajas:** Alta tendencia al overfitting; muy inestable (un pequeño cambio en los datos de train puede cambiar todo el árbol).
- *Solución estándar de la industria:* **Random Forest** (ensamble de cientos de árboles distintos que votan en conjunto).

### 1.6. Código Scikit-Learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

modelo = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
modelo.fit(X_train, y_train)

# Gráfico del árbol generado
fig, ax = plt.subplots(figsize=(12, 6))
plot_tree(modelo, feature_names=X.columns, class_names=["Clase 0", "Clase 1"], filled=True)
plt.show()
```

---

## 2. Algoritmo k-Nearest Neighbors (kNN)

> **Principio de Funcionamiento:** "Dime con quién andas y te diré quién eres". Para predecir la etiqueta de una muestra nueva, el algoritmo busca los $k$ puntos más cercanos en el conjunto de entrenamiento y asigna la clase más votada.

### 2.1. Procedimiento de Inferencia
1. Llega un nuevo punto $x_{\text{nuevo}}$.
2. Se calcula la distancia euclídea contra **todos y cada uno** de los puntos de entrenamiento:
   $$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^n (a_i - b_i)^2}$$
3. Se seleccionan los $k$ puntos con menor distancia.
4. Las etiquetas de esos $k$ vecinos votan; gana la mayoría.

### 2.2. Por qué el Escalado es OBLIGATORIO en kNN
Si una variable (Ingresos) se mide en miles y otra (Edad) entre 18 y 80, la fórmula de distancia euclídea quedará dominada en un 99.9% por los ingresos, ignorando por completo la edad. **Siempre se debe aplicar `StandardScaler`**.

### 2.3. Elección del Hiperparámetro $k$
- **$k = 1$:** Memoriza cada punto de entrenamiento (muy ruidoso, frontera extremadamente irregular $\implies$ **Overfitting**).
- **$k$ excesivamente grande:** Promedia todo el dataset, prediciendo siempre la clase mayoritaria ($\implies$ **Underfitting**).
- Se recomienda usar números impares ($k=3, 5, 7$) para evitar empates en problemas binarios.

> [!NOTE]
> A kNN se lo clasifica como un **Lazy Learner** (aprendiz perezoso): no realiza cálculos pesados durante el `fit` (solo guarda los datos en memoria), pero su etapa de predicción es muy lenta si el dataset tiene millones de filas.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_escalado, y_train)
y_pred = knn.predict(X_test_escalado)
```

---

## 3. Support Vector Machines (SVM / Máquinas de Vectores de Soporte)

> **Principio de Funcionamiento:** Si dos clases son separables linealmente, existen infinitos hiperplanos válidos. SVM busca aquel que **maximiza el margen de separación** entre ambas clases.

### 3.1. Margen y Vectores de Soporte
- **Margen:** La distancia perpendicular entre el hiperplano separador y los datos más cercanos de cada clase.
- **Vectores de Soporte (Support Vectors):** Son exactamente los puntos que descansan sobre el borde del margen. **Solo estos puntos determinan la posición del hiperplano**. Si movemos o borramos cualquier otro punto alejado del margen, la frontera no cambia en absoluto.

### 3.2. Margen Blando y el Hiperparámetro $C$
En datos reales, las clases casi siempre se superponen por ruido:
- **$C$ Grande (Margen Estricto):** Penaliza severamente cualquier error. Margen muy estrecho $\implies$ riesgo de Overfitting.
- **$C$ Pequeño (Margen Tolerante):** Permite que algunos puntos invadan el margen para lograr una frontera más suave y robusta $\implies$ mayor capacidad de generalización.

### 3.3. El Truco del Kernel (Kernel Trick)
Cuando los datos no son separables por una recta plana, el Kernel proyecta matemáticamente los datos a una dimensión superior donde sí sean separables, calculando productos internos sin tener que computar explícitamente las coordenadas de esa dimensión:
- `kernel="linear"`: Hiperplano plano tradicional.
- `kernel="poly"`: Fronteras curvas polinómicas.
- `kernel="rbf"` (Radial Basis Function): El más versátil, permite generar islas cerradas y contornos muy adaptables.

> [!IMPORTANT]
> SVM se basa enteramente en distancias geométricas para calcular márgenes; por lo tanto, **el escalado de datos es imprescindible**.

```python
from sklearn.svm import SVC

# Instanciación con kernel gaussiano RBF
svm = SVC(kernel="rbf", C=1.0)
svm.fit(X_train_escalado, y_train)

# Vectores de soporte que definieron el margen
print("Número de vectores de soporte:", len(svm.support_vectors_))
```
