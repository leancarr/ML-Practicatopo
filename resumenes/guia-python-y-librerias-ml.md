# Guía Rápida de Python y Librerías de Machine Learning
*Guía pensada para quienes vienen de otros lenguajes o nunca tocaron Python antes.*

---

## 1. Lo Esencial de Python para no perderse

### 1.1. Sintaxis Básica e Indentación
Python no usa llaves `{}` ni punto y coma `;` para delimitar bloques. **Usa espacios / indentación (típicamente 4 espacios)**.

```python
# Comentarios con '#'
edad = 20           # Entero (int)
precio = 19.99      # Decimal (float)
nombre = "Juan"     # Cadena de texto (string)
activo = True       # Booleano (True o False con mayúscula inicial)

# Condicionales: los dos puntos ':' abren el bloque
if edad >= 18:
    print("Mayor de edad")
else:
    print("Menor de edad")
```

### 1.2. Listas y Diccionarios

```python
# 1. Listas (indexación base 0, permiten tipos mezclados)
numeros = [10, 20, 30, 40, 50]
numeros[0]          # 10 (primer elemento)
numeros[-1]         # 50 (último elemento)

# Slicing [inicio : fin_sin_incluir]
numeros[1:4]        # [20, 30, 40]

# 2. Diccionarios (clave -> valor)
persona = {"nombre": "Lean", "edad": 22, "es_estudiante": True}
print(persona["nombre"]) # "Lean"
```

### 1.3. Importación de Módulos (Aliases estándar)
En Python se usan alias estándar universales:
```python
import numpy as np               # Siempre se renombra a np
import pandas as pd              # Siempre se renombra a pd
import matplotlib.pyplot as plt  # Siempre se renombra a plt
```

---

## 2. NumPy: Arreglos y Operaciones Vectorizadas (`np`)

En Machine Learning casi nunca usamos bucles `for` para operar números. NumPy ejecuta operaciones sobre vectores y matrices en lenguaje C a máxima velocidad.

### 2.1. Creación e Inspección
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
matriz = np.array([[1, 2], [3, 4]])

print(arr.shape)       # (5,) -> 1 dimensión con 5 elementos
print(matriz.shape)    # (2, 2) -> 2 filas, 2 columnas
```

### 2.2. Operaciones Elemento a Elemento (Vectorización)
```python
x = np.array([10, 20, 30])
y = np.array([1, 2, 3])

x + y          # array([11, 22, 33])
x * 2          # array([20, 40, 60])
x > 15         # array([False, True, True]) -> Arreglo de booleanos
```

### 2.3. Filtrado Booleano (Máscaras)
```python
edades = np.array([14, 18, 22, 16, 25])
filtro = edades >= 18
print(edades[filtro])   # array([18, 22, 25])
```

### 2.4. Estadísticas Inmediatas
```python
arr.mean()     # Promedio / Media
arr.std()      # Desvío estándar
arr.min()      # Mínimo
arr.max()      # Máximo
arr.sum()      # Sumatoria total
```

---

## 3. Pandas: Tablas de Datos (`pd.DataFrame`)

Pandas es la herramienta principal para cargar, limpiar y preparar datasets.

### 3.1. Cargar y Explorar un Dataset
```python
import pandas as pd

# Cargar CSV
df = pd.read_csv("dataset.csv")

# Inspección rápida
df.head()        # Primeras 5 filas
df.tail(3)       # Últimas 3 filas
df.info()        # Tipos de datos, columnas y nulos presentes
df.describe()    # Tabla estadística (media, desvío, mín, máx, percentiles)
df.shape         # (filas, columnas)
```

### 3.2. Selección y Filtrado
```python
# Elegir una columna (devuelve una Serie de Pandas)
df["edad"]

# Elegir varias columnas (devuelve un DataFrame)
df[["edad", "precio", "ciudad"]]

# Filtrar filas según condiciones lógicas
# (& = AND, | = OR, ~ = NOT)
filtro = (df["edad"] > 25) & (df["ciudad"] == "Rosario")
df_filtrado = df[filtro]
```

### 3.3. Tratamiento de Valores Faltantes (`NaN`)
```python
df.isna().sum()                    # Cuenta cuántos NaN hay por columna
df_sin_nulos = df.dropna()         # Elimina filas con algún valor nulo
df["edad"] = df["edad"].fillna(df["edad"].mean())  # Imputar con la media
```

### 3.4. Transformación de Categóricas a Numéricas (One-Hot Encoding)
```python
# Convierte columnas con texto a columnas binarias (0 y 1)
df = pd.get_dummies(df, columns=["genero", "zona"], drop_first=True)
```

---

## 4. Matplotlib: Gráficos Rápidos (`plt`)

El estándar moderno en Python es usar la interfaz orientada a objetos: `fig, ax = plt.subplots()`.

```python
import matplotlib.pyplot as plt

# 1. Crear el lienzo (fig) y el sistema de ejes (ax)
fig, ax = plt.subplots(figsize=(8, 4))

# 2. Dibujar según el tipo de gráfico:
# - Dispersión: ax.scatter(x, y)
# - Línea:      ax.plot(x, y)
# - Histograma: ax.hist(datos, bins=20)
ax.scatter(df["edad"], df["salario"], color="crimson", alpha=0.6)

# 3. Etiquetas y título
ax.set_title("Relación entre Edad y Salario")
ax.set_xlabel("Edad (años)")
ax.set_ylabel("Salario ($)")
ax.grid(True)

# 4. Mostrar gráfico
plt.show()
```

---

## 5. Scikit-Learn: La Librería Universal de Machine Learning

Scikit-Learn (`sklearn`) tiene la API más consistente del mundo de la programación. **Todos los algoritmos se usan con exactamente la misma estructura de 4 pasos**.

### 5.1. El Patrón Universal de 4 Pasos

```python
# PASO 1: Importar e instanciar el modelo con sus hiperparámetros
from sklearn.linear_model import LogisticRegression
modelo = LogisticRegression(C=1.0)

# PASO 2: Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train, y_train)

# PASO 3: Generar predicciones sobre datos nuevos
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)  # Solo en clasificadores probabilísticos

# PASO 4: Evaluar el rendimiento
puntaje = modelo.score(X_test, y_test)
```

### 5.2. `fit_transform` vs `transform` (¡Muy importante!)

Los preprocesadores (como `StandardScaler`) aprenden parámetros de los datos:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# EN TRAIN: Aprende la media y el desvío (fit) Y transforma los datos (transform)
X_train_escalado = scaler.fit_transform(X_train)

# EN TEST: ¡SOLO transforma con la media y desvío aprendidos de train!
# JAMÁS usar fit en test para evitar Data Leakage (fuga de datos).
X_test_escalado = scaler.transform(X_test)
```

### 5.3. Tabla Resumen de Modelos en Scikit-Learn

| Tarea | Modelo | Importación |
| :--- | :--- | :--- |
| **Regresión** | Regresión Lineal | `from sklearn.linear_model import LinearRegression` |
| **Clasificación** | Regresión Logística | `from sklearn.linear_model import LogisticRegression` |
| **Clasificación** | Árbol de Decisión | `from sklearn.tree import DecisionTreeClassifier` |
| **Clasificación** | kNN | `from sklearn.neighbors import KNeighborsClassifier` |
| **Clasificación** | SVM | `from sklearn.svm import SVC` |

### 5.4. Métricas de Evaluación Clave
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print(classification_report(y_test, y_pred))
```
