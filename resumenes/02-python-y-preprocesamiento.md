# Sesión 2: Python Científico y Preprocesamiento de Datos

---

## 1. El Ecosistema Científico de Python

Las listas y estructuras nativas de Python son lentas para operar millones de números. Para procesamiento de datos y Machine Learning utilizamos tres librerías esenciales:

1. **NumPy:** Computación numérica de alto rendimiento con arreglos multidimensionales (vectores y matrices).
2. **Pandas:** Estructuras tabulares llamadas *DataFrames* para manipular datos con columnas heterogéneas (números, texto, fechas).
3. **Matplotlib:** Generación de gráficos y visualización de datos.

> [!NOTE]
> Casi todo el ecosistema de Machine Learning en Python (incluido **scikit-learn**) está construido internamente sobre NumPy y Pandas.

---

## 2. NumPy: Operaciones Vectorizadas

Un arreglo (`np.array`) opera sobre todos los elementos simultáneamente a nivel de lenguaje C sin escribir bucles `for`:

```python
import numpy as np

# Crear array
edades = np.array([15, 16, 17, 18, 17])

# Operaciones elemento a elemento (vectorización)
dobles = edades * 2     # array([30, 32, 34, 36, 34])

# Estadísticas rápidas
media = edades.mean()   # 16.6
desvio = edades.std()   # Desvío estándar
maximo = edades.max()   # 18

# Filtrado booleano (máscaras)
mayores = edades[edades > 16]  # array([17, 18, 17])
```

---

## 3. Pandas: Tablas de Datos (DataFrames)

Un `DataFrame` representa una tabla bidimensional con nombres en columnas e índices en filas.

### Operaciones más comunes

```python
import pandas as pd

# 1. Cargar archivo CSV
df = pd.read_csv("datos.csv")

# 2. Exploración inicial
df.head()       # Muestra las primeras 5 filas
df.info()       # Resumen de columnas, tipos y valores no nulos
df.describe()   # Estadísticas numéricas (media, desvío, mín, máx, cuartiles)

# 3. Selección de columnas
col_edad = df["edad"]                 # Devuelve una Serie (1 columna)
sub_df = df[["edad", "ingresos"]]     # Devuelve un DataFrame (varias columnas)

# 4. Filtrado de filas por condición
adultos = df[df["edad"] >= 18]

# 5. Agrupación y ordenamiento
df.groupby("ciudad")["ingresos"].mean() # Promedio de ingresos por ciudad
df.sort_values(by="edad", ascending=False)
```

---

## 4. Matplotlib: Visualización

```python
import matplotlib.pyplot as plt

# Crear figura (lienzo) y ejes (gráfico)
fig, ax = plt.subplots(figsize=(6, 4))

# Dispersión (Scatter plot)
ax.scatter(df["edad"], df["ingresos"], color="blue", alpha=0.7)
ax.set_title("Edad vs Ingresos")
ax.set_xlabel("Edad")
ax.set_ylabel("Ingresos")
ax.grid(True)

plt.show()
```

---

## 5. Preprocesamiento de Datos (De datos crudos a datos limpios)

> [!IMPORTANT]
> *"Garbage In, Garbage Out"* (Si alimentas al modelo con basura, obtendrás predicciones basura). Los modelos de Machine Learning requieren datos numéricos limpios y escalados.

### 5.1. Datos Faltantes (Missing Values / `NaN`)
Los valores nulos se generan por sensores que fallan, encuestas vacías, etc.

- **Detección:**
  ```python
  df.isna().sum()   # Cantidad de nulos por columna
  ```
- **Opción A: Eliminar filas o columnas:**
  ```python
  df_limpio = df.dropna()  # Elimina filas con cualquier valor nulo
  ```
- **Opción B: Imputación (rellenar con estadísticos):**
  ```python
  df_relleno = df.fillna(df.mean(numeric_only=True))  # Rellena con la media
  ```

### 5.2. Variables Categóricas (One-Hot Encoding)
Los algoritmos matemáticos no pueden multiplicar palabras ("rojo", "verde", "azul"). Se debe transformar el texto en números.

- **One-Hot Encoding (`pd.get_dummies`):** Crea una columna binaria (0 o 1) por cada categoría presente:
  ```python
  df_encoded = pd.get_dummies(df, columns=["color"], drop_first=True)
  ```

| color original | color_verde | color_azul |
| :---: | :---: | :---: |
| rojo | 0 | 0 |
| verde | 1 | 0 |
| azul | 0 | 1 |

### 5.3. Escalado de Variables (Feature Scaling)
Si una feature como **Sueldo** varía entre 0 y 5.000.000 y otra como **Edad** varía entre 0 y 80, cualquier modelo basado en distancias o gradiente considerará que el Sueldo es miles de veces más relevante simplemente por la magnitud numérica.

- **StandardScaler:** Transforma cada variable para que tenga media $\mu = 0$ y desvío $\sigma = 1$:
  $$z = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)
```

---

## 6. División de Datos: Train / Test Split

Nunca se debe evaluar el rendimiento de un modelo con los mismos datos utilizados para entrenarlo. Sería el equivalente a entregarle las respuestas de un examen a un estudiante antes de tomarle la prueba.

```python
from sklearn.model_selection import train_test_split

# Dividir dataset (80% entrenamiento, 20% testeo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- **X_train, y_train:** Datos conocidos para aprender patrones.
- **X_test, y_test:** Datos reservados que el modelo jamás ve durante el aprendizaje, utilizados para medir su desempeño real en producción.
