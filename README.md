# ML-Practicatopo 🚀
*Repositorio personal de práctica y apuntes para el Taller de Introducción a Machine Learning (6º año Informática - Instituto Politécnico Superior "Gral. San Martín").*

---

## 📌 Contexto del Proyecto

* **Estudiante:** Leandro Caraballo ([@leancarr](https://github.com/leancarr))
* **Docente / Material Base:** Agustín Gurvich ([Intro-A-ML](https://github.com/AgustinGurvich/Intro-A-ML))
* **Entornos de Trabajo:**
  * **PC Principal:** Windows 10/11 (`C:\Users\USER\Desktop\MACHINE-Topo`)
  * **Notebook Portátil:** Pop!_OS (Linux)
* **Integración Segundo Cerebro:** Bóveda de Obsidian (`AntigravityBrain/01_Proyectos/Machine-Learning-Topo`)

---

## 📂 Estructura del Repositorio

```text
├── Intro-A-ML/                           # Clases, diapositivas y notebooks originales del docente
│   ├── clases/
│   │   ├── sesion-01-introduccion-y-estadistica/    # [COMPLETADA] Notebook 1 resuelto
│   │   ├── sesion-02-python-y-preprocesamiento/
│   │   ├── sesion-03-regresion-lineal-y-logistica/
│   │   ├── sesion-04-arboles-knn-svm/
│   │   └── sesion-05-evaluacion-de-modelos/
│   └── requirements.txt                  # Dependencias del curso
├── resumenes/                            # Apuntes teóricos en Markdown (sincronizados con Obsidian)
│   ├── 01-introduccion-y-estadistica.md
│   ├── 02-python-y-preprocesamiento.md
│   ├── 03-regresion-lineal-y-logistica.md
│   ├── 04-arboles-knn-svm.md
│   ├── 05-evaluacion-de-modelos.md
│   ├── guia-python-y-librerias-ml.md     # Cheatsheet de NumPy, Pandas, Matplotlib y Scikit-Learn
│   ├── GUIA-SETUP-ENTORNO.md             # Guía paso a paso para Windows y Pop!_OS
│   └── ejercicio-4-tipos-de-aprendizaje.md # Apunte sobre Supervisado, No Supervisado y Refuerzo
├── GUIA-SETUP-ENTORNO.md                 # Copia en raíz para acceso rápido
└── README.md                             # Este archivo de bienvenida y contexto
```

---

## 🛠️ Puesta en Marcha Rápida (Setup de Entorno)

Para ver la guía detallada de resolución de errores y configuración de kernels, consultá [**GUIA-SETUP-ENTORNO.md**](GUIA-SETUP-ENTORNO.md).

### En tu Notebook con Pop!_OS (Linux)
```bash
# 1. Clonar o actualizar el repositorio
git clone https://github.com/leancarr/ML-Practicatopo.git
cd ML-Practicatopo
git pull

# 2. Crear y activar el entorno virtual (obligatorio en Linux)
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar librerías
pip install -r Intro-A-ML/requirements.txt

# 4. En VS Code: seleccionar kernel .venv en la esquina superior derecha del notebook.
```

### En tu PC con Windows
```powershell
# 1. Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r Intro-A-ML/requirements.txt
```

---

## 🗺️ Mapa de Contenidos y Apuntes

| Sesión | Tema Principal | Apunte Teórico | Notebook Práctico | Estado |
| :---: | :--- | :---: | :---: | :---: |
| **01** | Introducción a ML y Estadística | [Apunte 01](resumenes/01-introduccion-y-estadistica.md) | [Notebook 1](Intro-A-ML/clases/sesion-01-introduccion-y-estadistica/notebook-1-introduccion-ml.ipynb) | ✅ Resuelto |
| **02** | Python Científico y Preprocesamiento | [Apunte 02](resumenes/02-python-y-preprocesamiento.md) | Notebook 1 y 2 | ⏳ Pendiente |
| **03** | Regresión Lineal y Logística | [Apunte 03](resumenes/03-regresion-lineal-y-logistica.md) | Notebook 1 y 2 | ⏳ Pendiente |
| **04** | Árboles de Decisión, kNN y SVM | [Apunte 04](resumenes/04-arboles-knn-svm.md) | Notebook 1 y 2 | ⏳ Pendiente |
| **05** | Evaluación de Modelos y Validación | [Apunte 05](resumenes/05-evaluacion-de-modelos.md) | Notebook 1 y 2 | ⏳ Pendiente |

---

## 🧠 Guías y Recursos Extra
* [**Guía Rápida de Python y Librerías ML**](resumenes/guia-python-y-librerias-ml.md): Resumen de sintaxis de Python, manipulación de tablas con Pandas, matrices con NumPy y el flujo `fit / predict / score` de Scikit-Learn.
* [**Ejercicio 4: Tipos de Aprendizaje**](resumenes/ejercicio-4-tipos-de-aprendizaje.md): Justificación y casos del mundo real para modelos supervisados, no supervisados y por refuerzo.
