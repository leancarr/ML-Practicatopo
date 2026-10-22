# Guía de Configuración de Entorno (Windows y Pop!_OS / Linux)
*Cómo solucionar errores de módulos (`ModuleNotFoundError`), configurar Jupyter en VS Code y sincronizar el trabajo entre tu PC y tu notebook.*

---

## 1. ¿Por qué ocurre el error `ModuleNotFoundError: No module named 'numpy'`?

Cuando abrís un Notebook (`.ipynb`) en VS Code, el editor debe conectarse a un **intérprete de Python** (llamado *Kernel*) para ejecutar el código.

Si VS Code toma el Python del sistema recién instalado (que viene vacío) en lugar del entorno donde están instaladas las librerías, Python intentará buscar `numpy`, `pandas`, `sklearn`, etc., no las encontrará y fallará.

Para solucionarlo existen dos caminos:
1. **Método 1 (Recomendado / Estándar Profesional):** Crear un Entorno Virtual (`.venv`).
2. **Método 2 (Rápido / Directo):** Instalar las dependencias en el Python global.

---

## 2. Método 1: Entorno Virtual `.venv` (Recomendado)

> [!NOTE]
> **¿Por qué usar un entorno virtual?**  
> Un entorno virtual es una "caja aislada" para este proyecto. Evita que las librerías de Machine Learning interfieran con otros programas o con el sistema operativo.

---

### A) En Pop!_OS / Linux (Tu Notebook)

Pop!_OS está basado en Ubuntu/Debian. En Linux moderno (Python 3.11/3.12+), el sistema operativo protege el Python global con una regla llamada `externally-managed-environment`. Por eso, **en Linux siempre debés usar entorno virtual**.

#### Paso 1: Instalar herramientas base (solo la primera vez)
Abrí la terminal de Pop!_OS (`Super + T` o buscando Terminal):
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

#### Paso 2: Clonar tu repositorio en Pop!_OS
```bash
git clone https://github.com/leancarr/ML-Practicatopo.git
cd ML-Practicatopo
```

#### Paso 3: Crear y activar el entorno virtual
```bash
# Crear la carpeta .venv
python3 -m venv .venv

# Activar el entorno
source .venv/bin/activate
```
*(Al activarse, vas a ver que tu terminal muestra `(.venv)` a la izquierda).*

#### Paso 4: Instalar las dependencias
Con el entorno activado:
```bash
pip install -r Intro-A-ML/requirements.txt
```

---

### B) En Windows (Tu PC de Escritorio)

#### Paso 1: Crear y activar el entorno virtual
Desde PowerShell o terminal de VS Code:
```powershell
# Crear el entorno
python -m venv .venv

# Activar el entorno
.\.venv\Scripts\Activate.ps1
```

> [!TIP]
> Si PowerShell te dice *"la ejecución de scripts está deshabilitada en este sistema"*, ejecutá una sola vez:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> y volvé a correr `.\.venv\Scripts\Activate.ps1`.

#### Paso 2: Instalar las librerías
```powershell
pip install -r Intro-A-ML/requirements.txt
```

---

## 3. Cómo conectar el Entorno Virtual a VS Code (Windows y Linux)

Una vez creado el `.venv`, tenés que indicarle al notebook que use ese kernel:

1. Abrí VS Code y asegurate de tener instaladas las extensiones:
   - **Python** (de Microsoft)
   - **Jupyter** (de Microsoft)
2. Abrí cualquier archivo `.ipynb`.
3. Arriba a la derecha, hacé clic en **"Select Kernel"** (o en la versión de Python que figure).
4. Seleccioná **Python Environments...**
5. Elegí el que dice **`.venv`** (suele decir *Recommended* o apuntar a la carpeta del proyecto).

#### ¿Qué hacer si VS Code NO encuentra el `.venv` automáticamente?
1. Hacé clic en **"Select Another Kernel..."**.
2. Elegí **"Python Environments..."** $\to$ **"Find..."** / **"Enter interpreter path..."**.
3. Seleccioná manualmente el archivo ejecutable de Python de tu entorno:
   - En **Windows:** `MACHINE-Topo\.venv\Scripts\python.exe`
   - En **Pop!_OS:** `ML-Practicatopo/.venv/bin/python`

---

## 4. Método 2: Instalación Global Directa (La solución rápida)

Si alguna vez tenés apuro y no querés configurar entornos virtuales:

### En Windows:
```powershell
python -m pip install -r Intro-A-ML/requirements.txt
```
Luego seleccionás tu Python global (`Python 3.10...`) arriba a la derecha en el notebook y ya tiene todo instalado.

### En Pop!_OS:
Si intentás instalar globalmente en Pop!_OS moderno, te tirará:
`error: externally-managed-environment`  
Para forzarlo (no recomendado a largo plazo, pero funciona si estás apurado):
```bash
pip3 install --break-system-packages -r Intro-A-ML/requirements.txt
```

---

## 5. Flujo de Trabajo para pasar de una computadora a otra

```mermaid
sequenceDiagram
    participant PC as PC Windows
    participant GH as GitHub (ML-Practicatopo)
    participant Note as Notebook Pop!_OS

    Note over PC: Hacés ejercicios en clase/casa
    PC->>GH: git add . && git commit && git push
    Note over GH: Cambios guardados en la nube
    Note over Note: Abrís la notebook en la facu
    GH->>Note: git pull
    Note over Note: Resolvés ejercicios en Pop!_OS
    Note->>GH: git add . && git commit && git push
```

### Reglas clave para no tener conflictos:

1. **La carpeta `.venv` NUNCA se sube a GitHub:**  
   Ya está incluida en el `.gitignore`. El `.venv` de Windows no funciona en Linux (los binarios son distintos). Cada máquina tiene su propio `.venv`.
2. **Al terminar en una máquina:**
   ```bash
   git add .
   git commit -m "feat: avance ejercicios clase X"
   git push
   ```
3. **Al sentarte en la otra máquina antes de empezar:**
   ```bash
   git pull
   ```
