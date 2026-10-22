# Ejercicio 4: Tipos de Aprendizaje en Machine Learning (Casos Reales y Justificación)

---

## 1. Ejemplos del Mundo Real

### 1.1. Aprendizaje Supervisado
* **Problema del mundo real:** Predecir el precio de venta de autos usados (o evaluar si un cliente devolverá un préstamo bancario).
* **Características de entrada ($X$):** Año, marca, kilometraje, cantidad de puertas, tipo de combustible.
* **Etiqueta objetivo ($y$):** Precio final de venta en dólares.
* **Justificación:** Es supervisado porque contamos con un **dataset histórico donde ya conocemos la respuesta correcta** para miles de ejemplos pasados. El modelo aprende una función matemática que asocia las variables de entrada ($X$) con la salida deseada ($y$). Cuando llega un auto nuevo, el modelo infiere su precio.

---

### 1.2. Aprendizaje No Supervisado
* **Problema del mundo real:** Segmentación y descubrimiento de perfiles de usuarios en plataformas como Spotify o Netflix.
* **Características de entrada ($X$):** Géneros musicales más escuchados, horarios de conexión, porcentaje de canciones salteadas, ritmo (tempo/BPM) preferido.
* **Etiqueta objetivo ($y$):** **No existe**.
* **Justificación:** Es no supervisado porque **no hay una lista de respuestas correctas ni categorías predefinidas** (nadie etiquetó a mano a los usuarios como "oyente melancólico" o "fanático del trap"). El algoritmo (como *K-Means*) analiza las distancias entre usuarios en el espacio de características y agrupa por similitud, descubriendo patrones y tribus de consumo que los humanos no habíamos definido.

---

### 1.3. Aprendizaje por Refuerzo
* **Problema del mundo real:** Sistema autónomo de control y refrigeración eficiente en centros de datos (data centers) de servidores.
* **Elementos del sistema:**
  - **Agente:** El software de control inteligente.
  - **Entorno:** El centro de datos, servidores, temperatura ambiente, carga de trabajo y ventiladores.
  - **Acciones:** Aumentar o disminuir la potencia de enfriamiento, redirigir flujo de aire, apagar sectores ociosos.
  - **Recompensa / Castigo:** Premio numérico positivo ($+1$) por cada kWh ahorrado en la factura eléctrica; penalización severa ($-1000$) si la temperatura de un servidor supera los $27^\circ\text{C}$ de seguridad.
* **Justificación:** Es por refuerzo porque **no hay una tabla estática de datos con la "jugada perfecta" para cada segundo**. El agente aprende mediante **ensayo y error interactivo**, tomando decisiones sucesivas para maximizar su recompensa acumulada a largo plazo.

---

## 2. Marco Práctico: ¿Cómo saber qué tipo de problema tengo sin probar a ciegas?

La categoría del problema **NO depende del algoritmo**, sino de **qué datos tenés y cuál es el objetivo**:

| Pregunta Clave | Si la respuesta es SÍ... | Categoría |
| :--- | :--- | :--- |
| ¿Tengo una columna histórica con la **respuesta correcta** que quiero predecir? | Datos con pares $(X, y)$ | **Supervisado** |
| ¿Tengo datos ($X$) pero **ninguna respuesta** y quiero descubrir agrupamientos u orden oculto? | Solo datos $(X)$ | **No Supervisado** |
| ¿No hay tabla fija, sino un **entorno interactivo** donde tomar decisiones con **premios y castigos**? | Agente + Entorno + Recompensas | **Por Refuerzo** |

---

## 3. Realidad en la Industria: ¿Por qué el Refuerzo es para casos tan puntuales?

* **Supervisado (~80-85%):** Es el rey indiscutido de las empresas porque la mayoría de los negocios tienen bases de datos SQL / planillas con historia acumulada.
* **No Supervisado (~10-15%):** Muy utilizado para segmentar clientes, detectar fraudes atípicos y sistemas de recomendación.
* **Por Refuerzo (<5% tradicionalmente):**
  - Es el más costoso computacionalmente.
  - Requiere simuladores virtuales porque no podés permitir que un agente falle millones de veces en el mundo físico real (ej: chocar autos o fundir maquinaria).
  - **Excepción de vanguardia:** Es la base del entrenamiento de razonamiento de los grandes modelos de lenguaje modernos (LLMs como ChatGPT con *RLHF* y DeepSeek) y de la robótica autónoma avanzada.
