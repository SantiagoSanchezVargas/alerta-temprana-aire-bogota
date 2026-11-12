# Alerta Temprana de Calidad del Aire – Bogotá

Sistema predictivo basado en redes neuronales **LSTM** para estimar las concentraciones de **PM10** en Bogotá y Sabana Centro con un horizonte de **24 horas**. El proyecto incluye un pipeline de ingesta de datos del **IBOCA**, entrenamiento del modelo, generación de un archivo `predicciones.json` y un **dashboard web** para visualizar los resultados.

> **Aviso:** Este es un proyecto académico/experimental. No sustituye los sistemas oficiales de alerta temprana de la Secretaría Distrital de Ambiente (SATAB) ni las decisiones de las autoridades ambientales.

---

## Tabla de contenido

- [Contexto](#contexto)
- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Modelo LSTM](#modelo-lstm)
- [Fuentes de datos](#fuentes-de-datos)
- [Dashboard](#dashboard)
- [Descargo de responsabilidad](#descargo-de-responsabilidad)
- [Licencia](#licencia)
- [Autores](#autor)

---

## Contexto

Bogotá cuenta con la **Red de Monitoreo de Calidad del Aire (RMCAB)** y el **Índice Bogotano de Calidad del Aire y Riesgo en Salud (IBOCA)** como herramientas oficiales para la vigilancia de contaminantes criterio como PM10 y PM2.5.

Este proyecto busca complementar esos esfuerzos con un modelo predictivo de corto plazo que permita anticipar episodios de alta contaminación y apoyar la toma de decisiones preventivas. El sistema se alimenta de los reportes horarios del IBOCA en formato Excel y genera predicciones para las estaciones disponibles, como **Bolivia** y **Ciudad Bolívar**, entre otras estaciones de la RMCAB.

---

## Características

- **Pipeline de ingesta** que lee y limpia el reporte del IBOCA en formato Excel (`ingesta.py`).
- **Preprocesamiento** de series temporales: imputación, transformación logarítmica y codificación cíclica de hora y día de la semana.
- **Modelo LSTM compacto** con salida multisalida para predecir 24 horas consecutivas (`modelo.py`).
- **Entrenamiento y evaluación** con división cronológica 80/20, callbacks de *Early Stopping* y *ReduceLROnPlateau* (`entrenar_evaluar.py`).
- **Generación de predicciones** en formato JSON con intervalos de incertidumbre percentiles 10 y 90 (`generar_json.py`).
- **Dashboard web estático** (`index.html`) que consume `predicciones.json` y muestra el estado del sistema y las predicciones por estación.
- **Modelo preentrenado** incluido en el repositorio: `modelo_lstm_calidad_aire.h5`.

---

## Estructura del proyecto

```text
alerta-temprana-aire-bogota/
├── ingesta.py                  # Carga y limpieza del reporte IBOCA (Excel)
├── procesamiento.py            # Requerido: preprocesamiento y ventanas deslizantes
├── modelo.py                   # Definición y compilación del modelo LSTM
├── entrenar_evaluar.py         # Entrenamiento del modelo y guardado en .h5
├── generar_json.py             # Generación de predicciones.json a partir del modelo entrenado
├── index.html                  # Dashboard web para visualizar predicciones
├── modelo_lstm_calidad_aire.h5 # Modelo LSTM preentrenado
├── .gitignore
└── LICENSE
```

> **Nota:** Los scripts `entrenar_evaluar.py` y `generar_json.py` importan funciones desde `procesamiento` (`preparar_dataset`, `crear_ventanas_deslizantes`, `LOOKBACK`, `HORIZON`). Asegúrate de incluir ese módulo en el repositorio para que el pipeline funcione correctamente.

---

## Requisitos

- Python 3.9 o superior.
- Librerías de Python:

```bash
pip install pandas numpy tensorflow keras openpyxl lxml
```

> `openpyxl` y `lxml` son necesarios para que `pandas.read_html` pueda leer el archivo Excel del IBOCA.

- Navegador web moderno para visualizar el dashboard.

---

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/SantiagoSanchezVargas/alerta-temprana-aire-bogota.git
cd alerta-temprana-aire-bogota
```

2. Crea un entorno virtual e instala las dependencias:

```bash
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install pandas numpy tensorflow keras openpyxl lxml
```

3. Coloca el archivo de datos del IBOCA (`reporteIboca (1).xls`) en la raíz del proyecto. Puedes obtenerlo desde los portales de datos abiertos de la Secretaría Distrital de Ambiente o desde la RMCAB.

---

## Uso

### 1. Entrenar el modelo (opcional)

Si deseas reentrenar el modelo con nuevos datos:

```bash
python entrenar_evaluar.py
```

Esto cargará el Excel del IBOCA, preprocesará los datos, entrenará la LSTM y guardará el modelo como `modelo_lstm_calidad_aire.h5`.

### 2. Generar las predicciones

Con el modelo entrenado o el incluido en el repositorio, ejecuta:

```bash
python generar_json.py
```

Se creará el archivo `predicciones.json` con las predicciones de las próximas 24 horas para cada estación.

### 3. Visualizar el dashboard

Abre el archivo `index.html` en tu navegador. El dashboard leerá `predicciones.json` y mostrará el estado del sistema y las predicciones por estación.

Si prefieres usar un servidor local:

```bash
python -m http.server 8000
```

Luego visita:

```text
http://localhost:8000
```

---

## Modelo LSTM

El modelo se define en `modelo.py` y tiene las siguientes características:

- **Entrada:** ventanas deslizantes de 48 horas (`LOOKBACK = 48`) con 6 características:
  - `pm10_log`: transformación logarítmica de PM10.
  - `pm10_imputado`: valor imputado.
  - `hora_sin` y `hora_cos`: codificación cíclica de la hora.
  - `dow_sin` y `dow_cos`: codificación cíclica del día de la semana.
- **Arquitectura:** una capa LSTM de 64 unidades + Dropout de 0.2 + capa densa de salida con 24 neuronas, una por cada hora futura.
- **Función de pérdida:** Huber por defecto o *Pinball* para penalizar más la subestimación de picos.
- **Optimizador:** Adam con `clipnorm=1.0`.
- **Métrica:** MAE.
- **Horizonte de predicción:** 24 horas (`HORIZON = 24`).

---

## Fuentes de datos

- **IBOCA (Índice Bogotano de Calidad del Aire):** reporte horario en formato Excel que contiene las concentraciones de PM10 y PM2.5 de las estaciones de la RMCAB. El archivo debe llamarse `reporteIboca (1).xls` y ubicarse en la raíz del proyecto.
- **RMCAB (Red de Monitoreo de Calidad del Aire de Bogotá):** red oficial de estaciones fijas de la Secretaría Distrital de Ambiente. Los datos utilizados en este proyecto provienen de estaciones como **Bolivia** y **Ciudad Bolívar**.

Para más información sobre los datos abiertos, consulta el portal de la RMCAB:

```text
http://rmcab.ambientebogota.gov.co
```

---

## Dashboard

El archivo `index.html` es un dashboard estático que:

- Muestra el estado del sistema: “Operando con normalidad”.
- Lista las estaciones disponibles.
- Presenta las predicciones de PM10 para las próximas 24 horas, incluyendo los intervalos de incertidumbre p10 y p90.
- Está diseñado para ser desplegado fácilmente en GitHub Pages o cualquier servidor estático.

---

## Descargo de responsabilidad

Este proyecto tiene fines **académicos y de investigación**. Las predicciones generadas por el modelo LSTM no han sido validadas por las autoridades ambientales y **no deben utilizarse como única fuente para la toma de decisiones** relacionadas con alertas de calidad del aire.

Para información oficial, consulta el **SATAB** (Sistema de Alertas Tempranas Ambientales de Bogotá) y el **IBOCA**.

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Autores

- **Daniela Alejandra Álvarez Velandia**  
- **Juan Sebastián Bustos Ruiz**  
- **Cristian Camilo Naranjo Muñoz**
- **Santiago Andrés Sánchez Vargas**  
GitHub: [@SantiagoSanchezVargas](https://github.com/SantiagoSanchezVargas)

---

¿Tienes dudas o sugerencias? ¡Abre un *issue* o envía un *pull request*! 🚀
