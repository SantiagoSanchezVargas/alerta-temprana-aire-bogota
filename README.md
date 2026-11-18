# Sistema de Alerta Temprana de Calidad del Aire - Bogotá y Sabana Centro

Sistema predictivo de concentración de material particulado ($PM_{10}$ y $PM_{2.5}$) basado en redes neuronales recurrentes **LSTM (Long Short-Term Memory)**, acompañado de un dashboard web interactivo desacoplado para el monitoreo operativo en tiempo real y el pronóstico a 24 horas.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Flujo de Datos y Arquitectura](#flujo-de-datos-y-arquitectura)
3. [Estructura del Repositorio](#estructura-del-repositorio)
4. [Especificaciones del Modelo y Umbrales](#especificaciones-del-modelo-y-umbrales)
5. [Requisitos e Instalación](#requisitos-e-instalación)
6. [Guía de Ejecución](#guía-de-ejecución)
7. [Licencia](#licencia)

---

## Descripción General

Este proyecto implementa un pipeline completo de aprendizaje profundo y visualización estática sobre registros históricos de la Red de Monitoreo de Calidad del Aire de Bogotá (RMCAB). El sistema está diseñado en dos capas independientes:

* **Capa de Inferencia y Modelado (Python / TensorFlow):** Procesa series temporales multivariadas, entrena una red recurrente LSTM para estimar concentraciones futuras de $PM_{10}$ con bandas de incertidumbre ($P_{10}$ y $P_{90}$) y exporta los resultados a un formato ligero (`JSON`).
* **Capa de Presentación (HTML / Tailwind CSS / Chart.js):** Consume los pronósticos de forma asíncrona sin requerir un servidor backend activo. Presenta una interfaz tipo aplicación meteorológica con simulación de datos en vivo, pestañas de navegación (Hoy / Día Siguiente) y alertas visuales por niveles de riesgo.

---

## Flujo de Datos y Arquitectura

```text
[Datos Crudos RMCAB (.xls)]
           │
           ▼
    ingesta.py  ──────────► Limpieza y estandarización de estaciones
           │
           ▼
  procesamiento.py ───────► Codificación cíclica (sin/cos) + Ventanas (48h)
           │
           ▼
entrenar_evaluar.py ──────► Entrenamiento LSTM (Huber Loss) ──► modelo_lstm_calidad_aire.h5
           │
           ▼
   generar_json.py ───────► Inferencia a 24h (P10, Media, P90) ──► predicciones.json
                                                                          │
           ┌──────────────────────────────────────────────────────────────┘
           ▼
     index.html ──────────► Dashboard Web (Monitoreo en Vivo + Semáforo de Alerta)
