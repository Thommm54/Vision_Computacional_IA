# 🛣️ Visión Computacional para la Detección de Baches en Entornos Urbanos

<div align="center">

**Computer vision system for automated pothole detection in urban environments**

*Universidad Andina del Cusco*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.60-00FFFF?style=flat)
![Gradio](https://img.shields.io/badge/Demo-Gradio-FF7C00?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 📌 Descripción

Sistema de visión por computadora que automatiza la **detección y localización de baches** en imágenes viales urbanas usando el modelo de Deep Learning **YOLO11n**. Aborda un problema crítico para la gestión municipal en zonas como la Av. de la Cultura, Santiago y Wanchaq en Cusco, donde la degradación vial genera accidentes y daños vehiculares evitables.

> El enfoque tradicional depende de cuadrillas humanas de inspección. Este sistema propone automatizar ese proceso con visión artificial.

---

## 🔁 Pipeline del Proyecto

```
┌─────────────────────────────────────────┐
│  Etapa 1 — Random Forest + HOG          │
│  Clasificación binaria de parches       │
│  Entrada: parche recortado 64x64px      │
│  Salida: ¿Hay bache? → Sí / No          │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Etapa 2 — YOLO11n  ✅ IMPLEMENTADA     │
│  Detección + localización simultánea   │
│  Entrada: imagen completa 640x640px     │
│  Salida: bounding boxes sobre imagen   │
│  F1: 61.8% · mAP@50: 59.0% · 7.9ms/img│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Etapa 3 — RT-DETR  🔜 PRÓXIMA ETAPA   │
│  Arquitectura Transformer               │
│  Atención global sobre imagen completa │
│  Detección en tiempo real               │
└─────────────────────────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
Vision_Computacional_IA/
│
├── demo/
│   ├── app.py              # Interfaz Gradio para inferencia interactiva
│   └── screen/             # Captura de pantalla del demo
│
├── Entrenamiento_yolo11.pt # Pesos del modelo YOLO11n entrenado
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Métricas del Modelo

> Evaluadas sobre conjunto de validación · 5% del dataset · ~200 imágenes · Tesla T4 GPU

| Métrica | Valor | Interpretación |
|---|---|---|
| Precisión | 59.1% | 6 de cada 10 detecciones son baches reales |
| Recall | 64.7% | Detecta el 64.7% de los baches existentes |
| **F1-Score** | **61.8%** | Balance armónico — métrica principal |
| mAP@50 | 59.0% | Precisión media con IoU ≥ 50% |
| mAP@50-95 | 34.3% | Precisión media con umbrales estrictos |
| Velocidad | ~7.9 ms/img | Viable para aplicaciones semi-tiempo real |

> ⚠️ El Recall supera a la Precisión de forma intencional. El umbral de confianza se fijó en **0.4** para priorizar no omitir baches reales, aun a costa de algunos falsos positivos controlables.

---

## 🤖 Especificaciones del Modelo

| Parámetro | Valor |
|---|---|
| Arquitectura | YOLO11n (fused) |
| Framework | PyTorch 2.11.0 + CUDA 12.8 |
| Librería | Ultralytics 8.4.60 |
| Capas totales | 101 |
| Parámetros | 2,582,347 (~2.6M) |
| Tamaño | 5.5 MB |
| Épocas | 50 |
| Batch size | 16 |
| Image size | 640 × 640 px |
| Hardware | Tesla T4 GPU — Google Colab |
| Tiempo de entrenamiento | ~3.1 minutos |

---

## 🖥️ Demo Interactivo

Interfaz web construida con **Gradio** que permite:

- 📤 Cargar cualquier imagen vial
- 🎚️ Ajustar el umbral de confianza (0.1 — 0.9)
- 🔍 Visualizar bounding boxes sobre la imagen en tiempo real
- 📈 Consultar métricas del modelo en el pie de página

**Resultado en imagen de prueba:**
- Baches detectados: **4**
- Confianza máxima: **86.0%**
- Confianza promedio: **64.4%**
- Tiempo de inferencia: **~7.9 ms**

### ▶️ Ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/Vision_Computacional_IA.git
cd Vision_Computacional_IA

# 2. Instalar dependencias
pip install ultralytics gradio opencv-python matplotlib

# 3. Ejecutar el demo
python demo/app.py
```

La interfaz estará disponible en `http://127.0.0.1:7860`

---

## 📦 Dataset

**Pothole Detection Dataset: YOLOv11 Optimized** — Kaggle

| Split | Total | Usado (5%) |
|---|---|---|
| Entrenamiento | ~3,152 | ~158 |
| Validación | ~394 | 19 |
| Prueba | ~394 | ~20 |
| **Total** | **3,940** | **~200** |

- Formato de anotaciones: **YOLO** (coordenadas normalizadas de bounding boxes)
- Clase única: `pothole`
- Semilla de reproducibilidad: `random.seed(42)`

---

## ⚙️ Requisitos

```
Python        >= 3.10
torch         == 2.11.0
ultralytics   == 8.4.60
gradio
opencv-python
matplotlib
```

---

## ⚖️ Consideraciones Éticas

| Sesgo identificado | Descripción | Mitigación |
|---|---|---|
| Climático | Dataset mayormente diurno y pavimento seco | Augmentación con variación de brillo planificada |
| Geográfico | No representa asfalto cusqueño específicamente | Recolección local de imágenes en Etapa 3 |
| Tamaño de bache | Favorece baches medianos sobre microfisuras | Ampliar dataset con baches pequeños |

> Un **falso negativo** (bache no detectado) representa mayor riesgo que un falso positivo, justificando el umbral conservador de 0.4 que prioriza el Recall.

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver archivo [`LICENSE`](./LICENSE) para más detalles.

---

<div align="center">

Universidad Andina del Cusco · Visión por Computadora e Inteligencia Artificial · 2026

</div>
