# 🛣️ Visión Computacional para la Detección de Baches en Entornos Urbanos

<div align="center">

**Computer vision system for automated pothole detection in urban environments**

*Universidad Andina del Cusco*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.62-00FFFF?style=flat)
![Gradio](https://img.shields.io/badge/Demo-Gradio-FF7C00?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

## 📌 Descripción

Sistema de visión por computadora que automatiza la **detección y localización de baches** en imágenes viales urbanas. El proyecto evoluciona en tres fases, partiendo de clasificación con Random Forest hasta detección en tiempo real con el modelo Transformer **RT-DETR-L**, logrando un mAP@50 del **91.6%**. Aborda un problema crítico para la gestión municipal en zonas como la Av. de la Cultura, Santiago y Wanchaq en Cusco, donde la degradación vial genera accidentes y daños vehiculares evitables.

> El enfoque tradicional depende de cuadrillas humanas de inspección. Este sistema propone automatizar ese proceso con visión artificial.

---

## 🔁 Pipeline del Proyecto

```
┌─────────────────────────────────────────────────────────┐
│  Fase 1 — Random Forest + HOG  ✅ IMPLEMENTADA          │
│  Clasificación binaria de parches                       │
│  Entrada: parche recortado 64×64 px                     │
│  Salida: ¿Hay bache? → Sí / No                          │
│  Dataset: 100 imágenes de entrenamiento                 │
│  Accuracy: 82.6% · F1: 82.4%                           │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Fase 2 — YOLO11n  ✅ IMPLEMENTADA                      │
│  Detección + localización simultánea                   │
│  Entrada: imagen completa 640×640 px                    │
│  Salida: bounding boxes sobre imagen                   │
│  Dataset: 5% (~197 imágenes)                            │
│  F1: 61.8% · mAP@50: 59.0% · ~7.9 ms/img              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Fase 3 — RT-DETR-L  ✅ IMPLEMENTADA                    │
│  Arquitectura Transformer                               │
│  Entrada: imagen completa 640×640 px                    │
│  Salida: bounding boxes + detección en tiempo real     │
│  Dataset: 100% (3,940 imágenes)                         │
│  F1: 86.5% · mAP@50: 91.6% · ~39 ms/img               │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
Vision_Computacional_IA/
│
├── fase1/
│   ├── notebook_fase1.ipynb    # Clasificación con Random Forest + HOG
│   └── ...
│
├── fase2/
│   ├── notebook_fase2.ipynb    # Detección con YOLO11n (5% dataset)
│   └── demo/
│       ├── app.py              # Demo Gradio — subir foto
│       ├── best.pt             # Pesos YOLO11n entrenados
│       ├── logo.png
│       ├── logo_1.png
│       ├── requirements.txt
│       └── README.md
│
├── fase3/
│   ├── RT_DETR_F3.ipynb        # Entrenamiento RT-DETR-L (dataset completo)
│   └── demo_f3/
│       ├── app.py              # Demo Gradio — tiempo real + foto
│       ├── best.pt             # Pesos RT-DETR-L entrenados
│       ├── logo.png
│       ├── logo_1.png
│       ├── requirements.txt
│       └── README.md
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Comparativa de Métricas por Fase

> Evolución del sistema a lo largo de las tres fases del proyecto

| Métrica | Fase 1 — Random Forest | Fase 2 — YOLO11n | Fase 3 — RT-DETR-L |
|---|---|---|---|
| Tarea | Clasificación binaria | Detección | Detección |
| Dataset | 100 imágenes | 5% (~197 imgs) | 100% (3,940 imgs) |
| Precisión | 83.0% | 59.1% | **87.9%** |
| Recall | 83.0% | 64.7% | **85.1%** |
| F1-Score | 82.4% | 61.8% | **86.5%** |
| mAP@50 | N/A | 59.0% | **91.6%** |
| mAP@50-95 | N/A | 34.3% | **63.3%** |
| Velocidad | — | ~7.9 ms/img | ~39 ms/img |

> ⚠️ La caída de Fase 1 a Fase 2 se explica por el cambio de tarea (clasificación → detección, más difícil) y el uso de solo el 5% del dataset. La recuperación en Fase 3 demuestra que con arquitectura Transformer y el dataset completo se supera ampliamente el baseline.

---

## 🤖 Especificaciones de los Modelos

### Fase 1 — Random Forest

| Parámetro | Valor |
|---|---|
| Algoritmo | Random Forest + HOG features |
| Tipo de tarea | Clasificación binaria (Asfalto / Bache) |
| Tamaño de parche | 64 × 64 px |
| Dataset de entrenamiento | 100 imágenes |
| Accuracy | 82.6% |

### Fase 2 — YOLO11n

| Parámetro | Valor |
|---|---|
| Arquitectura | YOLO11n (fused) |
| Framework | PyTorch 2.11.0 + CUDA 12.8 |
| Librería | Ultralytics 8.4.60 |
| Parámetros | ~2.6M |
| Tamaño | 5.5 MB |
| Épocas | 50 · Batch 16 · Imagen 640×640 |
| Hardware | Tesla T4 — Google Colab |

### Fase 3 — RT-DETR-L

| Parámetro | Valor |
|---|---|
| Arquitectura | RT-DETR-L (Transformer) |
| Framework | PyTorch 2.11.0 + CUDA 12.8 |
| Librería | Ultralytics 8.4.62 |
| Capas | 310 · Parámetros: ~32M |
| Épocas | 20 · Batch 4 · Imagen 640×640 |
| Hardware | Tesla T4 (16 GB) — Google Colab |
| Tiempo de entrenamiento | ~4.7 horas |

---

## 🖥️ Demo Interactivo — Fase 3

La demo final fue construida con **Gradio** e incorpora detección en **tiempo real** con cámara.

**Funcionalidades:**
- 📷 Stream en tiempo real desde cámara laptop o celular (iRun Webcam)
- 🖼️ Subir foto para inferencia estática
- 🎚️ Ajustar umbral de confianza (0.1 — 0.9)
- 🔴🟡🟢 Colores de bounding box según nivel de confianza

**Leyenda de colores:**

| Color | Confianza |
|---|---|
| 🔴 Rojo | Alta (≥ 75%) |
| 🟡 Amarillo | Media (50–75%) |
| 🟢 Verde | Baja (< 50%) |

### ▶️ Ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/Thommm54/Vision_Computacional_IA.git
cd Vision_Computacional_IA/fase3/demo_f3

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el demo
python app.py
```

La interfaz estará disponible en `http://127.0.0.1:7860`

> Para acceder desde el celular en la misma red WiFi, usa la IP local de tu PC: `http://192.168.X.X:7860`

---

## 📦 Dataset

**Pothole Detection Dataset: YOLOv11 Optimized** — Kaggle

| Split | Total | Usado Fase 2 (5%) | Usado Fase 3 (100%) |
|---|---|---|---|
| Entrenamiento | ~3,152 | ~158 | ~3,152 |
| Validación | ~394 | ~20 | ~394 |
| Prueba | ~394 | ~20 | ~394 |
| **Total** | **3,940** | **~197** | **3,940** |

- Formato: **YOLO** (coordenadas normalizadas de bounding boxes)
- Clase única: `pothole`
- Semilla de reproducibilidad: `random.seed(42)`

---

## ⚙️ Requisitos

```
Python        >= 3.10
torch         == 2.11.0
ultralytics   == 8.4.62
gradio        >= 6.0
opencv-python
matplotlib
```

---

## ⚖️ Consideraciones Éticas

| Sesgo identificado | Descripción | Mitigación |
|---|---|---|
| Climático | Dataset mayormente diurno y pavimento seco | Augmentación con variación de brillo |
| Geográfico | No representa asfalto cusqueño específicamente | Recolección local de imágenes planificada |
| Tamaño de bache | Favorece baches medianos sobre microfisuras | Ampliar dataset con baches pequeños |

> Un **falso negativo** (bache no detectado) representa mayor riesgo que un falso positivo, justificando el umbral conservador de 0.4 que prioriza el Recall.

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver archivo [`LICENSE`](./LICENSE) para más detalles.

---

<div align="center">

Universidad Andina del Cusco · Visión por Computadora e Inteligencia Artificial · 2026

</div>