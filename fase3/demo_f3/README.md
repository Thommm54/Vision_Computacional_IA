# Demo — Detección de Baches (Fase 3)

Modelo **RT-DETR-L** (Transformer) entrenado con el dataset completo (3,940 imágenes).  
Mejora significativa respecto a Fase 2: mAP@50 `91.6%` vs `59.0%` anterior.

## Requisitos previos

- Python 3.10+
- El archivo `best.pt` descargado de Colab o Google Drive (ver abajo)
- iRun Webcam instalado en PC y celular (opcional, para stream desde celular)

## 1. Descargar el modelo desde Google Drive

Ejecuta esta celda al final del notebook `RT_DETR_F3.ipynb`:

```python
from google.colab import files
files.download('/content/drive/MyDrive/rtdetr_fase3/train_fase3/weights/best.pt')
```

Coloca `best.pt` en la misma carpeta que `app.py`.

## 2. Instalar dependencias

```
pip install -r requirements.txt
```

## 3. Ejecutar el demo

```
python app.py
```

Abre el navegador en: http://127.0.0.1:7860

## Modos de uso

- **📷 Tiempo Real** → selecciona la fuente de video y presiona Iniciar
  - Cámara Laptop → cámara integrada
  - Celular (iRun Webcam) → requiere iRun Webcam activo en PC y celular
- **🖼️ Subir Foto** → sube una imagen y presiona Detectar Baches

## Leyenda de colores

| Color | Significado |
|-------|-------------|
| 🔴 Rojo | Confianza alta (≥ 75%) |
| 🟡 Amarillo | Confianza media (50–75%) |
| 🟢 Verde | Confianza baja (< 50%) |

## Métricas del modelo

| Métrica | Valor |
|---------|-------|
| Precisión | 87.9% |
| Recall | 85.1% |
| F1-Score | 86.5% |
| mAP@50 | 91.6% |
| mAP@50-95 | 63.3% |

## Estructura de carpetas

```
demo_f3/
├── app.py
├── best.pt        ← descargar de Colab/Drive
├── logo.png
├── logo_1.png
├── requirements.txt
└── README.md
```
