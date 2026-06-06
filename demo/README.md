# Demo — Detección de Baches (Fase 2)

## Requisitos previos
- Python 3.10+
- El archivo `best.pt` descargado de Colab (ver abajo)

## 1. Descargar el modelo desde Colab
Ejecuta esta celda al final del notebook:
```python
from google.colab import files
files.download('/content/runs/detect/train_subset/weights/best.pt')
```
Coloca `best.pt` en la **misma carpeta** que `app.py`.

## 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 3. Ejecutar el demo
```bash
python app.py
```
Abre el navegador en: **http://127.0.0.1:7860**

## Estructura de carpetas
```
demo_baches/
├── app.py
├── best.pt        ← descargar de Colab
├── requirements.txt
└── README.md
```
