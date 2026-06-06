# Visión Computacional para la Detección Automatizada de Baches y Rejillas en Entornos Urbanos
Computer vision system for automated pothole and grate detection in urban environments — Universidad Andina del Cusco

Sistema de visión por computadora que automatiza la detección y localización de baches en imágenes viales urbanas mediante el modelo de Deep Learning YOLO11n. El proyecto forma parte de un pipeline progresivo de tres etapas que evoluciona desde clasificación clásica hasta detección en tiempo real con arquitecturas Transformer.
El problema abordado es crítico para la gestión urbana: el deterioro vial en zonas como la Av. de la Cultura, distrito de Santiago y Wanchaq en Cusco genera daños vehiculares, accidentes y costos de mantenimiento evitables. Este sistema propone automatizar la inspección vial que hoy depende exclusivamente de cuadrillas humanas.

Etapa 1 — Random Forest + HOG
          Clasificación binaria de parches recortados
          ¿Hay bache en este parche? → Sí / No

          ↓

Etapa 2 — YOLO11n (Deep Learning)   ← IMPLEMENTADA
          Detección y localización simultánea
          Bounding boxes sobre imagen completa
          F1: 61.8% · mAP@50: 59.0% · Inferencia: 7.9ms

          ↓

Etapa 3 — RT-DETR (Transformer)     ← PRÓXIMA ETAPA
          Atención global sobre imagen completa
          Detección en tiempo real con arquitectura Transformer


          
📁 Estructura del Repositorio

Vision_Computacional_IA/
│
├── demo/
│   ├── app.py                  # Interfaz Gradio para inferencia interactiva
│   └── screen/                 # Captura de pantalla del demo en funcionamiento
│
├── Entrenamiento_yolo11.pt     # Pesos del modelo YOLO11n entrenado
├── .gitignore
├── LICENSE
└── README.md
