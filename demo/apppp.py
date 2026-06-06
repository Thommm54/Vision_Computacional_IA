"""
Demo funcional — Detección de Baches con YOLO11n
Fase 2 — Deep Learning
Ejecutar: python app.py
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# ── Cargar modelo ──────────────────────────────────────────────
MODEL_PATH = "best.pt"   # <── coloca best.pt en la misma carpeta que app.py

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró '{MODEL_PATH}'.\n"
        "Descárgalo de Colab con:\n"
        "  from google.colab import files\n"
        "  files.download('/content/runs/detect/train_subset/weights/best.pt')\n"
        "y colócalo en la misma carpeta que app.py."
    )

model = YOLO(MODEL_PATH)
print(f"✅ Modelo cargado: {MODEL_PATH}")


# ── Función de inferencia ──────────────────────────────────────
def detectar_baches(imagen: np.ndarray, confianza: float):
    """
    Recibe imagen (numpy RGB), devuelve imagen anotada + reporte de texto.
    """
    if imagen is None:
        return None, "⚠️ No se cargó ninguna imagen."

    # Ultralytics espera BGR
    img_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=img_bgr,
        conf=confianza,
        verbose=False,
    )[0]

    boxes   = results.boxes
    n_det   = len(boxes) if boxes is not None else 0

    # ── Dibujar bounding boxes ────────────────────────────────
    img_out = img_bgr.copy()

    if n_det > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])

            # Color: rojo más intenso cuanto mayor la confianza
            intensity = int(55 + 200 * conf_val)
            color = (0, 0, intensity)

            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 3)

            label = f"Bache {conf_val:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img_out, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                img_out, label,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

    img_rgb_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    # ── Reporte de texto ──────────────────────────────────────
    if n_det == 0:
        reporte = "✅ No se detectaron baches con la confianza establecida."
    else:
        confs = [float(b.conf[0]) for b in boxes]
        lineas = [
            f"🚧 Baches detectados: **{n_det}**",
            f"📊 Confianza promedio: **{np.mean(confs):.1%}**",
            f"📈 Confianza máxima:   **{max(confs):.1%}**",
            f"📉 Confianza mínima:   **{min(confs):.1%}**",
            "",
            "**Detalle por detección:**",
        ]
        for i, c in enumerate(confs, 1):
            lineas.append(f"  • Bache #{i}: {c:.1%}")
        reporte = "\n".join(lineas)

    return img_rgb_out, reporte


# ── Interfaz Gradio ────────────────────────────────────────────
with gr.Blocks(
    title="Detección de Baches — YOLO11n",
    theme=gr.themes.Base(
        primary_hue="red",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
    #titulo { text-align: center; margin-bottom: 0.5rem; }
    #subtitulo { text-align: center; color: #94a3b8; margin-bottom: 1.5rem; font-size: 0.9rem; }
    .gradio-container { max-width: 1100px !important; }
    """,
) as demo:

    gr.Markdown("# 🚧 Detección de Baches", elem_id="titulo")
    gr.Markdown(
        "Modelo **YOLO11n** entrenado sobre *Pothole Detection Dataset* · Fase 2",
        elem_id="subtitulo",
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="📷 Imagen de entrada",
                type="numpy",
                sources=["upload", "webcam"],
            )
            slider_conf = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                label="Umbral de confianza",
                info="0.4 = balance recall/precisión (valor del paper)",
            )
            btn = gr.Button("🔍 Detectar Baches", variant="primary", size="lg")

        with gr.Column(scale=1):
            img_output = gr.Image(label="📍 Detecciones", type="numpy")
            txt_output = gr.Markdown(label="📋 Reporte")

    btn.click(
        fn=detectar_baches,
        inputs=[img_input, slider_conf],
        outputs=[img_output, txt_output],
    )

    gr.Markdown(
        """
        ---
        **Métricas del modelo** (validación, 5% dataset):  
        Precisión `59.1%` · Recall `64.7%` · F1 `61.8%` · mAP@50 `59.0%` · Inferencia `~7.9 ms/img`
        """,
        elem_id="footer",
    )


if __name__ == "__main__":
    demo.launch(share=False)
