"""
Demo Fase 3 — Detección de Baches con RT-DETR
Ejecutar: py app.py
"""

import gradio as gr
import cv2
import numpy as np
from ultralytics import RTDETR
import os

# ── Cargar modelo ──────────────────────────────────────────────
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró '{MODEL_PATH}'.")

model = RTDETR(MODEL_PATH)
print(f"✅ Modelo RT-DETR cargado: {MODEL_PATH}")

stream_activo = False


# ── Color por confianza ────────────────────────────────────────
def color_por_confianza(conf_val: float):
    if conf_val >= 0.75:
        return (0, 0, 220)      # Rojo   → alta confianza
    elif conf_val >= 0.50:
        return (0, 200, 255)    # Amarillo → media
    else:
        return (0, 200, 0)      # Verde  → baja confianza


# ── Inferencia ─────────────────────────────────────────────────
def inferencia(img_bgr: np.ndarray, confianza: float):
    results = model.predict(
        source=img_bgr,
        conf=confianza,
        verbose=False,
    )[0]

    boxes = results.boxes
    n_det = len(boxes) if boxes is not None else 0
    img_out = img_bgr.copy()

    if n_det > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            color = color_por_confianza(conf_val)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 3)

    confs = [float(b.conf[0]) for b in boxes] if n_det > 0 else []
    return img_out, n_det, confs


# ── Detección foto ─────────────────────────────────────────────
def detectar_foto(imagen: np.ndarray, confianza: float):
    if imagen is None:
        return None, "⚠️ No se cargó ninguna imagen."

    img_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    img_out, n_det, confs = inferencia(img_bgr, confianza)
    img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    if n_det == 0:
        reporte = "✅ No se detectaron baches con la confianza establecida."
    else:
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

    return img_rgb, reporte


# ── Stream tiempo real ─────────────────────────────────────────
def detectar_stream(fuente: str, confianza: float):
    global stream_activo
    stream_activo = True

    indices = {
        "Cámara Laptop": 0,
        "Celular (iRiun Webcam)": 1,        
    }
    cap = cv2.VideoCapture(indices.get(fuente, 0))

    if not cap.isOpened():
        yield np.zeros((480, 640, 3), dtype=np.uint8)
        return

    try:
        while stream_activo:
            ret, frame = cap.read()
            if not ret:
                break

            img_out, n_det, confs = inferencia(frame, confianza)

            # HUD minimalista
            overlay = img_out.copy()
            cv2.rectangle(overlay, (10, 10), (280, 45), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, img_out, 0.5, 0, img_out)
            avg = f"{np.mean(confs):.0%}" if confs else "---"
            cv2.putText(img_out, f"RT-DETR  Baches: {n_det}  Conf: {avg}",
                        (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)

            img_rgb = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
            yield img_rgb

    finally:
        cap.release()
        stream_activo = False


def detener_stream():
    global stream_activo
    stream_activo = False


# ── CSS ────────────────────────────────────────────────────────
css = """
#titulo { text-align: center; margin-bottom: 0.3rem; }
#subtitulo { text-align: center; color: #94a3b8; margin-bottom: 1rem; font-size: 0.9rem; }
#footer { text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 1rem; }
#leyenda { margin-top: 0.8rem; padding: 0.6rem 1rem; background: #1e293b;
           border-radius: 8px; font-size: 0.85rem; }
.gradio-container { max-width: 95% !important; margin: auto !important; }
footer { display: none !important; }
"""

# ── Interfaz ───────────────────────────────────────────────────
with gr.Blocks() as demo:

    # Header
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=100):
            gr.Image(
                value="logo.png",
                show_label=False,
                interactive=False,
                container=False,
                height=80,
            )
        with gr.Column(scale=6):
            gr.Markdown(
                "# 🚧 Visión Computacional — Detección de Baches en Entornos Urbanos",
                elem_id="titulo",
            )
            gr.Markdown(
                "Modelo **RT-DETR-L** (Transformer) · Fase 3",
                elem_id="subtitulo",
            )
        with gr.Column(scale=1, min_width=100):
            gr.Image(
                value="logo_1.png",
                show_label=False,
                interactive=False,
                container=False,
                height=80,
            )

    # Tabs
    with gr.Tabs():

        # Tab 1: Tiempo Real
        with gr.TabItem("📷 Tiempo Real"):
            with gr.Row():
                with gr.Column(scale=1):
                    dropdown_fuente = gr.Dropdown(
                        choices=["Cámara Laptop", "Celular (iRiun Webcam)"],
                        value="Cámara Laptop",
                        label="Fuente de video",
                        info="Selecciona la fuente antes de iniciar",
                    )
                    slider_conf_rt = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="Umbral de confianza",
                        info="0.4 = balance recomendado",
                    )

                    with gr.Row():
                        btn_iniciar = gr.Button("▶ Iniciar", variant="primary")
                        btn_detener = gr.Button("⏹ Detener", variant="stop")

                    # Leyenda fuera del video
                    gr.Markdown(
                        """
                        **Leyenda de detección:**

                        🔴 &nbsp;**Rojo** — Confianza alta ( ≥ 75% )  
                        🟡 &nbsp;**Amarillo** — Confianza media ( 50–75% )  
                        🟢 &nbsp;**Verde** — Confianza baja ( < 50% )
                        """,
                        elem_id="leyenda",
                    )

                with gr.Column(scale=2):
                    stream_output = gr.Image(
                        label="📡 Stream en tiempo real",
                        streaming=True,
                        type="numpy",
                    )

            btn_iniciar_event = btn_iniciar.click(
                fn=detectar_stream,
                inputs=[dropdown_fuente, slider_conf_rt],
                outputs=stream_output,
            )
            btn_detener.click(
                fn=detener_stream,
                inputs=[],
                outputs=[],
                cancels=[btn_iniciar_event],
            )

        # Tab 2: Subir Foto
        with gr.TabItem("🖼️ Subir Foto"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Imagen de entrada",
                        type="numpy",
                        sources=["upload"],
                    )
                    slider_conf_foto = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="Umbral de confianza",
                    )
                    btn_detectar = gr.Button(
                        "🔍 Detectar Baches", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label="📍 Detecciones",
                        type="numpy",
                    )
                    txt_output = gr.Markdown()

            btn_detectar.click(
                fn=detectar_foto,
                inputs=[img_input, slider_conf_foto],
                outputs=[img_output, txt_output],
            )

    # Footer
    gr.Markdown(
        """
        ---
        **Métricas RT-DETR-L** (validación, dataset completo 3,940 imgs):  
        Precisión `87.9%` · Recall `85.1%` · F1 `86.5%` · mAP@50 `91.6%` · mAP@50-95 `63.3%` · Inferencia `~39ms/img`
        """,
        elem_id="footer",
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        theme=gr.themes.Base(
            primary_hue="red",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("IBM Plex Mono"),
        ),
        css=css,
    )