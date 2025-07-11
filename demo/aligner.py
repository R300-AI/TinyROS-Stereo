import gradio as gr
import os
from utils.tools import Camera
from utils.stereo import StereoAligner

capL, capR = Camera(5), Camera(7)
aligner = StereoAligner(chessboard_size=(9, 6))

os.makedirs('data/calibrate/left', exist_ok=True)
os.makedirs('data/calibrate/right', exist_ok=True)

def streaming(threadhold, state):
    frameL, frameR = capL.read(), capR.read()
    result = aligner.fit(frameL, frameR)
    img = result.plot(threadhold=threadhold)
    return gr.update(value=img), result

with gr.Blocks() as demo:
    gr.Markdown("# Stereo Alignment Streaming (Gradio)")
    threadhold = gr.Number(value=2.0, label="Threadhold", precision=1, step=0.1)
    image = gr.Image(label="Stereo Merged")
    state = gr.State(None)
    timer = gr.Timer(interval=0.1, repeat=True)
    timer.tick(fn=streaming, inputs=[threadhold, state], outputs=[image, state])

demo.launch(share=False, inbrowser=True)