import gradio as gr
import os
from utils.tools import Camera
from utils.stereo import StereoAligner

capL, capR = Camera(5), Camera(7)
aligner = StereoAligner(chessboard_size=(9, 6))

os.makedirs('data/calibrate/left', exist_ok=True)
os.makedirs('data/calibrate/right', exist_ok=True)

def streaming(threadhold):
    while True:
        frameL, frameR = capL.read(), capR.read()
        result = aligner.fit(frameL, frameR)
        yield cv2.cvtColor(result.plot(threadhold=threadhold), cv2.COLOR_BGR2RGB), result

with gr.Blocks() as demo:
    gr.Markdown("# Stereo Alignment Streaming (Gradio)")
    threadhold = gr.Number(value=2.0, label="Threadhold", precision=1, step=0.1)
    image = gr.Image(label="Stereo Merged")
    state = gr.State(None)
    demo.load(fn=streaming, inputs=[threadhold], outputs=[image, state])

demo.launch(share=False, inbrowser=True)