import gradio as gr
import os
from utils.tools import Camera
from utils.stereo import StereoAligner

capL, capR = Capture(5), Capture(7)
aligner = StereoAligner(chessboard_size=(9, 6))

os.makedirs('data/calibrate/left', exist_ok=True)
os.makedirs('data/calibrate/right', exist_ok=True)

latest_result = {'result': None, 'threadhold': 2.0}

# get_merged 只回傳 merged_rgb，狀態由 gr.State 自動保存

def get_merged(threadhold):
    frameL, frameR = capL.read(), capR.read()
    result = aligner.fit(frameL, frameR)
    return result.plot(threadhold=threadhold), result

with gr.Blocks() as demo:
    gr.Markdown("# Stereo Alignment Streaming (Gradio)")
    threadhold = gr.Number(value=2.0, label="Threadhold", precision=1, step=0.1)
    image = gr.Image(label="Stereo Merged", live=True)
    state = gr.State(None)
    demo.load(fn=get_merged, inputs=[threadhold], outputs=[image, state])

demo.launch(share=False, inbrowser=True)
