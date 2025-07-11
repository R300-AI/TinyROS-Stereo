import gradio as gr
import os, cv2, time, glob
from utils.tools import Camera
from utils.stereo import StereoAligner

capL, capR = Camera(5), Camera(7)

os.makedirs('data/calibrate/left', exist_ok=True)
os.makedirs('data/calibrate/right', exist_ok=True)

def streaming():
    while True:
        frameL, frameR = capL.read(), capR.read()
        aligner = StereoAligner(align_reference=side.value, chessboard_size=(9, 6))
        result = aligner.fit(frameL, frameR)
        yield cv2.cvtColor(result.plot(threadhold=threadhold.value, scale=1), cv2.COLOR_BGR2RGB), result

def list_calibrate_files(side):
    return glob.glob(f"data/calibrate/left/*.jpg")

def snapshot(side, result):
    file_name = f"{ time.strftime('%Y%m%d_%H%M%S')}.jpg"
    left_path = os.path.join('data/calibrate/left', file_name)
    right_path = os.path.join('data/calibrate/right', file_name)
    cv2.imwrite(left_path, result.left.img)
    cv2.imwrite(right_path, result.right.img)
    return glob.glob(f"data/calibrate/{side}/*.jpg")

def remove(evt: gr.SelectData):
    value = evt.value
    orig_name = value["image"]["orig_name"]
    print(f"Selected sample path: {orig_name}")

with gr.Blocks() as demo:
    gr.Markdown("# Stereo Alignment Streaming (Gradio)")
    state = gr.State(None)
    with gr.Row():
        side = gr.Dropdown(value="left", choices=["left", "right"], label="Select Side")
        threadhold = gr.Number(value=1.5, label="Threadhold", precision=1, step=0.1)
        snapshot_btn = gr.Button("Take Snapshot")
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(label="Stereo Merged")
        with gr.Column(scale=1):
            gallery = gr.Gallery(label="Snapshot", columns=3, height="360px", allow_preview=False)

    snapshot_btn.click(fn=snapshot, inputs=[side, state], outputs=gallery)
    gallery.select(fn=remove, inputs=None, outputs=gallery)

    demo.load(fn=streaming, inputs=None, outputs=[image, state])
    demo.load(fn=list_calibrate_files, inputs=None, outputs=gallery)
    
demo.launch(share=False, inbrowser=True)