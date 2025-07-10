import gradio as gr
import cv2, os
import numpy as np

left_dir = os.path.join('data', 'calibrate', 'C270', 'Left', '1.jpg')
right_dir = os.path.join('data', 'calibrate', 'C270', 'Right', '1.jpg')
left_img = cv2.imread(left_dir, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_dir, cv2.IMREAD_GRAYSCALE)

def store_point(evt: gr.SelectData):
    return evt.index

def show_coords(coords):
    if coords is None:
        return ""
    return f"({coords[0]}, {coords[1]})"

with gr.Blocks() as demo:
    gr.Markdown("## 點選左右影像上的對應點，計算3D世界座標 (cm)")
    with gr.Row():
        with gr.Column():
            left_coords = gr.State()
            left = gr.Image(value=left_img, label="左影像 (點一下)", interactive=True, type="numpy")
            left_coords_box = gr.Textbox(label="左影像座標", interactive=False)
        with gr.Column():
            right_coords = gr.State()
            right = gr.Image(value=right_img, label="右影像 (點一下)", interactive=True, type="numpy")
            right_coords_box = gr.Textbox(label="右影像座標", interactive=False)
    
    result = gr.Textbox(label="3D世界座標 (cm)")


    left.select(store_point, None, left_coords).then(show_coords, left_coords, left_coords_box)
    right.select(store_point, None, right_coords).then(show_coords, right_coords, right_coords_box)

demo.launch(share=False, inbrowser=True)
