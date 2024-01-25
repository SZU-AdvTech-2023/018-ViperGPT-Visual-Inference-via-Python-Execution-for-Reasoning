import torch
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes


# ========================================
#             Model Initialization
# ========================================

from main_simple_lib import *



# ========================================
#             Gradio Setting
# ========================================

def gradio_reset():
    return gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(value=None, interactive=True)

def upload_img(gr_img, BoxThreshold0, BoxThreshold1, BoxThreshold2 , text_input1, text_input2, text_input3):
    if gr_img is None:
        return gr.update(interactive=True, placeholder='Input should not be empty!')
    image = transforms.ToTensor()(gr_img)
    img_out, change_image = multi_vocabulary(image, BoxThreshold0, BoxThreshold1, BoxThreshold2, subject=text_input1, property=text_input2, without=text_input3)
    return img_out


title = """<h1 align="center">Demo of detecting objects with multiple vocabularies</h1>"""
description = """<h3>This is the demo for detecting objects with multiple vocabularies. Upload your images and start detecting!</h3>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            img_in = gr.Image()
            text_input1 = gr.Textbox(label='主体', placeholder="请输入检测主体，如：person")
            BoxThreshold0 = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.001,
                interactive=True,
                label="Box_Threshold",
            )
            text_input2 = gr.Textbox(label='相关属性', placeholder='请输入相关属性，如：helmet， motorcycle')
            BoxThreshold1 = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.30,
                step=0.001,
                interactive=True,
                label="Box_Threshold",
            )
            text_input3 = gr.Textbox(label='不具备属性', placeholder='请输入不具备属性，如：helmet， motorcycle')
            BoxThreshold2 = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.45,
                step=0.001,
                interactive=True,
                label="Box_Threshold",
            )
            upload_button = gr.Button(value="Upload & Start detect", interactive=True, variant="primary")
            clear = gr.Button("Restart")


        with gr.Column():
            img_out = gr.Image()

        upload_button.click(upload_img, inputs=[img_in, BoxThreshold0, BoxThreshold1, BoxThreshold2, text_input1, text_input2, text_input3], outputs=img_out)
        clear.click(gradio_reset, outputs=[img_in, img_out, text_input1, text_input2, text_input3])

demo.launch(share=True, enable_queue=True, server_port=7581)
