import requests
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
from main_simple_lib import *
from PIL import Image
import io
import os
import spacy
from spacy import displacy
from io import BytesIO
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import base64

# ========================================
#             Model Initialization
# ========================================


# ========================================
#             Gradio Setting
# ========================================

def gradio_reset():
    return None, gr.update(placeholder='Please build index firstt', interactive=True)

# def gradio_answer(user_message, chatbot, chat_state, input_file):
#     url = 'http://ir.himygpt.cn:8508/indices/build'
#     chatbot = chatbot + [[user_message, None]]
#     chatbot[-1][1] = 'input_file'
#     return '', chatbot, chat_state

# def upload_file(input_files, chatbot):
#
#     chatbot = chatbot + [((input_files.name,), None)]
#     # url = 'http://ir.himygpt.cn:8508/indices/build'
#     # user_id = '1'
#     # basename = os.path.basename(input_files.name)
#     # files = [('files', (basename, input_files.read()))]
#     # response = requests.post(url, params={'user_id': '1', 'index_mode': 'regex'}, files=files)
#     # chatbot[-1][1] = response.text
#     return chatbot

# def show_image(image, chatbot):
#     chatbot = chatbot + [((image.name,), None)]
#     im = load_image(image.name)
#     query = 'Is the word in the logo "A"? Please answer yes or no.'
#
#     show_single_image(im)
#     code = get_code(query)
def ex_c(image1, codesy1, html1):
    image1 = transforms.ToTensor()(image1)

    result = execute_code(codesy1, image1)
    im = result.cropped_image
    image = im.detach().cpu()
    if image.dtype == torch.float32:
        image = image.clamp(0, 1)
    image = image.squeeze(0).permute(1, 2, 0)
    image = torch.tensor(image)
    image_np = image.numpy()
    pil_image = Image.fromarray(np.uint8(image_np))
    image_bytes_io = BytesIO()
    pil_image.save(image_bytes_io, format='RGB')
    base64_image = base64.b64encode(image_bytes_io.getvalue()).decode('utf-8')
    # 创建HTML代码，嵌入Base64编码的图像字符串
    html_code = f'<img src="data:image/png;base64,{base64_image}" alt="image">'
    html1 = html1 + html_code

    return html1


def chat1(user_message, chatbot, chat_state):
    chatbot = chatbot + [[user_message, None]]
    code, sy = get_code(user_message)
    # result = response.json()
    chatbot[-1][1] = str(sy)
    return '', chatbot, chat_state

def chat2(user_message, html1, chat_state):
    # chatbot = chatbot + [[user_message, None]]
    codesy1 = get_code(user_message)
    code, sy = codesy1
    a = aaa(code)
    html1 = a
    # result = response.json()
    return '', html1, codesy1


def aaa(code):
    code1 = code.replace("(image, my_fig, time_wait_between_lines, syntax)", "(image)")
    lexer = get_lexer_by_name("python", stripall=True)
    formatter = HtmlFormatter(style="monokai", linenos=True, full=True)

    html_code = highlight(code1, lexer, formatter)
    return html_code

title = """<h1 align="center">Demo of vipergpt</h1>"""
description = """<h3>This is the demo for !</h3>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil")
            text_input = gr.Textbox(label='User', placeholder='Please iuput image first', interactive=True)
            upload_button1 = gr.Button(value="执行代码", interactive=True, variant="primary")
            html = gr.HTML(label='vipergpt')
            codesy = gr.State()
        # with gr.Column():
        #     # text_input1 = gr.Textbox(label='用户名', placeholder="请输入用户名")
        #     # text_input2 = gr.Textbox(label='密码', placeholder="请输入密码", password=True)
        #     # text_input3 = gr.Textbox(label='文件路径', placeholder="请输入文件路径")
        #     image = gr.Image(type="pil")
        #     upload_button1 = gr.Button(value="上传图片", interactive=True, variant="primary")
        #     BoxThreshold0 = gr.Slider(
        #         minimum=0.0,
        #         maximum=1.0,
        #         value=0.25,
        #         step=0.001,
        #         interactive=True,
        #         label="Threshold0",
        #     )
        #     BoxThreshold1 = gr.Slider(
        #         minimum=0.0,
        #         maximum=1.0,
        #         value=0.30,
        #         step=0.001,
        #         interactive=True,
        #         label="Threshold1",
        #     )
        #     clear = gr.Button("Restart")
        #
        # with gr.Column(scale=2):
        #     # image = gr.State()
        #     chat_state = gr.State()
        #     img_list = gr.State()
        #     chatbot = gr.Chatbot(label='vipergpt')
        #     html = gr.HTML(label='vipergpt')
        #     text_input = gr.Textbox(label='User', placeholder='Please iuput image first', interactive=True)

        # upload_button1.click(upload_file, [image, chatbot], [chatbot])
        # text_input.submit(chat1, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state])

        text_input.submit(chat2, [text_input, html], [text_input, html, codesy])
        upload_button1.click(ex_c, [image, codesy, html], [html])

        # clear.click(gradio_reset, outputs=[chatbot, text_input])

demo.launch(share=True, server_port=7590)