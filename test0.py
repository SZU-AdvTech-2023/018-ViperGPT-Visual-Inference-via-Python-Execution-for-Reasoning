from main_simple_lib import *
#
# im = load_image("/public26_data/lsx/test/测试数据/570081农林路与侨香路交叉口/clipbord_1687918992828.png")
# image_patch = ImagePatch(im)
# result = image_patch.simple_query('What words are in the picture?')
# print(result)
# # people = image_patch.find('people')
# # print(people)
# # key = image_patch.get_keyword()
# # print(key)
# print(9)
# pip install accelerate
# import requests
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, \
#     Blip2ForConditionalGeneration
# print(9)
# processor = Blip2Processor.from_pretrained("/public26_data/lsx/vipergpt/pretrained_models/blip2-flan-t5-xl")
# dev = "cuda:1" if torch.cuda.is_available() else "cpu"
# max_memory = {1: torch.cuda.mem_get_info(dev)[0]}
# model = Blip2ForConditionalGeneration.from_pretrained(
#                     f"/public26_data/lsx/vipergpt/pretrained_models/blip2-flan-t5-xl", load_in_8bit=True,
#                     torch_dtype=torch.float16 if True else "auto",
#                     device_map="sequential", max_memory=max_memory
#                 )
#
# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
#
# question = "how many dogs are in the picture?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
#
# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# pip install accelerate bitsandbytes
# import torch
# import requests
# from PIL import Image
# from transformers import Blip2Processor, Blip2ForConditionalGeneration
#
# processor = Blip2Processor.from_pretrained("/public26_data/lsx/vipergpt/pretrained_models/blip2-flan-t5-xl")
# model = Blip2ForConditionalGeneration.from_pretrained("/public26_data/lsx/vipergpt/pretrained_models/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="sequential")
#
# img_url = "/public26_data/lsx/test/MME_Benchmark_release_version/color/000000053529.jpg"
# raw_image = Image.open(img_url).convert('RGB')
#
# question = "Is there a blue hat in the image?"
# inputs = processor(raw_image, question, return_tensors="pt").to("cuda:1", torch.float16)
#
# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))
# from main_simple_lib import *
# from configs import config
# print(1)
# a = config.load_models.ram
# print(a)
# from ruamel.yaml import YAML
# yaml = YAML(typ='safe')
# # 读取 YAML 配置文件
# with open("/public26_data/lsx/vipergpt/viper/configs/base_config.yaml", 'r') as file:
#     config = yaml.load(file)
#
# # 修改配置文件中的值
# config['load_models']['ram'] = False
#
# # 保存修改后的配置文件
# with open("/public26_data/lsx/vipergpt/viper/configs/base_config.yaml", 'w') as file:
#     yaml.dump(config, file)
#
# a = config.load_models.ram
# print(a)
#
# # from main_simple_lib import *
# print(2)
# from main_simple_lib import *
# query = 'Is the word in the logo "A"? Please answer yes or no.'
# code = get_code(query)
# print(code)
from rich.syntax import Syntax
# from rich.console import Console
# console = Console(highlight=False, force_terminal=False)
# code = f'def execute_command(image, my_fig, time_wait_between_lines, syntax):'  # chat models give execute_command due to system behaviour
# from pygments import highlight
# from pygments.lexers import get_lexer_by_name
# from pygments.formatters import HtmlFormatter
#
# code = 'def execute_command(image, my_fig, time_wait_between_lines, syntax):'
# lexer = get_lexer_by_name("python", stripall=True)
# formatter = HtmlFormatter(style="monokai", linenos=True, full=True)
#
# html_code = highlight(code, lexer, formatter)
# print(html_code)
from timm.data import create_transform