from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text = ["caption"],images=image, return_tensors="pt")

image_features = model.get_image_features(**inputs)

outputs = model(**inputs)
text_embeds = outputs['text_embeds']
pooler_output = outputs['text_model_output']['pooler_output']