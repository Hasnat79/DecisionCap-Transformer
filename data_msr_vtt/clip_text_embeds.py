from transformers import AutoTokenizer, TFCLIPModel

model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

inputs = tokenizer(".", padding=True, return_tensors="tf")
print(f"input sentence: a dog")
print(f"encodings: {inputs['input_ids'].numpy()[0]}")
# [49406,   273,   271,   272,   277, 49407]
# print(f"decoded: {tokenizer.decode(inputs['input_ids'].numpy()[0])}")

print(f"decoded: {tokenizer.decode([49406,   273,   271,   272,   277, 49407])}")

# text_features = model.get_text_features(**inputs)
#
# print(f"text feature shape: {text_features.shape}") #1x512

# print(text_features)