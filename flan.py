# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
# model_name = "google/mt5-large"
# model_name = "google/umt5-xl"
# tokenizer = MT5Tokenizer.from_pretrained(model_name)
# model = MT5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

input_text = "translate English to spanish: It is working well."
# input_text = "Did Jesus weep? respond with any proof verses if they exist"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids,max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
