import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_default_device("cuda")

model_name = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

# query = '''def print_prime(n):
#    """
#    Print all primes between 1 and n
#    """'''
query = "did Jesus weep? give proof."
inputs = tokenizer(query, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
