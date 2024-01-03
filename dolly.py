import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

# query = "Explain to me the difference between nuclear fission and fusion."
query = "Did Jesus weep? give proof."

res = generate_text(query)
print(res[0]["generated_text"])
