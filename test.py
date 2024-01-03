print ('it works!')

from transformers import pipeline

model = pipeline("text-generation", model="microsoft/phi-2")

res = model("i'm so glad this disease is going to kill me")

print(res)
