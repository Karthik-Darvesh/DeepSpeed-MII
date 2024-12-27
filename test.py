import mii
pipe = mii.pipeline("meta-llama/Llama-3.1-8B-Instruct") #4.9GB
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)