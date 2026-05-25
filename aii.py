from google import genai

client = genai.Client(api_key="AIzaSyBYy6Wx9Sd9qrKSgI69HyPa4GPdcynVT4c")

for model in client.models.list():
    print(model.name)