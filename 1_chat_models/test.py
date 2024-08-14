from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()
generator = pipeline('text-generation', model='distilgpt2')

response = generator("There was a maiden in a village called Aksara.")
print(response)
