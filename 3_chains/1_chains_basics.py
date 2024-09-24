import torch
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# model_name = "distilbert-base-uncased-distilled-squad"
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# model = DistilBertForQuestionAnswering.from_pretrained(model_name)


# Define a custom LLM class for DistilBERT
# class DistilBertQA:
#     def __init__(self, tokenizer, model):
#         self.tokenizer = tokenizer
#         self.model = model
#
#     def answer(self, context: str, question: str) -> str:
#         # Tokenize inputs
#         inputs = self.tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)
#
#         # Get start and end scores for the answer
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             start_scores, end_scores = outputs.start_logits, outputs.end_logits
#
#         # Find the tokens with the highest start and end scores
#         start_index = torch.argmax(start_scores)
#         end_index = torch.argmax(end_scores) + 1
#
#         # Decode the tokens into the answer
#         answer = self.tokenizer.convert_tokens_to_string(
#             self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
#         )
#
#         return answer
#
#
# # Initialize the custom LLM
# qa_model = DistilBertQA(tokenizer, model)
#
# prompt_template = PromptTemplate.from_template(
#     "Context: {context}\nWho is {name}?"
# )
#
# # Initialize a simple string output parser
# output_parser = StrOutputParser()
#
#
# def run_chain(context, name):
#     # Format the prompt
#     prompt = prompt_template.format(context=context, name=name)
#
#     # Use the custom QA model to get an answer
#     answer = qa_model.answer(context, prompt)
#
#     # Parse the output (this step is redundant in this case, but here for structure)
#     parsed_answer = output_parser.parse(answer)
#
#     return parsed_answer
#
#
# context = """Bill Gates is a co-founder of Microsoft Corporation, one of the world's largest software companies.
# He was born on October 28, 1955, in Seattle, Washington, USA. Gates attended Harvard University but dropped out
# to pursue his passion for computer programming. Under his leadership, Microsoft grew into a technology giant.
# He is also known for his philanthropic work through the Bill & Melinda Gates Foundation.
# """
# name = "Bill Gates"
#
# # Run the chain
# response = run_chain(context, name)
# print("Answer:", response)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

chain = prompt_template | model | StrOutputParser()
# Chain Runnable
#result = chain.invoke({"topic": "Chuck Norris", "joke_count": 10})

# Stream Runnable
for chunk in chain.stream({"topic": "Chuck Norris", "joke_count": 5}):
    print(chunk, end="|", flush=True)

#print(result)