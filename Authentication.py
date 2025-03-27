# from langchain_openai import ChatOpenAI
# import os

# # Set up Hugging Face credentials
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 

# llm = ChatOpenAI(
#     model_name="tgi",
#     openai_api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
#     openai_api_base="https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/",
# )

# response = llm.invoke("Explain quantum computing simply")
# print(response.content)


import os
from dotenv import load_dotenv

load_dotenv()
print("GROQ_API_KEY exists:", "YES" if "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" in os.environ else "NO")