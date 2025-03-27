import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Initialize environment
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def extract_text_from_csv(csv_path):
    # Suppress dtype warnings by setting low_memory=False
    df = pd.read_csv(csv_path, low_memory=False)
    return df.to_string(index=False)

def create_vector_store(text_chunks):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)

# File processing (specific to customer CSV dataset)
file_path = r"C:\Users\Asarv\Downloads\customers-100.csv"  
file_text = extract_text_from_csv(file_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", r"(?<=\. )", " "]
)

text_chunks = text_splitter.split_text(file_text)
vector_store = create_vector_store(text_chunks)

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

def rag_query(question):
    docs = vector_store.similarity_search(question, k=2)  
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""<s>[INST]You are an expert in customer relationship management and data analysis. Analyze the provided customer dataset and answer the query based on customer demographics, subscription patterns, and contact details. If relevant data is not found, state that clearly.
Dataset_context: {context}
Question: {question} [/INST]"""
    
    print("\n\nPrompt:\n", prompt, "\n\n")
    
    return llm(prompt)

# Example question tailored to customer dataset content
print(rag_query("list the emails of all customers whose first names start with the letter 's'."))
