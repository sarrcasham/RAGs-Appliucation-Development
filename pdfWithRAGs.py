import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# Initialize environment
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)

def create_vector_store(text_chunks):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return FAISS.from_texts(text_chunks, embeddings)

# PDF processing
pdf_path = r"C:\Users\U.pdf"
pdf_text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=150,
    separators=["\n\n", "\n", r"(?<=\. )", " "]
)

text_chunks = text_splitter.split_text(pdf_text)
vector_store = create_vector_store(text_chunks)

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
)

def rag_query(question):
    docs = vector_store.similarity_search(question, k=2)  
    context = "\n".join([d.page_content for d in docs])
    
    prompt = f"""<s>[INST]You are an expert in evaluation process of hiring, analyze the provided resume. If nothing found in resume, state that clearly. keep the answer quite sharp and only in 2-3 lines
    Candidate_personal_data: {context}
    Question: {question} [/INST]"""
    print("\n \n \n" , prompt , "\n \n \n")
    return llm(prompt)

print(rag_query("can you rate this resume out of 10?"))
# print("\n \n \n" , prompt , "\n \n \n")

# PROMPT_TEMPLATE = """
# You are an expert in Insider Risk Management. Analyze the provided activity logs for security risks, anomalies, or suspicious patterns. Summarize the findings in a single sentence, highlighting only critical risks. If no risks are found, state that clearly.
# Query: {user_query}
# Activity Logs: {document_context}
