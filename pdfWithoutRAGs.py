import os
from groq import Groq
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

load_dotenv()

client = Groq(api_key="")
pdf_path = r"C:\Users\ Plan.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
text_chunks = split_text(pdf_text)
try:
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[   {"role": "system", "content": "You are an AI assistant that answers questions based on the content of a PDF."},
            {"role": "user", "content": f"Here's the content of the PDF: {text_chunks[:5]}... Please answer questions based on this content."},
            {"role": "user", "content": "What is the main topic of this PDF?"}]
    )
    print("API Response:", completion.choices[0].message.content)
    print("Connection successful!")
except Exception as e:
    print("Connection failed:", str(e))




