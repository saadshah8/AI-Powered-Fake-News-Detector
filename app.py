# # import os
# # from dotenv import load_dotenv
# # from fastapi import FastAPI, Request, Form
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# # from newspaper import Article
# # import requests

# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain_community.vectorstores import Chroma
# # from transformers import pipeline
# # from huggingface_hub import InferenceApi

# # # Load environment variables
# # load_dotenv()
# # os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # # Initialize Hugging Face Embeddings
# # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # # Initialize text splitter
# # text_splitter = RecursiveCharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=200,
# #     length_function=len
# # )

# # # Initialize Hugging Face Inference API
# # inference = InferenceApi(
# #     repo_id="facebook/bart-large-mnli", 
# #     token=os.getenv("HF_TOKEN")
# # )

# # # Custom download function to fetch news
# # def custom_download(url):
# #     headers = {
# #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
# #         'Accept-Language': 'en-US,en;q=0.9',
# #         'Referer': 'https://www.google.com/'
# #     }
# #     response = requests.get(url, headers=headers)
# #     if response.status_code == 200:
# #         return response.content
# #     else:
# #         print(f"Failed to download article. Status code: {response.status_code}")
# #         return None

# # # Fetch news content using Newspaper3k
# # def fetch_news_content(url):
# #     article = Article(url)
# #     html = custom_download(url)
# #     if html:
# #         article.set_html(html)
# #         article.parse()
# #         title = article.title
# #         content = article.text
# #         return title, content
# #     else:
# #         return "Failed to retrieve title", "Failed to retrieve content"

# # # Store news in Vector Database (ChromaDB)
# # def store_news_in_db(url):
# #     title, content = fetch_news_content(url)
# #     chunks = text_splitter.split_text(content)
    
# #     # Store in ChromaDB
# #     vector_db = Chroma.from_texts(
# #         texts=chunks,
# #         embedding=embeddings,
# #         metadatas=[{"title": title, "url": url}] * len(chunks)
# #     )
# #     return vector_db

# # # Check for contradictions using Hugging Face Inference API
# # # def check_contradictions(text, labels):
# # #     result = inference(inputs=text, parameters={"candidate_labels": labels})
# # #     return result

# # def check_contradictions(text, labels):
# #     # Initialize the zero-shot classification model
# #     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# #     result = classifier(text, candidate_labels=labels)
# #     return result

# # # Fact-check news by comparing with VectorDB
# # def fact_check_news(news_text, vector_db):

# #     labels = ["contradiction", "neutral", "entailment"]
# #     contradiction_result = check_contradictions(news_text, labels)
    
# #     search_results = vector_db.similarity_search(news_text, k=3)
    
# #     # Check contradictions against top 3 similar articles
# #     contradictions = []
# #     for result in search_results:
# #         contradiction_result = check_contradictions(
# #             news_text, 
# #             labels=["true", "false", "misleading", "accurate"]
# #         )
# #         contradictions.append(contradiction_result)
    
# #     return contradictions

# # # FastAPI app setup
# # app = FastAPI()

# # # Define templates folder path
# # templates = Jinja2Templates(directory="templates")

# # @app.get("/", response_class=HTMLResponse)
# # async def upload_news_page(request: Request):
# #     return templates.TemplateResponse("index.html", {"request": request})

# # @app.post("/upload")
# # async def upload_news(request: Request, url: str = Form(...)):
# #     vector_db = store_news_in_db(url)
# #     success_message = "The article has been successfully uploaded and stored."
# #     return templates.TemplateResponse("index.html", {"request": request, "success_message": success_message})

# # @app.get("/fact-check", response_class=HTMLResponse)
# # async def fact_check_page(request: Request):
# #     return templates.TemplateResponse("fact_check.html", {"request": request})

# # @app.post("/fact-check")
# # async def fact_check(request: Request, news_text: str = Form(...)):
# #     vector_db = Chroma(embedding_function=embeddings)
# #     result = fact_check_news(news_text, vector_db)
# #     return templates.TemplateResponse("fact_check.html", {"request": request, "result": result})



# import os
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from newspaper import Article
# import requests
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from transformers import pipeline
# from huggingface_hub import InferenceApi

# # Load environment variables
# load_dotenv()
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # Initialize Hugging Face Embeddings for text encoding
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Initialize text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

# # Initialize Hugging Face Inference API for contradiction detection
# inference = InferenceApi(
#     repo_id="facebook/bart-large-mnli", 
#     token=os.getenv("HF_TOKEN")
# )

# # Custom download function to fetch news
# def custom_download(url):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
#         'Accept-Language': 'en-US,en;q=0.9',
#         'Referer': 'https://www.google.com/'
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.content
#     else:
#         print(f"Failed to download article. Status code: {response.status_code}")
#         return None

# # Fetch news content using Newspaper3k
# def fetch_news_content(url):
#     article = Article(url)
#     html = custom_download(url)
#     if html:
#         article.set_html(html)
#         article.parse()
#         title = article.title
#         content = article.text
#         return title, content
#     else:
#         return "Failed to retrieve title", "Failed to retrieve content"

# # Store news in Vector Database (ChromaDB)
# def store_news_in_db(url):
#     title, content = fetch_news_content(url)
#     chunks = text_splitter.split_text(content)
    
#     # Store in ChromaDB
#     vector_db = Chroma.from_texts(
#         texts=chunks,
#         embedding=embeddings,
#         metadatas=[{"title": title, "url": url}] * len(chunks)
#     )
#     return vector_db

# # Check for contradictions using Hugging Face Inference API (Zero-shot classification)
# def check_contradictions(text, labels):
#     result = inference(inputs=text, parameters={"candidate_labels": labels})
#     return result

# # Fact-check news by comparing with VectorDB
# def fact_check_news(news_text, vector_db):
#     labels = ["contradiction", "neutral", "entailment"]
#     contradiction_result = check_contradictions(news_text, labels)
    
#     search_results = vector_db.similarity_search(news_text, k=3)
    
#     # Check contradictions against top 3 similar articles
#     contradictions = []
#     for result in search_results:
#         contradiction_result = check_contradictions(
#             news_text, 
#             labels=["true", "false", "misleading", "accurate"]
#         )
#         contradictions.append(contradiction_result)
    
#     return contradictions

# # FastAPI app setup
# app = FastAPI()

# # Define templates folder path
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def upload_news_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/upload")
# async def upload_news(request: Request, url: str = Form(...)):
#     vector_db = store_news_in_db(url)
#     success_message = "The article has been successfully uploaded and stored."
#     return templates.TemplateResponse("index.html", {"request": request, "success_message": success_message})

# @app.get("/fact-check", response_class=HTMLResponse)
# async def fact_check_page(request: Request):
#     return templates.TemplateResponse("fact_check.html", {"request": request})

# @app.post("/fact-check")
# async def fact_check(request: Request, news_text: str = Form(...)):
#     vector_db = Chroma(embedding_function=embeddings)  # Ensure the vector database is correctly initialized
#     result = fact_check_news(news_text, vector_db)
#     return templates.TemplateResponse("fact_check.html", {"request": request, "result": result})


# import os
# import requests
# from newspaper import Article
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq

# # Load environment variables
# load_dotenv()

# # Set Hugging Face Token
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # Setup HuggingFace Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Setup ChromaDB Directories for Persistent Storage
# TRUSTED_DB_DIR = "trusted_db"
# FLAGGED_DB_DIR = "flagged_db"

# # Initialize text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

# def custom_download(url):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
#         'Accept-Language': 'en-US,en;q=0.9',
#         'Referer': 'https://www.google.com/'
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.content
#     else:
#         print(f"Failed to download article. Status code: {response.status_code}")
#         return None

# def fetch_news_content(url):
#     article = Article(url)
#     html = custom_download(url)
#     if html:
#         article.set_html(html)
#         article.parse()
#         title = article.title
#         content = article.text
#         return title, content
#     else:
#         return "Failed to retrieve title", "Failed to retrieve content"

# def store_news_in_db(url, article_type):
#     # Fetch news content
#     title, content = fetch_news_content(url)
    
#     # Split content into chunks
#     chunks = text_splitter.split_text(content)
    
#     # Determine which database to use
#     db_dir = TRUSTED_DB_DIR if article_type == "trusted" else FLAGGED_DB_DIR
    
#     # Load existing Vector DB or create a new one
#     vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
#     # Store in ChromaDB
#     vector_db.add_texts(
#         texts=chunks,
#         metadatas=[{"title": title, "url": url, "type": article_type}] * len(chunks)
#     )
    
#     # Persist data
#     vector_db.persist()
#     return vector_db

# def fact_check_news(news_text):
#     # Initialize LLM
#     llm = ChatGroq(model="llama3-8b-8192")
    
#     # Load both databases
#     trusted_db = Chroma(persist_directory=TRUSTED_DB_DIR, embedding_function=embeddings)
#     flagged_db = Chroma(persist_directory=FLAGGED_DB_DIR, embedding_function=embeddings)
    
#     # Search in both databases
#     trusted_results = trusted_db.similarity_search_with_score(news_text, k=1)
#     flagged_results = flagged_db.similarity_search_with_score(news_text, k=1)
    
#     # Calculate confidence scores (converting similarity to percentage)
#     trusted_score = 0
#     flagged_score = 0
    
#     if trusted_results:
#         trusted_score = (1 - trusted_results[0][1]) * 100  # Convert distance to similarity percentage
#     if flagged_results:
#         flagged_score = (1 - flagged_results[0][1]) * 100  # Convert distance to similarity percentage
    
#     # Create retrieval chain with appropriate database based on scores
#     if trusted_score > flagged_score:
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=trusted_db.as_retriever()
#         )
#         source_type = "TRUSTED"
#         confidence = trusted_score
#     else:
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=flagged_db.as_retriever()
#         )
#         source_type = "FLAGGED"
#         confidence = flagged_score
    
#     # Get the response from the chain
#     response = qa_chain.invoke(news_text)
    
#     return {
#         "response": response,
#         "source_type": source_type,
#         "confidence": round(confidence, 2)
#     }

# # FastAPI app setup
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# @app.get("/", response_class=HTMLResponse)
# async def upload_news_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/upload")
# async def upload_news(request: Request, url: str = Form(...), article_type: str = Form(...)):
#     vector_db = store_news_in_db(url, article_type)
#     success_message = f"The article has been successfully uploaded and stored as a {article_type} source."
#     return templates.TemplateResponse("index.html", {"request": request, "success_message": success_message})

# @app.get("/fact-check", response_class=HTMLResponse)
# async def fact_check_page(request: Request):
#     return templates.TemplateResponse("fact_check.html", {"request": request})

# @app.post("/fact-check")
# async def fact_check(request: Request, news_text: str = Form(...)):
#     result = fact_check_news(news_text)
#     return templates.TemplateResponse("fact_check.html", {"request": request, "result": result})


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import os
# import requests
# from newspaper import Article
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq

# # Load environment variables
# load_dotenv()

# # Set Hugging Face Token
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# # Setup HuggingFace Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Setup ChromaDB Directory for Persistent Storage
# CHROMA_DB_DIR = "chroma_db"

# # Initialize text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
#     length_function=len
# )

# # Function to download and parse news content
# def custom_download(url):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
#         'Accept-Language': 'en-US,en;q=0.9',
#         'Referer': 'https://www.google.com/'
#     }
#     response = requests.get(url, headers=headers)
#     if response.status_code == 200:
#         return response.content
#     else:
#         print(f"Failed to download article. Status code: {response.status_code}")
#         return None

# def fetch_news_content(url):
#     article = Article(url)
#     html = custom_download(url)
#     if html:
#         article.set_html(html)
#         article.parse()
#         title = article.title
#         content = article.text
#         return title, content
#     else:
#         return "Failed to retrieve title", "Failed to retrieve content"

# # Function to Store News Content in ChromaDB
# def store_news_in_db(url):
#     # Fetch news content
#     title, content = fetch_news_content(url)
    
#     # Split content into chunks
#     chunks = text_splitter.split_text(content)
    
#     # Load existing Vector DB or create a new one
#     vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
#     # Store in ChromaDB
#     vector_db.add_texts(
#         texts=chunks,
#         metadatas=[{"title": title, "url": url}] * len(chunks)
#     )
    
#     # Persist data
#     vector_db.persist()
#     return vector_db

# # Function to Fact-Check News
# def fact_check_news(news_text, vector_db):
#     # Initialize LLM
#     llm = ChatGroq(model="llama3-8b-8192")
    
#     # Create retrieval chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_db.as_retriever()
#     )
    
#     # Query the chain
#     response = qa_chain.invoke(news_text)
#     return response

# # FastAPI app setup
# app = FastAPI()

# # Define the templates folder path
# templates = Jinja2Templates(directory="templates")

# # Home Page
# @app.get("/", response_class=HTMLResponse)
# async def upload_news_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Upload News Page
# @app.post("/upload")
# async def upload_news(request: Request, url: str = Form(...)):
#     vector_db = store_news_in_db(url)
#     success_message = "The article has been successfully uploaded and stored."
#     return templates.TemplateResponse("index.html", {"request": request, "success_message": success_message})

# # Fact Check Page
# @app.get("/fact-check", response_class=HTMLResponse)
# async def fact_check_page(request: Request):
#     return templates.TemplateResponse("fact_check.html", {"request": request})

# # Fact Check Functionality
# @app.post("/fact-check")
# async def fact_check(request: Request, news_text: str = Form(...)):
#     # Load existing Vector DB
#     vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
#     # Perform Fact Check
#     result = fact_check_news(news_text, vector_db)
#     return templates.TemplateResponse("fact_check.html", {"request": request, "result": result['result']})


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import requests
from newspaper import Article
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set Hugging Face Token
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Setup HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setup ChromaDB Directories for Persistent Storage
TRUSTED_DB_DIR = "trusted_db"
FLAGGED_DB_DIR = "flagged_db"

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Function to download and parse news content
def custom_download(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        print(f"Failed to download article. Status code: {response.status_code}")
        return None

def fetch_news_content(url):
    article = Article(url)
    html = custom_download(url)
    if html:
        article.set_html(html)
        article.parse()
        title = article.title
        content = article.text
        return title, content
    else:
        return "Failed to retrieve title", "Failed to retrieve content"

# Function to Store News Content in ChromaDB
def store_news_in_db(url, category):
    # Fetch news content
    title, content = fetch_news_content(url)
    
    # Split content into chunks
    chunks = text_splitter.split_text(content)
    
    # Choose the DB based on category
    if category == "trusted":
        vector_db = Chroma(persist_directory=TRUSTED_DB_DIR, embedding_function=embeddings)
    else:
        vector_db = Chroma(persist_directory=FLAGGED_DB_DIR, embedding_function=embeddings)
    
    # Store in ChromaDB
    vector_db.add_texts(
        texts=chunks,
        metadatas=[{"title": title, "url": url}] * len(chunks)
    )
    
    return vector_db

# Function to Check News Credibility
def check_news_credibility(news_text):
    # Load both databases
    trusted_db = Chroma(persist_directory=TRUSTED_DB_DIR, embedding_function=embeddings)
    flagged_db = Chroma(persist_directory=FLAGGED_DB_DIR, embedding_function=embeddings)

    # Retrieve similar contexts
    trusted_context = trusted_db.similarity_search(news_text, k=2)
    flagged_context = flagged_db.similarity_search(news_text, k=2)

    # Prepare contexts for prompt
    trusted_content = " ".join([doc.page_content for doc in trusted_context])
    flagged_content = " ".join([doc.page_content for doc in flagged_context])

    # Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["news", "True_News", "flagged_news"],
        
        template = """
            You are an AI model tasked with verifying the credibility of a news statement. 
            You will be given:
            - News Statement: The statement to be verified.
            - True News: Trusted news articles for comparison.
            - Flagged News: Articles flagged for misinformation. 

            Your task is to:
            1. Compare the news statement with True News and Flagged News.
            2. Check for contradictions, inconsistencies, or misinformation.
            3. Provide a confidence score from 0 to 10:
                - 0: Completely False
                - 5: Inconclusive
                - 10: Completely True
            4. Make a concise conclusion:
                - TRUE: If the statement is consistent with True News and not contradicted by Flagged News.
                - FALSE: If the statement contradicts True News or aligns with Flagged News.
                - INCONCLUSIVE: If evidence is insufficient to determine truth or falsehood.

            News Statement: {news}

            True News: {True_News}

            Flagged News: {flagged_news}

            Respond **ONLY** in the following format, using new lines for each item:
            
            Credibility Score: [Confidence Score]  
            Verdict: [TRUE/FALSE/INCONCLUSIVE]  
            Explanation:  
            - [Very Clear and Very concise reasoning]  
            - [Supporting evidence or analysis very briefly]  
            - [Optional: Very Brief Additional relevant context]  

            Do not include any introductory phrases, such as "Here is the response."
        """

    )

    # Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192")

    chain = (
        prompt_template
        | llm
    )

    # Run the chain
    result = chain.invoke({
        "news": news_text,
        "True_News": trusted_content,
        "flagged_news": flagged_content
    })

    return result

# FastAPI app setup
app = FastAPI()

# Define the templates folder path
templates = Jinja2Templates(directory="templates")

# Home Page
@app.get("/", response_class=HTMLResponse)
async def upload_news_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Upload News Page
@app.post("/upload")
async def upload_news(request: Request, url: str = Form(...), category: str = Form(...)):
    store_news_in_db(url, category)
    success_message = f"The article has been successfully uploaded and stored in {category} database."
    return templates.TemplateResponse("index.html", {"request": request, "success_message": success_message})

# Fact Check Page
@app.get("/fact-check", response_class=HTMLResponse)
async def fact_check_page(request: Request):
    return templates.TemplateResponse("fact_check.html", {"request": request})

# Fact Check Functionality
@app.post("/fact-check")
async def fact_check(request: Request, news_text: str = Form(...)):
    result = check_news_credibility(news_text)
    return templates.TemplateResponse("fact_check.html", {"request": request, "result": result.content})

