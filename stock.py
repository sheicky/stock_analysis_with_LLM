from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os

# Import and configure warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# Load environment variables
load_dotenv()

# Initialize HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# Connect to Pinecone index
pinecone_index = pc.Index("stock-index")
vectorstore = PineconeVectorStore(
    index_name="stock-index",
    embedding=hf_embeddings
)

def fetch_from_pinecone():
    """
    Fetch all namespaces from Pinecone index
    """
    try:
        index_stats = pinecone_index.describe_index_stats()
        all_namespaces = list(index_stats.get('namespaces', {}).keys())
        return all_namespaces
    except Exception as e:
        print(f"An error occurred while fetching from Pinecone: {e}")
        return []

# Initialize Groq
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

def get_stock_info(symbol: str) -> dict:
    """
    Retrieves and formats detailed information about a stock from Yahoo Finance.
    """
    data = yf.Ticker(symbol)
    stock_info = data.info

    properties = {
        "Ticker": stock_info.get('symbol', 'Information not available'),
        'Name': stock_info.get('longName', 'Information not available'),
        'Business Summary': stock_info.get('longBusinessSummary'),
        'City': stock_info.get('city', 'Information not available'),
        'State': stock_info.get('state', 'Information not available'),
        'Country': stock_info.get('country', 'Information not available'),
        'Industry': stock_info.get('industry', 'Information not available'),
        'Sector': stock_info.get('sector', 'Information not available'),
        'Market Cap': stock_info.get('marketCap', 0)
    }

    return properties

def process_stock(stock_ticker: str) -> str:
    """
    Process a single stock and store its data in Pinecone.
    """
    try:
        stock_data = get_stock_info(stock_ticker)
        stock_description = stock_data['Business Summary']

        # Use vectorstore directly
        vectorstore.add_texts(
            texts=[stock_description],
            metadatas=[stock_data]
        )

        return f"Processed {stock_ticker} successfully"
    except Exception as e:
        return f"ERROR processing {stock_ticker}: {e}"

def parallel_process_stocks(tickers: list, max_workers: int = 10) -> None:
    """
    Process multiple stocks in parallel.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_stock, ticker): ticker
            for ticker in tickers
        }

        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                print(result)
            except Exception as exc:
                print(f'{ticker} generated an exception: {exc}')

def query_pinecone(query_text, top_k=10):
    """
    Query Pinecone using the vectorstore.
    """
    try:
        results = vectorstore.similarity_search_with_score(
            query=query_text,
            k=top_k
        )
        return results
    except Exception as e:
        raise RuntimeError(f"Error querying Pinecone: {e}")

def generate_response(query, top_k, filters):
    """
    Generate a response using Pinecone and Groq.
    """
    try:
        # Query using vectorstore
        search_results = query_pinecone(query, top_k)
        
        # Format results
        matches = []
        for doc, score in search_results:
            metadata = doc.metadata
            if filters:
                sector_filter = filters.get("sector")
                min_cap = filters.get("min_cap")
                max_cap = filters.get("max_cap")
                
                if sector_filter and sector_filter != "All":
                    if metadata.get("Sector", "Unknown") != sector_filter:
                        continue
                
                market_cap = metadata.get('Market Cap', 0)
                if not (min_cap*1000000000 <= market_cap <= max_cap*1000000000):
                    continue
            
            matches.append({
                "metadata": metadata,
                "score": score
            })
        
        if not matches:
            return "No matching results found for your criteria."

        contexts = [
            f"Name: {match['metadata'].get('Name', 'Unknown')}\n"
            f"Market Cap: {match['metadata'].get('Market Cap', 'Unknown')}\n"
            f"Details: {match['metadata'].get('Business Summary', 'No details available')}"
            for match in matches
        ]

        filter_details = (
            f"Filters Applied:\n"
            f"Sector: {filters.get('sector', 'All')}\n"
            f"Market Cap min: {filters.get('min_cap')}B\n"
            f"Market Cap max: {filters.get('max_cap')}B\n"
            if filters else ""
        )

        augmented_query = (
            "<CONTEXT>\n"
            + "\n\n-------\n\n".join(contexts)
            + "\n-------\n</CONTEXT>\n\n"
            + filter_details
            + "QUESTION:\n"
            + query
        )

        system_prompt = """
        You are an expert in financial analysis and stock market trends.
        Using the context provided, answer the user's question as clearly and concisely as possible.
        When you give a response, this is the format I want you to follow:
        CompanyName(ticker): 
        - Market Cap:
        - Description:
        """

        llm_response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ],
        )
        
        return llm_response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"




