from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import json
import yfinance as yf
import concurrent.futures
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np
import requests
import os
import logging
import time

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

def process_stock(stock_ticker: str, company_data: dict = None) -> str:
    """
    Process a single stock with enhanced error handling and data enrichment
    """
    try:
        # Look for additional information in company_data
        ticker_info = None
        if company_data:
            for _, company in company_data.items():
                if company['ticker'] == stock_ticker:
                    ticker_info = company
                    break
        
        # Get Yahoo Finance data
        stock_data = get_stock_info(stock_ticker)
        
        # Enrich stock_data with information from company_tickers.json
        if ticker_info:
            stock_data.update({
                'CIK': ticker_info.get('cik_str'),
                'Title': ticker_info.get('title'),
            })
            
        if not stock_data.get('Business Summary'):
            return f"SKIP {stock_ticker}: No business summary available"
            
        rich_description = f"""
        {stock_data['Name']} ({stock_data['Ticker']}) is a company in the {stock_data['Sector']} 
        and {stock_data['Industry']} industry. 
        CIK: {stock_data.get('CIK', 'N/A')}
        Official Title: {stock_data.get('Title', 'N/A')}
        
        {stock_data['Business Summary']}
        
        Location: {stock_data['City']}, {stock_data['Country']}
        Market Cap: {stock_data['Market Cap']}
        """
        
        vectorstore.add_texts(
            texts=[rich_description],
            metadatas=[stock_data]
        )
        
        return f"SUCCESS {stock_ticker}"
        
    except Exception as e:
        return f"ERROR {stock_ticker}: {str(e)}"
def get_company_tickers():
    """
    Downloads and parses the Stock ticker symbols from the GitHub-hosted SEC company tickers JSON file.

    Returns:
        dict: A dictionary containing company tickers and related information.

    Notes:
        The data is sourced from the official SEC website via a GitHub repository:
        https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json
    """
    # URL to fetch the raw JSON file from GitHub
    url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"

    # Making a GET request to the URL
    response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        # Parse the JSON content directly
        company_tickers = json.loads(response.content.decode('utf-8'))

        # Optionally save the content to a local file for future use
        with open("company_tickers.json", "w", encoding="utf-8") as file:
            json.dump(company_tickers, file, indent=4)

        print("File downloaded successfully and saved as 'company_tickers.json'")
        return company_tickers
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        return None
def parallel_process_stocks(tickers, batch_size=50, max_retries=3):
    """
    Process stocks in parallel batches
    
    Args:
        tickers (list): List of stock tickers to process
        batch_size (int): Number of stocks to process in each batch
        max_retries (int): Maximum number of retries for failed processing
    """
    # Get company data once
    company_data = get_company_tickers()
    if not company_data:
        print("Failed to retrieve company data")
        return 0
    
    print(f"Total number of tickers to process: {len(tickers)}")
    
    processed_count = 0
    failed_tickers = []
    
    # Process in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(tickers)//batch_size + 1}")
        
        # Process the batch with company_data
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(process_stock, ticker, company_data): ticker
                for ticker in batch
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        processed_count += len(batch)
        print(f"Progress: {processed_count}/{len(tickers)} stocks processed")
        
        # Add a small delay between batches to avoid rate limiting
        time.sleep(1)
    
    return processed_count

def generate_response(query: str, top_k: int = 10, filters: dict = None) -> str:
    """
    Generates a structured response based on user query by searching the Pinecone database.
    
    Args:
        query (str): The user's query
        top_k (int): Number of results to return
        filters (dict): Optional filters for search (sector, market cap min/max)
    
    Returns:
        str: Formatted response with company information
    """
    try:
        # Search for similar documents
        results = vectorstore.similarity_search(
            query,
            k=top_k
        )
        
        # Filter results
        filtered_results = []
        for doc in results:
            metadata = doc.metadata
            
            # Apply filters if specified
            if filters:
                # Filter by sector
                if filters["sector"] != "All" and metadata.get("Sector") != filters["sector"]:
                    continue
                    
                # Filter by market cap
                market_cap = float(metadata.get("Market Cap", 0)) / 1_000_000_000  # Convert to billions
                if market_cap < filters["min_cap"] or market_cap > filters["max_cap"]:
                    continue
                    
            filtered_results.append(doc)
        
        # Format response
        response = ""
        for doc in filtered_results:
            metadata = doc.metadata
            market_cap = float(metadata.get("Market Cap", 0)) / 1_000_000_000  # Convert to billions
            
            response += f"{metadata.get('Name')} ({metadata.get('Ticker')})\n"
            response += f"Market Cap: {market_cap}\n"
            response += f"Description: {metadata.get('Business Summary', 'No description available')}\n\n"
            
        return response if response else "No companies match your criteria."
        
    except Exception as e:
        return f"An error occurred while generating response: {str(e)}"

# Code to run the processing
if __name__ == "__main__":
    print("Starting stock processing...")
    total_processed = parallel_process_stocks(batch_size=50)
    print(f"\nProcessing completed. Total stocks processed: {total_processed}")




