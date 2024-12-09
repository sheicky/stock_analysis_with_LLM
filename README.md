
# Stock Analysis with LLMs: Research Automation for Quantitative Investment  

This repository contains a cutting-edge project designed for **Company X**, a leading quantitative investment fund, exploring the use of **Large Language Models (LLMs)** to gain a competitive advantage in stock selection. The system is built to automate research and streamline the process of identifying promising investment opportunities by analyzing vast amounts of textual data and market metrics.  

## Project Overview  

The project focuses on building an intelligent **Research Automation System** capable of:  
- Understanding **natural language queries** to identify relevant stocks (e.g., *"What are companies that build data centers?"*).  
- Enabling advanced filtering and search capabilities for all stocks listed on the **New York Stock Exchange (NYSE)** based on key metrics, including **Market Capitalization**, **Volume**, **Sector**, and more.  
- Leveraging state-of-the-art **AI technologies** to bridge the gap between textual data analysis and actionable investment insights.  

This innovative system positions Company X as a leader in leveraging AI for smarter and faster decision-making in stock selection.  

---

## Features  

### 1. **Natural Language Stock Queries**  
   - Users can enter complex queries in natural language to identify stocks meeting specific criteria.  
   - Example: *"Show me tech companies with a market capitalization greater than $10 billion."*  

### 2. **Search by Metrics**  
   - Advanced search options for stocks based on:  
     - **Market Capitalization**  
     - **Volume**  
     - **Industry/Sector**  
     - And more.  

### 3. **Sentiment Analysis for Trading**  
   - Integrates **Large Language Models** to analyze sentiment from news articles, reports, and other textual sources, enhancing decision-making.  

### 4. **Real-Time Data Integration**  
   - Retrieves up-to-date market data using **Yahoo Finance (yFinance)**.  

### 5. **Embeddings and Similarity Search**  
   - Uses **vector embeddings** and similarity search to match user queries with relevant stocks efficiently.  

---

## Tech Stack  

### Core Technologies  
- **Streamlit**: Interactive and user-friendly web interface.  
- **Pinecone**: Vector database for fast similarity search.  
- **OpenAI API**: Natural Language Processing (NLP) with GPT models.  
- **Groq API**: High-performance AI computing for model execution.  

### Libraries and Tools  
- **LangChain**: Framework for working with LLMs and embeddings.  
- **HuggingFace Sentence Transformers**: For embedding textual data.  
- **scikit-learn**: To compute cosine similarity between embeddings.  
- **yFinance**: Real-time market data retrieval.  

### Other Dependencies  
- **dotenv**: Securely manage environment variables.  
- **NumPy**: Data manipulation and analysis.  
- **Requests**: To handle API requests.  

---

## Installation  

### Prerequisites  
- Python 3.8+  
- API keys for **OpenAI**, **Pinecone**, and **Groq**.  

### Steps to Run Locally  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/sheicky/stock_analysis_with_LLM.git  
   cd stock_analysis_with_LLM  
   ```  

2. **Install dependencies**:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Set up environment variables**:  
   Create a `.env` file in the root directory and add the following:  
   ```env  
   OPENAI_API_KEY=<your_openai_api_key>  
   PINECONE_API_KEY=<your_pinecone_api_key>  
   GROQ_API_KEY=<your_groq_api_key>  
   ```  

4. **Run the Streamlit application**:  
   ```bash  
   streamlit run app.py  
   ```  

---

## How It Works  

1. **Query Processing**  
   - User inputs are processed using **OpenAI’s GPT model**, converting natural language queries into actionable search commands.  

2. **Stock Retrieval**  
   - Stocks are filtered using **Yahoo Finance** data and further refined using vector similarity with **Pinecone**.  

3. **Sentiment Analysis**  
   - News articles and reports are embedded using **HuggingFace Sentence Transformers**, and sentiment scores are computed to aid trading decisions.  

---

## Future Enhancements  

- **Deep Sentiment Analysis**: Integrate advanced LLMs for context-aware sentiment scoring.  
- **Multi-Market Support**: Extend coverage to global stock markets beyond the NYSE.  
- **Prediction Models**: Incorporate time-series forecasting for price and volume trends.  

---
