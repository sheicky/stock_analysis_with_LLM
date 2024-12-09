import streamlit as st
import plotly.express as px
import pandas as pd
from stock import (
    generate_response, 
    process_stock,
    parallel_process_stocks,
    fetch_from_pinecone,
    pinecone_index,
    vectorstore
)
from datetime import datetime
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .company-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .metric-value {
        color: #00ff00;
        font-size: 20px;
    }
    .company-name {
        color: #ffffff;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Function to get company tickers
def get_company_tickers():
    url = "https://raw.githubusercontent.com/team-headstart/Financial-Analysis-and-Automation-with-LLMs/main/company_tickers.json"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.content.decode('utf-8'))
    else:
        st.error(f"Failed to download file. Status code: {response.status_code}")
        return None

# Check if data exists in Pinecone
def check_data_exists():
    try:
        namespaces = fetch_from_pinecone()
        return len(namespaces) > 0
    except Exception as e:
        st.error(f"Error checking Pinecone index: {e}")
        return False

# Initialize session state for data status
if 'data_initialized' not in st.session_state:
    st.session_state.data_initialized = check_data_exists()

# Main content
st.title("📊 Stock Market Analysis")

# Show data initialization message if needed
if not st.session_state.data_initialized:
    st.warning("⚠️ No stock data found. Please initialize the database first.")
    if st.button("Initialize Stock Database"):
        with st.spinner("Fetching and storing stock data..."):
            company_tickers = get_company_tickers()
            if company_tickers:
                tickers_to_process = [company_tickers[num]['ticker'] 
                                    for num in company_tickers.keys()]
                # Process first 50 companies for demo purposes
                parallel_process_stocks(tickers_to_process[:50])
                st.session_state.data_initialized = True
                st.success("✅ Database initialized successfully!")
                st.rerun()
            else:
                st.error("Failed to fetch company tickers")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🔍 Search Settings")
    
    sector = st.selectbox(
        "Select Sector",
        ["All", "Technology", "Healthcare", "Financials", 
         "Consumer Discretionary", "Industrials", "Energy",
         "Materials", "Consumer Staples", "Utilities", 
         "Real Estate", "Communication Services"],
        index=0
    )
    
    st.subheader("Market Cap Range (Billions $)")
    min_cap = st.number_input("Minimum", min_value=0, value=0)
    max_cap = st.number_input("Maximum", min_value=0, value=1000)
    
    top_k = st.slider(
        "Number of Results",
        min_value=1,
        max_value=20,
        value=10
    )
    
    st.subheader("🔄 Data Management")
    if st.button("Update Stock Data"):
        with st.spinner("Updating stock data..."):
            company_tickers = get_company_tickers()
            if company_tickers:
                tickers_to_process = [company_tickers[num]['ticker'] 
                                    for num in company_tickers.keys()]
                parallel_process_stocks(tickers_to_process[:50])
                st.success("✅ Stock data updated successfully!")
            else:
                st.error("Failed to fetch company tickers")

st.write("""
Enter your query about stocks and companies. Use natural language to ask about 
market caps, sectors, or business descriptions.
""")

# Query section with search button
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "What would you like to know?",
        placeholder="e.g., Show me technology companies with market cap over 100 billion"
    )
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Process query
if search_button and query:
    with st.spinner("Analyzing..."):
        filters = {
            "sector": sector,
            "min_cap": min_cap,
            "max_cap": max_cap
        }
        
        response = generate_response(query, top_k, filters)
        
        # Parse the response to extract structured data
        companies = []
        current_company = {}
        
        for line in response.split('\n'):
            if '(' in line and ')' in line:
                if current_company:
                    companies.append(current_company)
                current_company = {'name': line.strip()}
            elif 'Market Cap:' in line:
                current_company['market_cap'] = float(line.split(':')[1].strip().replace(',', ''))
            elif 'Description:' in line:
                current_company['description'] = line.split(':')[1].strip()
        
        if current_company:
            companies.append(current_company)

        # Create visualizations
        if companies:
            # Market Cap Comparison
            df = pd.DataFrame(companies)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Market Cap Comparison")
                fig = px.bar(
                    df,
                    x='name',
                    y='market_cap',
                    title='Company Market Capitalizations',
                    labels={'name': 'Company', 'market_cap': 'Market Cap ($B)'},
                    color='market_cap',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Market Share")
                fig = px.pie(
                    df,
                    values='market_cap',
                    names='name',
                    title='Market Share Distribution'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed Company Cards
            st.subheader("Detailed Analysis")
            for company in companies:
                with st.container():
                    st.markdown(f"""
                    <div class="company-card">
                        <div class="company-name">{company['name']}</div>
                        <div class="metric-value">Market Cap: ${company['market_cap']:,.2f}B</div>
                        <p>{company['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer with additional metrics
st.markdown("---")
metrics_cols = st.columns(4)
with metrics_cols[0]:
    st.metric("Total Companies", len(companies) if 'companies' in locals() else 0)
with metrics_cols[1]:
    if 'companies' in locals() and companies:
        avg_market_cap = sum(c['market_cap'] for c in companies) / len(companies)
        st.metric("Avg Market Cap", f"${avg_market_cap:,.2f}B")
with metrics_cols[2]:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with metrics_cols[3]:
    st.markdown("**Data Source:** Yahoo Finance")
    st.markdown("**Coded by :** Sheick with ❤️ ")







