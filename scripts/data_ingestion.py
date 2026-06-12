import yfinance as yf
import pandas as pd
import time

print("SDE Pipeline: Fetching 10-Year Master Market Data...")

tickers_list = [
    "^NSEI",      
    "^NSEBANK",  
    "RELIANCE.NS", 
    "HDFCBANK.NS", 
    "TCS.NS",     
    "INFY.NS",   
    "ITC.NS",     
    "LT.NS"       
]

all_data_frames = []

for ticker in tickers_list:
    print(f"Downloading 10-Year data for: {ticker}...")
    try:
        df = yf.download(ticker, period="10y", interval="1d", progress=False)
        
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel('Ticker')
                
            df['Ticker_Name'] = ticker 
            
            all_data_frames.append(df)
            print(f"Success: {len(df)} rows fetched.")
        else:
            print(f"Warning: No data found for {ticker}.")
            
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        
    time.sleep(1)

if all_data_frames:
    master_data = pd.concat(all_data_frames)
    
    master_data.reset_index(inplace=True)
    
    csv_filename = "data/raw/raw_market_data.csv"
    master_data.to_csv(csv_filename, index=False)
    
    print("Pipeline Executed.")
    print(f"File Saved: '{csv_filename}'")
    print(master_data.head())
else:
    print("Fatal Error: Pipeline failed to fetch any data.")