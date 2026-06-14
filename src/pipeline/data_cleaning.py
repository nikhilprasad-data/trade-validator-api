import pandas as pd

def clean_market_data():

     """Loads raw market data, standardizes the Date column to datetime format, and exports the cleaned base dataset."""

     df = pd.read_csv('data/raw/raw_market_data.csv')

     df['Date'] = pd.to_datetime(df['Date'])

     output_path = 'data/processed/base_market_data.csv'

     df.to_csv(output_path, index= False)

     print(f"Success: Cleaned data saved to {output_path}")

if __name__ == '__main__':
     clean_market_data()