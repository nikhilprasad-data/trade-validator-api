import pandas as pd

def select_features():

     """Filters out non-predictive categorical features (Ticker Names) based on prior statistical analysis, exporting the final ML-ready dataset."""

     feature_df  = pd.read_csv('/data/processed/engineered_market_data.csv')

     selected_df = feature_df.copy()

     selected_df.set_index('Date', inplace= True)

     column_to_drop = [col for col in selected_df.columns if 'Ticker_Name' in col]

     selected_df.drop(column_to_drop, axis= 1, inplace= True)

     output_path = 'data/processed/selected_features.csv'

     selected_df.to_csv(output_path, index= True)

     print(f"Success: Tickers dropped. Final ML data saved at {output_path}")

if __name__ == '__main__':
     select_features()