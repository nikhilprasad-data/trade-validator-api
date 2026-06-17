import pandas as pd
import numpy as np

def build_features():

     """Computes technical indicators (RSI) and candlestick patterns, applies one-hot encoding, and exports the final ML-ready dataset."""

     df = pd.read_csv('data/processed/base_market_data.csv')

     feature_df = df.copy()

     feature_df['Next_Close'] = feature_df.groupby('Ticker_Name')['Close'].shift(-1)

     feature_df['Target'] = (feature_df['Next_Close'] > feature_df['Close']).astype(int)

     feature_df['Price_Change'] = feature_df.groupby('Ticker_Name')['Close'].diff()

     feature_df['Gains'] = np.where(feature_df['Price_Change'] < 0 , 0, feature_df['Price_Change'])

     feature_df['Losses'] = np.where(feature_df['Price_Change'] > 0, 0, np.abs(feature_df['Price_Change']))

     feature_df['Avg_Gain'] = feature_df.groupby('Ticker_Name')['Gains'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())

     feature_df['Avg_Loss'] = feature_df.groupby('Ticker_Name')['Losses'].transform(lambda x: x.ewm(alpha= 1/14, adjust= False).mean())

     feature_df['RS'] = feature_df['Avg_Gain'] / feature_df['Avg_Loss']

     feature_df['RSI'] = 100 - (100 / (1+ feature_df['RS']))

     feature_df['Body_of_Candle'] = np.abs(feature_df['Open'] - feature_df['Close'])

     feature_df['Range_of_Candle'] = feature_df['High'] - feature_df['Low']

     feature_df['Doji'] = np.where(feature_df['Body_of_Candle'] <= ((feature_df['Range_of_Candle']) * 0.1), 1, 0)

     feature_df['Prev_Open'] = feature_df.groupby('Ticker_Name')['Open'].shift(1)

     feature_df['Prev_Close'] = feature_df.groupby('Ticker_Name')['Close'].shift(1)

     feature_df['Bullish_Engulfing'] = np.where(((feature_df['Prev_Close'] < feature_df['Prev_Open'])& 
                                            (feature_df['Close'] > feature_df['Open'])&
                                            (feature_df['Close'] > feature_df['Prev_Open'])&
                                            (feature_df['Open'] < feature_df['Prev_Close'])), 1, 0)
     
     feature_df['Bearish_Engulfing'] = np.where(((feature_df['Prev_Close'] > feature_df['Prev_Open'])&
                                            (feature_df['Close'] < feature_df['Open'])&
                                            (feature_df['Open'] > feature_df['Prev_Close'])&
                                            (feature_df['Close'] < feature_df['Prev_Open'])), 1, 0)
     
     feature_df['Lower_Shadow'] = (np.minimum(feature_df['Open'], feature_df['Close'])) - feature_df['Low']

     feature_df['Upper_Shadow'] = feature_df['High'] - (np.maximum(feature_df['Open'], feature_df['Close']))

     feature_df['Hammer'] = np.where((feature_df['Lower_Shadow'] >= (2* feature_df['Body_of_Candle']))& 
                                     (feature_df['Upper_Shadow'] <= feature_df['Body_of_Candle']), 1, 0)

     feature_df['EMA_12'] = feature_df['Close'].ewm(span=12, adjust=False).mean()
     feature_df['EMA_26'] = feature_df['Close'].ewm(span=26, adjust=False).mean()

     feature_df['MACD'] = feature_df['EMA_12'] - feature_df['EMA_26']

     feature_df['MACD_Signal'] = feature_df['MACD'].ewm(span=9, adjust=False).mean()
     
     columns_to_drop = ['Next_Close', 'Price_Change', 'Gains', 'Losses', 'Avg_Gain', 'Avg_Loss', 'RS',
                        'Body_of_Candle', 'Range_of_Candle', 'Prev_Open', 'Prev_Close','Lower_Shadow', 'Upper_Shadow']
     
     feature_df.drop(columns_to_drop, axis= 1, inplace= True)

     feature_df.dropna(inplace= True)

     feature_df = pd.get_dummies(data= feature_df, columns= ['Ticker_Name'], drop_first= True ,dtype= int)

     feature_df.set_index('Date', inplace= True)

     output_path = 'data/processed/engineered_market_data.csv'

     feature_df.to_csv(output_path, index= True)

     print(f"Success: Tech features & patterns saved at {output_path}")

if __name__ == '__main__':
     build_features()