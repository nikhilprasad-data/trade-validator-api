from fastapi import APIRouter, HTTPException, status
from src.api.schemas.trade_schema import TradeValidator, TradeValidatorResponse
import yfinance as yf
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import predict_trade_signal



trade = APIRouter() 

@trade.post('/trade-validator', response_model= TradeValidatorResponse, status_code= status.HTTP_200_OK)
def trade_validator(ticker_name: TradeValidator):

     """Fetches 1-year live data, calculates technical indicators, and returns the ML model's prediction."""
     
     user_ticker_name = ticker_name.ticker_name

     stock_data = yf.Ticker(user_ticker_name)
     df = stock_data.history(period='1mo', interval='5m')

     df.columns = df.columns.astype(str)

     if df.empty:
          raise HTTPException(status_code= 404, detail= "Invalid Ticker Name or Data not found!")


     feature_df = df.copy()

     feature_df['Price_Change'] = feature_df['Close'].diff()

     feature_df['Gains'] = np.where(feature_df['Price_Change'] < 0 , 0, feature_df['Price_Change'])

     feature_df['Losses'] = np.where(feature_df['Price_Change'] > 0, 0, np.abs(feature_df['Price_Change']))

     feature_df['Avg_Gain'] = feature_df['Gains'].transform(lambda x: x.ewm(alpha=1/14, adjust=False).mean())

     feature_df['Avg_Loss'] = feature_df['Losses'].transform(lambda x: x.ewm(alpha= 1/14, adjust= False).mean())

     feature_df['RS'] = feature_df['Avg_Gain'] / feature_df['Avg_Loss']

     feature_df['RSI'] = 100 - (100 / (1+ feature_df['RS']))

     feature_df['Body_of_Candle'] = np.abs(feature_df['Open'] - feature_df['Close'])

     feature_df['Range_of_Candle'] = feature_df['High'] - feature_df['Low']

     feature_df['Doji'] = np.where(feature_df['Body_of_Candle'] <= ((feature_df['Range_of_Candle']) * 0.1), 1, 0)

     feature_df['Prev_Open'] = feature_df['Open'].shift(1)

     feature_df['Prev_Close'] = feature_df['Close'].shift(1)

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
     
     columns_to_drop = ['Price_Change', 'Gains', 'Losses', 'Avg_Gain', 'Avg_Loss', 'RS',
                        'Body_of_Candle', 'Range_of_Candle', 'Prev_Open', 'Prev_Close','Lower_Shadow', 'Upper_Shadow']
     
     feature_df.drop(columns_to_drop, axis= 1, inplace= True)

     feature_df.dropna(inplace= True)

     final_live_data = feature_df.tail(1)

     model_result = predict_trade_signal(final_live_data)

     result = {
          'ticker_name'            : user_ticker_name,
          'ticker_probability'     : model_result['confidence_score'],
          'ticker_result'          : model_result['action'],
          'is_tradeable'           : model_result['is_tradeable']
     }

     return result