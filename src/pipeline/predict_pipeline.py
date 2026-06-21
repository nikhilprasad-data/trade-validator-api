import joblib as jb

def predict_trade_signal(input_data):

     """Loads the pre-trained ML model and scaler to predict the trade action based on today's data."""

     model = jb.load('models/trade_gb_model.pkl')

     scale = jb.load('models/trade_scaler.pkl')

     featured_columns = jb.load('models/trade_features.pkl')

     input_df = input_data.reindex(columns = featured_columns, fill_value= 0)

     scaled_data = scale.transform(input_df)

     buy_prob = model.predict_proba(scaled_data)[0][1]

     is_tradeable = bool(buy_prob >= 0.55)

     action = 'BUY' if is_tradeable else 'NO TRADE / HOLD'

     return {
          'action'            : action,
          'is_tradeable'      : is_tradeable,
          'confidence_score'  : round(buy_prob * 100, 2)
     }