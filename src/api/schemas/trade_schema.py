from pydantic import BaseModel, Field, ConfigDict

class TradeValidator(BaseModel):

     """Validates the incoming ticker name requested by the frontend."""

     ticker_name: str

class TradeValidatorResponse(TradeValidator):

     """Formats the final prediction result, probability, and tradeable flag for the user."""
     
     is_tradeable : bool = Field(default= False)

     ticker_probability: float

     ticker_result: str = Field(default= 'NO TRADE / HOLD')

     model_config = ConfigDict(from_attributes= True)