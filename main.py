from fastapi import FastAPI
from fastapi.middleware import cors
from src.api.routers import trade


app = FastAPI()

app.add_middleware(
     cors.CORSMiddleware,
     allow_origins  = ["*"],
     allow_credentials= True,
     allow_methods    = ["*"],
     allow_headers    = ["*"]
)

app.include_router(trade, prefix= '/api', tags= ['Trade-Validator'])