from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from challenge.model import DelayModel


class PredictRequest(BaseModel):
    flights: object


app = FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(data: PredictRequest) -> dict:
    """
    Predicts the delay of the flights in the data or raises an exception if the value in the column MES, TIPOVUELO or OPERA is incorrect.
    
    Body Parameters
    ----------
    data : PredictRequest
        Data to predict.
        
    Returns
    -------
    dict
        Predicted delay.
        
    Raises
    ------
    HTTPException
        If the column MES, TIPOVUELO or OPERA is not found.
    """
    
    response = DelayModel().check_response(data)
    
    if(isinstance(response, tuple)):
        if(response[0] == 'MES'):
            raise HTTPException(
                status_code=400, detail=f"Value in column MES is incorrect")
        elif(response[0] == 'TIPOVUELO'):
            raise HTTPException(
                status_code=400, detail=f"Value in column TIPOVUELO is incorrect")
        elif(response[0] == 'OPERA'):
            raise HTTPException(
                status_code=400, detail=f"Value in column OPERA is incorrect")

    response_ = {
        "predict": response
    }
    
    return response_
