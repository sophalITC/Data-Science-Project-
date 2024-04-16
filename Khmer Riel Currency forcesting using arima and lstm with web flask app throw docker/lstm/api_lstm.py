from fastapi import FastAPI, Query
from app_lstm import *

app = FastAPI()
stock_predictor = StockPricePredictor()

@app.get("/predict")
def predict(days: int = Query(10, description="The number of days to predict")):
    stock_predictor.load_model()
    predicted_prices, predicted_dates, plot = stock_predictor.predict_stock_prices(days)
    result = {
        'predicted_dates': [date.strftime('%Y-%m-%d') for date in predicted_dates],
        'predicted_prices': predicted_prices.tolist(),
        'plot': plot
    }
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6969)
