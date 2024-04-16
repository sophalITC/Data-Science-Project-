import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
# from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import io
import base64

class StockPricePredictor:
    def __init__(self, model_path='lstm_model.h5', data_path='data.csv'):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.sequence_length = 100

    def load_model(self):
        self.model = load_model(self.model_path)

    def preprocess_data(self, data):
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

        return scaled_data, scaler

    def generate_prediction_sequences(self, data):
        last_sequence = data[-self.sequence_length:]
        prediction_sequences = np.expand_dims(last_sequence, axis=0)
        return prediction_sequences

    def generate_predictions(self, prediction_sequences, scaler, num_predictions):
        predicted_prices = []

        for _ in range(num_predictions):
            next_price = self.model.predict(prediction_sequences)
            predicted_prices.append(next_price[0])

            prediction_sequences = np.concatenate(
                (prediction_sequences[:, 1:, :], next_price.reshape(1, 1, 1)), axis=1
            )

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

        return predicted_prices

    def generate_plot(self, actual_dates, actual_prices, predicted_dates, predicted_prices):
        plt.figure(figsize=(10, 6))
        plt.plot(actual_dates, actual_prices, label='Actual')
        plt.plot(predicted_dates, predicted_prices, label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'Actual vs. Predicted Prices in the next {len(predicted_dates)} days')
        plt.legend()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    def predict_stock_prices(self, prediction_days=10):
        data = pd.read_csv(self.data_path)
        scaled_data, scaler = self.preprocess_data(data)
        prediction_sequences = self.generate_prediction_sequences(scaled_data)
        predicted_prices = self.generate_predictions(prediction_sequences, scaler, prediction_days)

        last_date = data.index[-1]
        predicted_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predicted_prices))

        last_100_days_actual = data.iloc[-100:]
        actual_dates = last_100_days_actual.index
        actual_prices = last_100_days_actual['Price']

        plot = self.generate_plot(actual_dates, actual_prices, predicted_dates, predicted_prices)

        return predicted_prices, predicted_dates, plot

app = Flask(__name__)
# CORS(app)
stock_predictor = StockPricePredictor()

# Home route
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_days_input = request.form.get('predictionDays')
        if selected_days_input and selected_days_input.isdigit():
            selected_days = int(selected_days_input)
        else:
            # Invalid or empty input, set a default value
            selected_days = 10
    else:
        selected_days = 10

    stock_predictor.load_model()
    predicted_prices, predicted_dates, plot = stock_predictor.predict_stock_prices(selected_days)

    return render_template('index.html', prices=predicted_prices, dates=predicted_dates, plot=plot, selected_days=selected_days)

#=============================API========================
@app.route("/predict", methods=['GET'])
def predict():
    selected_days = request.args.get('days', default=10, type=int)
    stock_predictor.load_model()
    predicted_prices, predicted_dates, plot = stock_predictor.predict_stock_prices(selected_days)
    result = {
        'predicted_prices': predicted_prices.tolist(),
        'predicted_dates': [date.strftime('%Y-%m-%d') for date in predicted_dates],
        'plot': plot
    }
    return jsonify(result)

#=============================API========================



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
