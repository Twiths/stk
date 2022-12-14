import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM 


# StockPredictor class
class StockPredictor:
    def __init__(self, company, start, end, prediction_days):
        self.company = company
        self.start = start
        self.end = end
        self.prediction_days = prediction_days

    # get data from yahoo finance for the company for the given time period
    def get_data(self):
        data = web.DataReader(self.company, 'yahoo', self.start, self.end)
        return data

    # prepare data for neural networks by scaling down all the data we have to be between zeros and ones
    def prepare_data(self, data):
        global scaler 
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        return scaled_data

    # define empty lists for training data
    def train_data(self, scaled_data):
        x_train = []
        y_train = []

        # how many days we want to look in the past to predict the next day
        for x in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - self.prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        # convert lists into numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # return training data
        return x_train, y_train

    # build the model and train it
    def build_model(self, x_train, y_train):
        global model 
        model = Sequential()

        # specify the number of neurons in the LSTM layer
        # this is used to prevent overfitting of the model
        # return_sequences=True is used to return the full sequence of outputs
        # input_shape is the shape of the input data
        # units is the number of neurons in the layer
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # compile the model using adam optimizer and 
        # mean squared error loss function
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=32, epochs=1)

        # save the model to a file
        model.save('models/' + self.company + '.h5')

    # test the model
    def test_data(self, data):
        test_start = dt.datetime(2021, 1, 1)
        test_end = dt.datetime.now()

        # get the actual stock price for the test data
        test_data = web.DataReader(self.company, 'yahoo', test_start, test_end)
        actual_price = test_data['Close'].values

        # get the predicted stock price for the test data
        total_datasets = p.concat((data['Close'], test_data['Close']), axis=0)
        model_inputes = total_datasets[len(total_datasets) - len(actual_price) - self.prediction_days:].values
        model_inputes = model_inputes.reshape(-1, 1)
        model_inputes = scaler.transform(model_inputes)

        return model_inputes, actual_price

    # plot the predicted stock price against the actual stock price
    def predict(self, model_inputes, actual_price):
        x_test = []

        # get the predicted stock price for the test data
        for x in range(self.prediction_days, len(model_inputes)):
            x_test.append(model_inputes[x - self.prediction_days:x, 0])

        # convert lists into numpy arrays
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # get the predicted stock price
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # plot the predicted stock price against the actual stock price
        plt.plot(actual_price, color="green", label=f"Actual {self.company} Stock Price")
        plt.plot(predicted_prices, color="red", label=f"Predicted {self.company} Stock Price")
        plt.title(f"{self.company} Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel(f"{self.company} Share Price")
        plt.legend()
        plt.show()

        return predicted_prices

    # predict the next day's stock price
    def predict_next_day(self, model_inputes):
        real_data = []
        for x in range(self.prediction_days, len(model_inputes + 1)):
            real_data.append(model_inputes[x - self.prediction_days:x, 0])

        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)

        plt.plot(prediction, color="red", label=f"Predicted {self.company} Stock Price")
        plt.title(f"{self.company} Stock Price Prediction")
        plt.xlabel("Time")
        plt.ylabel(f"{self.company} Share Price")
        plt.legend()
        plt.show()

        return prediction

    # save the predicted stock price to a file
    def save_prediction(self, prediction):
        np.savetxt(f"predictions/{self.company}_prediction.csv", prediction, delimiter=",")
        print(f"Predicted price for next day for {self.company}: {prediction[0][0]}")

    # run the program
    def run(self):
        data = self.get_data()
        scaled_data = self.prepare_data(data)
        x_train, y_train = self.train_data(scaled_data)
        self.build_model(x_train, y_train)
        model_inputes, actual_price = self.test_data(data)
        predicted_prices = self.predict(model_inputes, actual_price)
        print(f"Predicted price for next day for {self.company}: {predicted_prices[0][0]}")
        prediction = self.predict_next_day(model_inputes)
        self.save_prediction(prediction)


if __name__ == '__main__':
    # get the company name, start date, end date and number of days to predict
    company = input("Enter the company name: ")
    start = input("Enter the start date in YYYY-MM-DD format: ")
    end = input("Enter the end date in YYYY-MM-DD format: ")
    prediction_days = int(input("Enter the number of days to predict: "))

    start = dt.datetime.strptime(start, '%Y-%m-%d')
    end = dt.datetime.strptime(end, '%Y-%m-%d')

    stock_predictor = StockPredictor(company, start, end, prediction_days)
    stock_predictor.run()