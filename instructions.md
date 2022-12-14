This program uses a machine learning technique called LSTM (Long Short-Term Memory) to build a model for predicting the closing price of a given stock. It first loads in stock data for the specified company from yahoo finance for the period from 2012 to 2021. It then scales the closing price data so that it falls between 0 and 1, which is necessary for training the LSTM model.

Next, the code trains the LSTM model by using the scaled closing price data to predict the closing price for the next day. The LSTM model is made up of multiple layers, each with a specified number of neurons. The model is then compiled and fit to the training data using the Adam optimizer and mean squared error as the loss function.

Once the model is trained, it is saved to a file in the `models` folder. The code then loads in test data for the specified company from 2021 to the current date, and uses this data to make predictions on the closing price. It then plots the actual and predicted closing prices, and calculates the mean squared error of the predictions.


1. Import necessary `modules`, including `numpy`, `matplotlib`, `pandas`, `pandas_datareader`, `datetime`, `sys`, `MinMaxScaler`, `Sequential`,`Dense`, `Dropout`, and `LSTM` from `tensorflow.keras`.

2. Check if a company symbol and time period have been specified as command line arguments, and exit if not.

3. Load stock data for the specified company from yahoo finance for the specified time period.

4. Scale the closing price data so that it falls between 0 and 1.

5. Define the number of days to look in the past to predict the next day.

6. Create training data for the LSTM model by using the scaled closing price data to predict the closing price for the next day.

7. Convert the training data into numpy arrays.
Build the LSTM model by adding multiple layers with specified numbers of neurons.

8. Compile and fit the model to the training data using the Adam optimizer and mean squared error as the loss function.

9. Save the trained model to a file in the models folder.

10. Load in test data for the specified company for the current year.

11. Use the trained model to make predictions on the test data.

12. Plot the actual and predicted closing prices, and calculate the mean squared error of the predictions.