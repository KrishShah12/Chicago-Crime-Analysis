import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten,Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

crimes=pd.read_csv("/Users/maxjacobs/Desktop/FALL 2023/REGRESSION/PROJECT/Crimes.csv")

# Community areas corresponding to Downtown Chicago
crimes_filtered = crimes[crimes['Community Area'].isin([5, 6, 7, 8, 21, 22, 24, 28, 29, 31, 32, 33, 34, 35])]

time_series_data = crimes_filtered.groupby('date').size().reset_index(name='Number_of_Crimes')
time_series_data = time_series_data.drop(time_series_data.index[-1])
time_series_data_list = time_series_data['Number_of_Crimes'].tolist()

#plt.figure(figsize=(10, 8))
#plt.plot(time_series_data['date'], time_series_data_list, color='black')
#plt.title('Time Series Data')
#plt.xlabel('Date')
#plt.ylabel('Number of Crimes')
#plt.show()

time_series_data.set_index('date', inplace=True)


split_index = int(0.9 * len(time_series_data))

train_data_unscaled, test_data_unscaled = time_series_data[:split_index], time_series_data[split_index:]

# Data scaling
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data_unscaled)
test_data = scaler.fit_transform(test_data_unscaled)

# Training data reshaping + lookback
lookback = 10
total_len=train_data.shape[0]
X_train=[]
y_train=[]
for i in range(lookback, total_len):
    X_train.append(train_data[i-lookback:i])
    y_train.append(train_data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Model creation
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train,y_train,epochs=30,batch_size=32)

# Test data reshaping
total = pd.concat([time_series_data,test_data_unscaled],axis=0)

total_new = total.values
test_arr = total_new[len(total_new) - len(test_data_unscaled)-lookback:]
test_arr = scaler.transform(test_arr.reshape(-1,1))

lookback = 10
X_test=[]
for i in range(lookback, test_arr.shape[0]):
    X_test.append(test_arr[i - lookback:i])

X_test = np.array(X_test)

# Generate predictions
y_test_pred = model.predict(X_test)

y_test_pred_actual = scaler.inverse_transform(y_test_pred)

test_pred = pd.DataFrame(y_test_pred_actual, columns = ['predicted'])
test_actual = test_data_unscaled

test_actual_reset = test_actual.reset_index()
full_test_actual = pd.merge(test_pred, test_actual_reset, left_index=True, right_index=True)

# Plot the predictions
full_test_actual.index = pd.to_datetime(full_test_actual['date'])

plt.figure(figsize=(10, 6))  
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.plot(full_test_actual.index, full_test_actual['Number_of_Crimes'], color='black', label='Observed (Test Set)')
plt.plot(full_test_actual.index, full_test_actual['predicted'], color='lime', label='Predicted (Test Set)')

plt.title('LSTM (RNN) Model Forecast') 
plt.xlabel('Date')  
plt.ylabel('Number of Crimes per Day')  

plt.legend()

# Calculate absolute percentage error
absolute_percentage_error = np.abs((full_test_actual['Number_of_Crimes'] - full_test_actual['predicted']) / full_test_actual['Number_of_Crimes'])

# Calculate MAPE
mape = np.mean(absolute_percentage_error) * 100
print("\n"+f'MAPE: {mape:.2f}%')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(full_test_actual['Number_of_Crimes'], full_test_actual['predicted']))
print(f'RMSE: {rmse:.2f}')

textstr = '\n'.join((
    f'MAPE: {mape:.2f}%',
    f'RMSE: {rmse:.2f}'))

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.85, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='bottom', bbox=props)


plt.show()









