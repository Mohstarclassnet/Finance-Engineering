딥러닝
pip install tensorflow
pip install --upgrade numpy
import tensorflow as tf
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

yf.pdr_override()

# 시작일, 종료일, '097950.KS'
start_date = datetime(2018, 11, 3)
end_date = datetime(2021, 1, 26)

df = pdr.get_data_yahoo('097950.KS', start_date, end_date)


print(df)


# min_max_scaler ( min : 0 , max : 1 ) I. data - min, II. max - min, I / II
def minMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)


# 'Open', 'High', 'Low', 'Volume', 'Close' 다섯개의 값을 사용하는 인풋
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = minMaxScaler(dfx)

print(dfx)
# 'Close' 결과값은 종가 한 개
dfy = dfx[['Close']]

# x, y -> tolist()
x = dfx.values.tolist()
y = dfy.values.tolist()

# data set 구성 10일의 데이터를 이용하여 11일째를 예측
data_x = []
data_y = []
window_size = 10
for i in range(len(y) - window_size):
    _x = x[i: i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

# training set, test_set 구성 (training_set 는 전체의 70% )
train_size = int(len(data_y) * 0.7)
train_x = np.array(data_x[0:train_size])
train_y = np.array(data_y[0:train_size])


test_size = len(data_y) - train_size
test_x = np.array(data_x[train_size:len(data_x)])
test_y = np.array(data_y[train_size:len(data_y)])

# 모델 구성 및 학습----------------------------------------------------------------------------------
# units=10, LSTM, activation='relu', input_shape = (10, 5), Dropout=0.1
model = Sequential()
model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(window_size, 5)))
model.add(Dropout(0.1))
model.add(LSTM(units=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()
# 학습횟수(epoch) = 60, 한번에 제공되는 데이터의 갯수(batch_size) = 30
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_x, train_y, epochs=60, batch_size=30)
pred_y = model.predict(test_x)
# 예측치와 실제 종가 비교

plt.figure(figsize=(10, 5))
plt.plot(test_y, color='black', label='real price')
plt.plot(pred_y, color='green', label='predicted price')
plt.title('final plot')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend(loc='best')
plt.show()








df= pd.DataFrame()
sr_1 = pd.Series([2,3,1,2,3,5,6])
window = 4

df['adj_close'] = sr_1
sr_rolling = df['adj_close'].rolling(window, min_periods = 1).min()

df['sr_new'] = sr_rolling
df = df.astype('int32')

print(df.sr_new[0]+ df.sr_new[3])



df = pd.DataFrame()

sr_1 = pd.Series([1,2,3])
sr_2 = pd.Series([3,4,6])
df['sr_1'] = sr_1
df['sr_2'] = sr_2

new_df = ((df-df.shift(1))/df.shift(1))
print(new_df.round(1))


c = [1,2,2,2,3,4,7]
loc = bisect.bisect(c,3)
print(loc)
c.insert(loc,3)
for i in c:
    i+=2*i
   
print(i)

words=['apple','bat','bar','atom','book']

by_letter ={}

for word in words:
    letter = word[0]
    if letter not in by_letter:
        by_letter[letter] = [word]
    else: by_letter[letter].append(word)
print(by_letter)





