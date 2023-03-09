# -*- coding: utf-8 -*-
"""
Created on Wed Fev  2 22:44:22 2023

@author: Anderson Almeida
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas import Timedelta
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# defini o tamanho da janela e o horizonte de previsão do modelo
window_size = 15
horizon = 15

history = np.load('PETR3.npy')
history =  pd.DataFrame(history) #convertendo em dataframe
history = history.set_index('datetime') #fixando a coluna datetime como index

horario_limite = 14

# obter o índice do último dia
ultimo_dia = history.index.date[-1]
mask_ultimo_dia = (history.index.date == ultimo_dia) & (history.index.hour < horario_limite)

# criar uma máscara booleana para todos os dias, exceto o último dia
mask_outros_dias = history.index.date < ultimo_dia

# combinar as duas máscaras booleanas 
mask = mask_ultimo_dia | mask_outros_dias

# aplicando a mascara
history_cut = history.loc[mask]

# normalizar o historico 
scaled_data = (history['Close']/100).values


# criar os dados de treinamento e teste
X_train, y_train = [], []
for i in range(window_size, len(scaled_data) - horizon):
    X_train.append(scaled_data[i-window_size:i]) 
    y_train.append(scaled_data[i:i+horizon]) #
X_train, y_train = np.array(X_train), np.array(y_train)

# remodelar os dados de treinamento para o formato 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# criando o modelo
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(horizon))
model.compile(optimizer='adam', loss='mean_squared_error')

# treinar o modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)


# realizando a previsao

last_window = scaled_data[-window_size:].reshape((1, window_size, 1))
prediction = model.predict(last_window).reshape((horizon,))

# criar o índice de datas para as previsões
last_date = pd.to_datetime(history_cut.index[-1])
dates = pd.date_range(last_date + pd.Timedelta(minutes=1), periods=horizon, freq='T')

# criar o DataFrame com as previsões
predictions_df = pd.DataFrame({'Close': prediction}, index=dates)

# grafico

fig, ax = plt.subplots()
plt.plot((predictions_df*100)*1.005, marker='.', label='Prediction')

ind = ((history.index >= predictions_df.index[0] - Timedelta(minutes=horizon)) & 
        (history.index <= predictions_df.index[-1]))

ax.plot(history['Close'][ind], marker='.', label='Real Data')
ax.axvline((predictions_df.index)[0], color = 'r',alpha=0.5)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
plt.grid(alpha=0.5)

plt.legend()

























