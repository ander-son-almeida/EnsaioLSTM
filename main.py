import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from pandas import Timedelta
import numpy as np
from yahooquery import Ticker
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy.polynomial.polynomial as poly


'''
 ^BVSP  ^DJI  ^SSMI  BNO  EMB  000001.SS  BRL=X  ^VIX  B3SA3.SA  VALE3.SA 
 ITUB4.SA  PETR4.SA  ABEV3.SA  BBDC4.SA  ELET3.SA  PETR3.SA  BBAS3.SA  ITSA4.SA  RENT3.SA  
 WEGE3.SA  BBSE3.SA  BPAC11.SA  EQTL3.SA  GGBR4.SA HAPV3.SA  JBSS3.SA  LREN3.SA  PRIO3.SA  
 RADL3.SA  RDOR3.SA  RAIL3.SA  SUZB3.SA B3SA3.SA BRL=X B3SA3.SA VALE3.SA ITUB4.SA PETR4.SA 
 ABEV3.SA BBDC4.SA ELET3.SA PETR3.SA BBAS3.SA ITSA4.SA RENT3.SA WEGE3.SA BBSE3.SA BPAC11.SA 
 EQTL3.SA GGBR4.SA HAPV3.SA JBSS3.SA LREN3.SA PRIO3.SA RADL3.SA RDOR3.SA RAIL3.SA SUZB3.SA'
'''

symbol = 'PETR3F.SA'
start_date = '2023-02-28' 
# end_date = '2023-03-02'
media_movel = 5

# defini o tamanho da janela e o horizonte de previsão do modelo
window_size = 15
horizon = 15
constante_norm = 100 #resolvi normalizar tudo com 100

# backtest
'''
O API fornece no máximo dados de 5 dias com intervalos de 1 minuto. Para realizar 
o backtest vou colocar um horario limite de dados para o modelo treinar. Por exemplo:
em um historico de 5 dias de treinamento fixo um horario limite até às 14 horas e tento
prever 15 minutos e faço o backtest com os dados reais.
'''

horario_limite = 14 #horas do ultimo dia

###############################################################################
# Historico
try:
    history = Ticker(symbol).history(start = start_date, interval='1m')
    history.reset_index(level=["symbol"], inplace=True)
    
    # media movel
    history['mm5m'] = history['close'].rolling(media_movel).mean()
    
    # removendo dados nan e coluna symbol, pois nao precisamos
    history = history.dropna()
    history = history.drop('symbol', axis=1)
except:
    raise Exception("Atualize o parametro 'start_date'")

###############################################################################
# corte para o backtest

# obter o índice do último dia
ultimo_dia = history.index.date[-1]
mask_ultimo_dia = (history.index.date == ultimo_dia) & (history.index.hour < horario_limite)

# criar uma máscara booleana para todos os dias, exceto o último dia
mask_outros_dias = history.index.date < ultimo_dia

# combinar as duas máscaras booleanas 
mask = mask_ultimo_dia | mask_outros_dias

# aplicando a mascara
history_cut = history.loc[mask]

###############################################################################
# normalizar o historico 
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(history_cut.values)

close_prices = history_cut['close']/constante_norm
scaled_data = close_prices.values

# criar os dados de treinamento e teste
X_train, y_train = [], []
for i in range(window_size, len(scaled_data) - horizon):
    X_train.append(scaled_data[i-window_size:i]) 
    y_train.append(scaled_data[i:i+horizon]) #
X_train, y_train = np.array(X_train), np.array(y_train)

'''
X_train e y_train são divididos dessa forma para preparar 
os dados de treinamento para o modelo LSTM.

A primeira parte, scaled_data[i-window_size:i], corresponde a um conjunto de 
dados de entrada com um tamanho definido pelo parâmetro window_size. Cada janela 
desse conjunto é usada como entrada para a rede LSTM em cada passo de tempo.

A segunda parte, scaled_data[i:i+horizon], corresponde ao valor esperado de 
saída para a rede LSTM. Esse valor é o próximo valor que a série temporal deve 
assumir após a janela de entrada
'''

# remodelar os dados de treinamento para o formato [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

###############################################################################
# criar o modelo
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(horizon))
model.compile(optimizer='adam', loss='mean_squared_error')

# treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=32)

################################################################################ prever os próximos X minutos com base no comportamentos dos X minutos anteriores
last_window = scaled_data[-window_size:].reshape((1, window_size, 1))
prediction = model.predict(last_window).reshape((horizon,))

# criar o índice de datas para as previsões
last_date = pd.to_datetime(history_cut.index[-1])
dates = pd.date_range(last_date + pd.Timedelta(minutes=1), periods=horizon, freq='T')

# criar o DataFrame com as previsões
predictions_df = pd.DataFrame({'close': prediction}, index=dates)

###############################################################################
# grafico

#ajustando retas para descobrir os sentidos (compra e venda)

ind = ((history.index >= predictions_df.index[0]) & 
       (history.index <= predictions_df.index[-1]))

xplot = np.arange(0, predictions_df['close'].size, 1)

#reta real
coef_reta_real = poly.polyfit(xplot, history['close'][ind], 1) 
f_reta_real = poly.Polynomial(coef_reta_real)
reta_real = f_reta_real(xplot) 

#reta predicao
coef_reta_pred = poly.polyfit(xplot, (predictions_df['close'].values)*constante_norm, 1) 
f_reta_pred = poly.Polynomial(coef_reta_pred)
reta_pred = f_reta_pred(xplot)


plt.figure()
plt.plot(predictions_df*constante_norm, marker='.', label='prediction')

ind2 = ((history.index >= predictions_df.index[0] - Timedelta(minutes=horizon)) & 
        (history.index <= predictions_df.index[-1]))

plt.plot(history['close'][ind2], marker='.', label='real data')
plt.plot(predictions_df.index, reta_real, alpha=0.5)

plt.axvline((predictions_df.index)[0], color = 'r',alpha=0.5)
plt.plot(predictions_df.index, reta_pred, alpha=0.5)

plt.legend()














































        