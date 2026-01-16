import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import yfinance as yf
import joblib

# Carregando o Modelo Treinado (BayesianRidge)
model = joblib.load("/home/luan19/Documentos/Projeto_de_Pesquisa/Mercado Financeiro/TESTE/BayesianRidge_Bitcoin/1_BayesianRidge_BTC-USD.pkl")

# Dataset para testar o modelo treinado
data = yf.download("BTC-USD", start="2026-01-08", end="2026-01-09", interval="5m", progress=False, auto_adjust=True, multi_level_index=False)

# Realizando compia do modelo
btc = data.copy()

# Pegando a média móvel exponencial
btc["MME20"] = btc["Close"].ewm(span=20).mean()

# O modelo foi treinado com [High, Low, Open, Volume, MME20]
# para prever o preço Close

# Separando a variavel preditora
X = btc.drop(columns="Close")
y = btc["Close"]

# Número de intervalo de tempo
n = 500

# tempo
t = btc.index[:n]

# Figura animada
#plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15,8))

ax.set_xlim(t[0], t[-1])
ax.set_ylim(min(btc["Close"][:n]), max(btc["Close"][:n]))

ax_preco_close, = ax.plot([], [], label="Preço Real")
ax_preco_close_preditc, = ax.plot([], [], "--", label="Preço Predito")
ax_MME20, = ax.plot([], [], label="MME20")


def animate(frames):
    ax_preco_close.set_data(t[:frames],
                            btc["Close"][:n][:frames])
    
    ax_preco_close_preditc.set_data(t[:frames],
                            model.predict(X[:n])[:frames])
    
    ax_MME20.set_data(t[:frames],
                      btc["MME20"][:n][:frames])
    
    return ax_preco_close, ax_preco_close_preditc, ax_MME20

anim = FuncAnimation(fig=fig,
                     func=animate,
                     frames=len(t),
                    interval=165)


ax.set_title("Predição do Preço do Bitcoin (BTC-USD)")
ax.set_xlabel("Tempo")
ax.set_ylabel("Preço (USD)")
ax.legend()
plt.show()