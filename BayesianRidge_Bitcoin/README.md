# 📈 Predição do Preço do Bitcoin com Machine Learning (Bayesian Ridge)

Este projeto aplica **Machine Learning** para prever o preço de fechamento (*Close*)
do Bitcoin (BTC-USD) utilizando dados históricos de alta frequência (5 minutos),
com foco em **visualização animada** e **inferência de modelos treinados**.

O modelo utilizado é o **Bayesian Ridge Regression**, escolhido por sua robustez
em cenários com ruído e correlação entre variáveis explicativas.

---

## 🔧 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yFinance
- Joblib

---

## 📊 Dados Utilizados

- Ativo: **Bitcoin (BTC-USD)**
- Fonte: Yahoo Finance (via `yfinance`)
- Intervalo: 5 minutos

---

## 🧠 Features do Modelo

O modelo foi treinado utilizando as seguintes variáveis preditoras:

- Open
- High
- Low
- Volume
- Média Móvel Exponencial de 20 períodos (MME20)

🎯 **Variável alvo:**  
- Preço de fechamento (*Close*)

---

## 📈 Visualização

O projeto inclui uma **animação em tempo real** que compara:

- Preço real do Bitcoin
- Preço predito pelo modelo
- MME20

Essa visualização tem como objetivo facilitar a análise do comportamento do modelo
ao longo do tempo.

---

## ▶️ Como Executar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

⚠️  Aviso Importante

Este projeto tem finalidade acadêmica e educacional.
Não constitui recomendação de investimento ou estratégia de trading.

Mercados financeiros envolvem risco.
 
![Predição do Bitcoin](images/btc_prediction.gif)
