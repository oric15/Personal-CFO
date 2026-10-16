🚦 Project: Personal CFO - Signal Builder (ENG)
📖 Description
A sentiment-driven trading signal dashboard. It uses FinBERT (NLP) to analyze Yahoo Finance news and 
combines it with price trends to generate BUY/SELL/WATCH signals for NVDA, AIR.PA, and IREN.

🚀 Setup & Installation
Requirements: Python

Install Dependencies:
* pip install streamlit pandas plotly yfinance transformers torch

💻 How to Run
Always run from the project root:

python3 -m streamlit run /app.py
Click the "Refresh Data" button on the dashboard to trigger the data pipeline (Yahoo Finance → FinBERT → Signal Logic).

🚦 Projeto: Personal CFO - Signal Builder (PT)
📖 Descrição
Um dashboard de sinais de trading baseado em análise de sentimento. Utiliza FinBERT (NLP) para analisar notícias 
do Yahoo Finance e combina com tendências de preço para gerar sinais de COMPRA/VENDA/AGUARDAR para NVDA, AIR.PA e IREN.

🚀 Configuração e Instalação
Requisitos: Python 

Instalar Dependências:
* pip install streamlit pandas plotly yfinance transformers torch

💻 Como Executar
Sempre execute a partir da raiz do projeto:
python3 -m streamlit run app.py
Clique no botão "Refresh Data" no dashboard para iniciar o processamento de dados 
(Yahoo Finance → FinBERT → Lógica de Sinais).

> ⚠️ **Disclaimer (ENG)**
> This project is for educational and portfolio purposes only. It does not
> constitute financial, investment, or trading advice. The BUY/SELL/WATCH
> signals are the output of an experimental model, have not been validated
> for real-world use, and past performance does not guarantee future results.
> Do not make investment decisions based on this tool. Use at your own risk.

> ⚠️ **Aviso (PT)**
> Este projeto tem fins exclusivamente educativos e de portefólio. Não
> constitui aconselhamento financeiro, de investimento ou de negociação. Os
> sinais de COMPRA/VENDA/AGUARDAR são o resultado de um modelo experimental,
> não foram validados para uso real, e o desempenho passado não garante
> resultados futuros. Não tomes decisões de investimento com base nesta
> ferramenta. Utilização por tua conta e risco.
