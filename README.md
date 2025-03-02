# Crypto Trading Assistant

A Flask-based API that provides cryptocurrency trading insights using OpenAI's GPT-4 model and real-time market data from CoinGecko and our own market sentiment predictors of coffee prices and skirt length

Inially to aid our trading assistant in decision making, we wanted to predict a metric using promising but unconventional data - coffee price and skirt lengths. 
Coffee is a globally traded commodity, and its prices can reflect various economic and environmental factors. Recent trends have shown significant price increases due to supply shortages caused by adverse weather conditions in major coffee-producing regions like Brazil and Vietnam. These shortages have led to record-high bean prices, affecting both consumers and businesses. 
The Hemline Index is an economic theory suggesting that skirt lengths rise and fall with stock prices. Specifically, shorter skirts are believed to indicate bullish markets, while longer skirts suggest bearish trends. Although widely debated, this index provides an interesting perspective on consumer behavior and economic sentiment.
Our general trends from our analysis was coffee prices have a positive correlaation with market sentiment due to increased consumer spending and shorter skirt lengths were associated with bullish markets (negative correlation to market sentiment)

Using these models we predicted a value for market sentiment according to our trading assistant's query. This value is taken into consideration along with other key metrics and tools to generate a response.

## Features

- Real-time cryptocurrency price data
- AI-powered trading insights
- Historical price data analysis
- Support for multiple cryptocurrencies
- Interactive price charts

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yellowlagoo/hacktheburgh25.git
cd hacktheburgh25
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
python app.py
```

The server will start at `http://localhost:5050`

## API Endpoints

- `GET /`: Welcome message
- `GET /price/<crypto>`: Get current price for a cryptocurrency
- `GET /insights/<crypto>`: Get AI-powered trading insights with price metrics

## Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Dogecoin (DOGE)
- Ripple (XRP)
- Solana (SOL)
- And many more (using CoinGecko IDs)

## Credits 
OpenAI API
CoinGecko
Cursor 
pandas
numpy
matplotlib.pyplot
sklearn
csv
pathlib
random
urllib.parse
hashlib
numpy
PIL
datasets: 
Coffee Prices - 45 Year Historical Chart from Macrotrends 
University of Michigan: Consumer Sentiment (UMCSENT)


## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License 