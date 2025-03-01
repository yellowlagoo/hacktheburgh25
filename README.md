# Crypto Trading Assistant

A Flask-based API that provides cryptocurrency trading insights using OpenAI's GPT-4 model and real-time market data from CoinGecko.

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

## Contributing

Feel free to open issues or submit pull requests for any improvements.

## License

MIT License 