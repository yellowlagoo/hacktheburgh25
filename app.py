from flask import Flask, jsonify
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logging.debug("OpenAI API Key: " + "Present" if os.getenv("OPENAI_API_KEY") else "Missing")

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Crypto Trading Assistant!"

@app.route('/price/<crypto>')
def get_crypto_price(crypto):
    try:
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={crypto}&vs_currencies=usd'
        response = requests.get(url)
        data = response.json()
        if data:
            return jsonify(data)
        else:
            return jsonify({'error': 'Crypto not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/insights/<crypto>')
def get_crypto_insights(crypto):
    try:
        # Convert common symbols to CoinGecko IDs
        crypto_mapping = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'doge': 'dogecoin',
            'xrp': 'ripple',
            'sol': 'solana'
        }
        
        # Convert to lowercase and get mapped ID if exists
        crypto_id = crypto.lower()
        crypto_id = crypto_mapping.get(crypto_id, crypto_id)
        
        price_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}'
        historical_url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart?vs_currency=usd&days=30&interval=daily'
        
        price_response = requests.get(price_url)
        if price_response.status_code != 200:
            return jsonify({'error': f'Cryptocurrency {crypto} not found. Please use valid CoinGecko ID (e.g., "bitcoin", "ethereum")'}), 404
            
        price_data = price_response.json()
        
        historical_response = requests.get(historical_url)
        if historical_response.status_code != 200:
            return jsonify({'error': 'Could not fetch historical data'}), 404
            
        historical_data = historical_response.json()
        
        current_price = price_data['market_data']['current_price']['usd']
        market_cap = price_data['market_data']['market_cap']['usd']
        volume = price_data['market_data']['total_volume']['usd']
        price_change = price_data['market_data']['price_change_percentage_24h']
        price_change_7d = price_data['market_data']['price_change_percentage_7d']
        price_change_30d = price_data['market_data']['price_change_percentage_30d']
        high_24h = price_data['market_data']['high_24h']['usd']
        low_24h = price_data['market_data']['low_24h']['usd']
        
        historical_prices = historical_data['prices'][-7:]  # Get just the last 7 days for cleaner prompt
        
        prompt = (f"Provide trading insights for {crypto_id} with the following metrics:\n"
                f"- Current Price: ${current_price:,.2f}\n"
                f"- Market Cap: ${market_cap:,.2f}\n"
                f"- 24h Volume: ${volume:,.2f}\n"
                f"- 24h High: ${high_24h:,.2f}\n"
                f"- 24h Low: ${low_24h:,.2f}\n"
                f"- 24h Change: {price_change:.2f}%\n"
                f"- 7d Change: {price_change_7d:.2f}%\n"
                f"- 30d Change: {price_change_30d:.2f}%\n"
                f"Recent price trend: {historical_prices}\n"
                "Please provide a detailed analysis of the current market situation and potential future trends based on these metrics.")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cryptocurrency trading expert. Provide clear, concise analysis based on the given metrics."},
                {"role": "user", "content": prompt}
            ]
        )
        
        insights = response.choices[0].message.content
        return jsonify({
            'insights': insights,
            'current_price': current_price,
            'market_cap': market_cap,
            'volume': volume,
            'price_change_24h': price_change,
            'price_change_7d': price_change_7d,
            'price_change_30d': price_change_30d
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
