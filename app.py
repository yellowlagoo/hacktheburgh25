from flask import Flask, jsonify, request
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from functools import lru_cache
from datetime import datetime, timedelta

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

app = Flask(__name__)

# Cache for storing API responses
price_cache = {}
CACHE_DURATION = 30  # seconds

# Extended crypto mapping
crypto_mapping = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'bnb': 'binancecoin',
    'sol': 'solana',
    'xrp': 'ripple',
    'ada': 'cardano',
    'doge': 'dogecoin',
    'dot': 'polkadot',
    'matic': 'matic-network',
    'link': 'chainlink',
    'uni': 'uniswap',
    'avax': 'avalanche-2',
    'atom': 'cosmos',
    'ltc': 'litecoin',
    'near': 'near'
}

# Default coins to analyze if none specified
DEFAULT_COINS = ['bitcoin', 'ethereum', 'binancecoin']
MAX_COINS = 5  # Maximum number of coins to analyze at once

def make_request(url):
    current_time = datetime.now()
    
    # Check cache first
    if url in price_cache:
        cached_data, cache_time = price_cache[url]
        if current_time - cache_time < timedelta(seconds=CACHE_DURATION):
            print(f"Using cached data for: {url}")
            return cached_data
    
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    }
    
    print(f"\nMaking request to: {url}")
    response = requests.get(url, headers=headers)
    
    print(f"Response Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response Text: {response.text}")
        
        if response.status_code == 429:
            # Get retry-after header or default to 60 seconds
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            # Retry the request once
            response = requests.get(url, headers=headers)
    
    # Cache successful responses
    if response.status_code == 200:
        price_cache[url] = (response, current_time)
    
    return response

@app.route('/')
def home():
    return "Welcome to the Crypto Trading Assistant!"

@app.route('/price/<crypto>')
def get_crypto_price(crypto):
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
        
        print(f"\nFetching price for: {crypto_id}")
        
        # Get detailed coin data
        url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false'
        response = make_request(url)
        
        if response.status_code == 429:
            error_msg = 'Rate limit exceeded. Please try again in a minute.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 429
            
        if response.status_code != 200:
            error_msg = f'Cryptocurrency {crypto_id} not found'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 404
            
        data = response.json()
        market_data = data.get('market_data', {})
        
        result = {
            crypto_id: {
                'usd': market_data.get('current_price', {}).get('usd', 0),
                'usd_market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'usd_24h_vol': market_data.get('total_volume', {}).get('usd', 0),
                'usd_24h_change': market_data.get('price_change_percentage_24h', 0),
                'usd_1h_change': market_data.get('price_change_percentage_1h', {}).get('usd', 0),
                'usd_24h_high': market_data.get('high_24h', {}).get('usd', 0),
                'usd_24h_low': market_data.get('low_24h', {}).get('usd', 0)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_msg = f"Error in get_crypto_price: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

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
        
        print(f"\nFetching insights for: {crypto_id}")
        
        # Get detailed coin data
        url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}?localization=false&tickers=false&market_data=true&community_data=false&developer_data=false'
        response = make_request(url)
        
        if response.status_code == 429:
            error_msg = 'Rate limit exceeded. Please try again in a minute.'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 429
            
        if response.status_code != 200:
            error_msg = f'Cryptocurrency {crypto_id} not found'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 404
            
        data = response.json()
        market_data = data.get('market_data', {})
        
        metrics = {
            'current_price': market_data.get('current_price', {}).get('usd', 0),
            'market_cap': market_data.get('market_cap', {}).get('usd', 0),
            'volume': market_data.get('total_volume', {}).get('usd', 0),
            'price_change_24h': market_data.get('price_change_percentage_24h', 0),
            'price_change_7d': market_data.get('price_change_percentage_7d', 0),
            'price_change_30d': market_data.get('price_change_percentage_30d', 0),
            'high_24h': market_data.get('high_24h', {}).get('usd', 0),
            'low_24h': market_data.get('low_24h', {}).get('usd', 0),
            'ath': market_data.get('ath', {}).get('usd', 0),
            'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0)
        }
        
        print(f"Metrics data: {metrics}")
        
        prompt = (f"Analyze {crypto_id} based on these real-time metrics:\n"
                f"- Current Price: ${metrics['current_price']:,.2f}\n"
                f"- Market Cap: ${metrics['market_cap']:,.2f}\n"
                f"- 24h Volume: ${metrics['volume']:,.2f}\n"
                f"- 24h High: ${metrics['high_24h']:,.2f}\n"
                f"- 24h Low: ${metrics['low_24h']:,.2f}\n"
                f"- 24h Change: {metrics['price_change_24h']:.2f}%\n"
                f"- 7d Change: {metrics['price_change_7d']:.2f}%\n"
                f"- 30d Change: {metrics['price_change_30d']:.2f}%\n"
                f"- Distance from ATH: {metrics['ath_change_percentage']:.2f}%\n\n"
                "Provide a detailed, data-driven analysis. Include specific numbers and percentages. Focus on current market conditions and notable changes.")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cryptocurrency analyst. Provide detailed, data-driven analysis based on real-time market data. Always reference specific numbers and metrics in your analysis. Never make up data or use placeholder values."},
                {"role": "user", "content": prompt}
            ]
        )
        
        insights = response.choices[0].message.content
        return jsonify({
            'insights': insights,
            'metrics': metrics
        })
        
    except Exception as e:
        error_msg = f"Error in get_crypto_insights: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

def format_market_analysis(metrics, analysis_content):
    # Calculate supply metrics
    circulating_ratio = (metrics['circulating_supply'] / metrics['max_supply'] * 100) if metrics.get('max_supply') else 0
    
    return f"""### Market Overview
{metrics['symbol']} is currently trading at **${metrics['current_price']:,.2f}** with a market cap of **${metrics['market_cap']/1e9:.2f}B**
The sentiment is {get_sentiment_description(metrics.get('price_change_24h', 0))} with a **{metrics.get('price_change_24h', 0):+.2f}%** change in 24h

### Price Performance
**Short-term Changes**
• 1h: {metrics.get('price_change_1h', 0):+.2f}%
• 24h: {metrics.get('price_change_24h', 0):+.2f}%
• 7d: {metrics.get('price_change_7d', 0):+.2f}%
• 14d: {metrics.get('price_change_14d', 0):+.2f}%
• 30d: {metrics.get('price_change_30d', 0):+.2f}%
• 1y: {metrics.get('price_change_1y', 0):+.2f}%

### Market Activity
**Trading Metrics**
• 24h Volume: ${metrics['volume']/1e9:.2f}B
• Volume/Market Cap: {(metrics['volume']/metrics['market_cap']*100):.2f}%
• Market Cap Change 24h: {metrics.get('market_cap_change_24h', 0):+.2f}%

**Price Range (24h)**
• High: ${metrics['high_24h']:,.2f}
• Low: ${metrics['low_24h']:,.2f}
• Range: ${(metrics['high_24h'] - metrics['low_24h']):,.2f} ({((metrics['high_24h'] - metrics['low_24h'])/metrics['low_24h']*100):.1f}% spread)

### Supply Analysis
• Circulating Supply: {metrics['circulating_supply']:,.0f}
• Maximum Supply: {metrics.get('max_supply', 0):,.0f}
• Supply Ratio: {circulating_ratio:.1f}% of max supply in circulation
• Fully Diluted Valuation: ${metrics.get('fully_diluted_valuation', 0)/1e9:.2f}B

### Technical Analysis
{analysis_content}

### Market Indicators
**Risk Assessment**
• Risk Level: {get_risk_level(metrics)}
• Market Phase: {get_market_phase(metrics)}
• Volatility: {get_volatility_measure(metrics)}
• Market Rank: #{metrics['market_cap_rank']}

### Derivatives Market Analysis
**Futures Market**
• Funding Rate: {get_funding_rate(metrics)}
• Open Interest: ${get_open_interest(metrics)/1e9:.2f}B
• Long/Short Ratio: {get_long_short_ratio(metrics):.2f}

**Options Market**
• Put/Call Ratio: {get_put_call_ratio(metrics):.2f}
• Implied Volatility: {get_implied_volatility(metrics)}%
• Max Pain Price: ${get_max_pain_price(metrics):,.2f}

### Inter-Market Correlations
**Crypto Correlations**
• BTC Correlation: {get_btc_correlation(metrics):.2f}
• ETH Correlation: {get_eth_correlation(metrics):.2f}

**Traditional Markets**
• S&P 500 Correlation: {get_sp500_correlation(metrics):.2f}
• Gold Correlation: {get_gold_correlation(metrics):.2f}
• DXY Correlation: {get_dxy_correlation(metrics):.2f}

### Trading Recommendations
{generate_enhanced_trading_recommendations(metrics)}"""

def get_funding_rate(metrics):
    # Simulated funding rate based on price changes
    return f"{(metrics.get('price_change_24h', 0) * 0.01):+.3f}%"

def get_open_interest(metrics):
    # Simulated open interest based on market cap
    return metrics.get('market_cap', 0) * 0.1

def get_long_short_ratio(metrics):
    # Simulated long/short ratio based on recent price action
    if metrics.get('price_change_24h', 0) > 0:
        return 1.2 + (metrics.get('price_change_24h', 0) * 0.02)
    return 0.8 + (metrics.get('price_change_24h', 0) * 0.02)

def get_put_call_ratio(metrics):
    # Simulated put/call ratio based on market sentiment
    sentiment = metrics.get('price_change_24h', 0)
    if sentiment > 0:
        return max(0.8 - (sentiment * 0.01), 0.5)
    return min(1.2 - (sentiment * 0.01), 2.0)

def get_implied_volatility(metrics):
    # Simulated implied volatility based on recent price changes
    return abs(metrics.get('price_change_24h', 0)) * 2 + 30

def get_max_pain_price(metrics):
    # Simulated max pain price near current price
    current_price = metrics.get('current_price', 0)
    change = metrics.get('price_change_24h', 0)
    return current_price * (1 + (change * 0.001))

def get_btc_correlation(metrics):
    # Simulated BTC correlation
    return 0.8 + (metrics.get('price_change_24h', 0) * 0.01)

def get_eth_correlation(metrics):
    # Simulated ETH correlation
    return 0.7 + (metrics.get('price_change_24h', 0) * 0.01)

def get_sp500_correlation(metrics):
    # Simulated S&P 500 correlation
    return 0.4 + (metrics.get('price_change_24h', 0) * 0.005)

def get_gold_correlation(metrics):
    # Simulated gold correlation
    return -0.2 + (metrics.get('price_change_24h', 0) * 0.002)

def get_dxy_correlation(metrics):
    # Simulated DXY (US Dollar Index) correlation
    return -0.3 + (metrics.get('price_change_24h', 0) * 0.002)

def get_sentiment_description(change):
    if change > 5: return "strongly bullish"
    if change > 2: return "moderately bullish"
    if change > 0: return "slightly bullish"
    if change > -2: return "slightly bearish"
    if change > -5: return "moderately bearish"
    return "strongly bearish"

def get_risk_level(metrics):
    volatility = abs(metrics.get('price_change_24h', 0))
    if volatility > 10: return "High"
    if volatility > 5: return "Moderate"
    return "Low"

def get_market_phase(metrics):
    price_change_24h = metrics.get('price_change_24h', 0)
    price_change_7d = metrics.get('price_change_7d', 0)
    
    if price_change_24h > 5 and price_change_7d > 0:
        return "Expansion"
    if price_change_24h < -5 and price_change_7d < 0:
        return "Contraction"
    return "Consolidation"

def get_volatility_measure(metrics):
    range_percent = (metrics['high_24h'] - metrics['low_24h']) / metrics['low_24h'] * 100
    if range_percent > 10: return "High"
    if range_percent > 5: return "Moderate"
    return "Low"

def generate_enhanced_trading_recommendations(metrics):
    recommendations = []
    
    # Volume analysis
    volume_to_mcap = metrics['volume'] / metrics['market_cap']
    if volume_to_mcap > 0.1:
        recommendations.append("**High trading volume** ({:.1f}% of market cap) indicates strong market interest - consider setting tight stop losses".format(volume_to_mcap * 100))
    
    # Price trend analysis
    if all(metrics[period] > 0 for period in ['price_change_1h', 'price_change_24h', 'price_change_7d']):
        recommendations.append("**Strong upward trend** across multiple timeframes (1h/24h/7d all positive) - momentum is building")
    elif all(metrics[period] < 0 for period in ['price_change_1h', 'price_change_24h', 'price_change_7d']):
        recommendations.append("**Strong downward trend** detected across timeframes - consider waiting for reversal signals")
    
    # Supply analysis
    if metrics.get('max_supply') and metrics.get('circulating_supply'):
        if metrics['circulating_supply'] / metrics['max_supply'] > 0.9:
            recommendations.append("**Limited supply remaining** - only {:.1f}% of maximum supply left to be mined".format(
                (1 - metrics['circulating_supply'] / metrics['max_supply']) * 100))
    
    # Volatility analysis
    range_percent = (metrics['high_24h'] - metrics['low_24h']) / metrics['low_24h'] * 100
    if range_percent > 5:
        recommendations.append(f"**High price volatility** ({range_percent:.1f}% 24h range) - consider using limit orders and wider stops")
    
    # ATH analysis
    if abs(metrics['ath_change_percentage']) < 5:
        recommendations.append("**Near all-time high** - consider taking partial profits or using trailing stops")
    elif abs(metrics['ath_change_percentage']) > 20:
        recommendations.append(f"**Significant discount from ATH** ({metrics['ath_change_percentage']:.1f}%) - potential value opportunity if market sentiment improves")
    
    return "\n".join(f"• {rec}" for rec in recommendations)

def extract_coin_symbols(user_input):
    """Extract cryptocurrency symbols from user input."""
    query_lower = user_input.lower()
    requested_coins = []
    
    # Look for specific coin mentions
    for symbol, coin_id in crypto_mapping.items():
        if symbol in query_lower or coin_id in query_lower:
            requested_coins.append(coin_id)
    
    # If no specific coins mentioned, use default set
    if not requested_coins:
        return DEFAULT_COINS
    
    # Limit to MAX_COINS
    return requested_coins[:MAX_COINS]

@app.route('/command', methods=['POST'])
def process_command():
    try:
        print("\n=== Received command request ===")
        print("Request headers:", request.headers)
        data = request.get_json()
        print("Request data:", data)
        
        if not data:
            print("Error: No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        user_input = data.get('command', '')
        if not user_input:
            print("Error: Empty command received")
            return jsonify({'error': 'Please provide a command'}), 400
            
        # Extract requested coins from user input
        crypto_ids = extract_coin_symbols(user_input)
        print(f"Analyzing cryptocurrencies: {crypto_ids}")
        
        all_metrics = {}
        rate_limited = False
        
        for crypto_id in crypto_ids:
            print(f"\nProcessing analysis for: {crypto_id}")
            
            # Get real-time crypto data
            url = f'https://api.coingecko.com/api/v3/coins/{crypto_id}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true'
            print(f"Fetching data from: {url}")
            response = make_request(url)
            
            if response.status_code == 429:
                print("Error: Rate limit exceeded")
                rate_limited = True
                break
                
            if response.status_code != 200:
                print(f"Error: Bad response from CoinGecko API: {response.status_code}")
                continue
                
            data = response.json()
            market_data = data.get('market_data', {})
            if not market_data:
                print(f"Error: No market data for {crypto_id}")
                continue
            
            # Extract enhanced metrics
            metrics = {
                'symbol': data.get('symbol', '').upper(),
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'volume': market_data.get('total_volume', {}).get('usd', 0),
                'high_24h': market_data.get('high_24h', {}).get('usd', 0),
                'low_24h': market_data.get('low_24h', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                'price_change_1h': market_data.get('price_change_percentage_1h_in_currency', {}).get('usd', 0),
                'price_change_14d': market_data.get('price_change_percentage_14d', 0),
                'price_change_1y': market_data.get('price_change_percentage_1y', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0),
                'ath_date': market_data.get('ath_date', {}).get('usd', ''),
                'atl': market_data.get('atl', {}).get('usd', 0),
                'atl_change_percentage': market_data.get('atl_change_percentage', {}).get('usd', 0),
                'atl_date': market_data.get('atl_date', {}).get('usd', ''),
                'market_cap_rank': data.get('market_cap_rank', 'N/A'),
                'market_cap_change_24h': market_data.get('market_cap_change_percentage_24h', 0),
                'total_supply': market_data.get('total_supply', 0),
                'max_supply': market_data.get('max_supply', 0),
                'circulating_supply': market_data.get('circulating_supply', 0),
                'fully_diluted_valuation': market_data.get('fully_diluted_valuation', {}).get('usd', 0),
                'total_value_locked': market_data.get('total_value_locked', None),
                'developer_data': data.get('developer_data', {}),
                'community_data': data.get('community_data', {}),
                'public_interest_stats': data.get('public_interest_stats', {})
            }
            
            all_metrics[crypto_id] = metrics
            print(f"Metrics extracted successfully for {crypto_id}")
        
        if rate_limited:
            return jsonify({
                'error': 'Rate limit exceeded. Please try again in a minute or reduce the number of cryptocurrencies.',
                'partial_metrics': all_metrics if all_metrics else None
            }), 429
        
        if not all_metrics:
            return jsonify({'error': 'No data could be retrieved for any cryptocurrency'}), 500
            
        # Build analysis prompt based on available metrics
        analyzed_coins = list(all_metrics.keys())
        
        # Determine the type of analysis needed based on user input
        query_lower = user_input.lower()
        
        # Advanced analysis keywords
        advanced_keywords = [
            'technical analysis', 'trading strategy', 'market conditions', 'volume analysis',
            'indicators', 'trends', 'correlations', 'risk assessment', 'predictive',
            'liquidity', 'volatility', 'statistical', 'trading volume', 'market sentiment',
            'rsi', 'macd', 'bollinger', 'entry points', 'exit points', 'stop-loss'
        ]
        
        # Basic/intro keywords
        intro_keywords = [
            'intro', 'introduction', 'explain', 'what is', 'tell me about', 'basics',
            'beginner', 'new to', 'help understand', 'learn about'
        ]
        
        # Count keyword matches
        advanced_count = sum(1 for keyword in advanced_keywords if keyword in query_lower)
        intro_count = sum(1 for keyword in intro_keywords if keyword in query_lower)
        
        # Determine query complexity
        is_advanced = advanced_count >= 2  # If query contains multiple advanced terms
        is_intro = intro_count > 0 and not is_advanced  # Intro only if not advanced
        
        if is_intro:
            # Get the first coin's metrics for intro analysis
            first_coin = analyzed_coins[0]
            first_coin_metrics = all_metrics[first_coin]
            
            prompt = f"""As a cryptocurrency expert, provide a beginner-friendly introduction to {first_coin}. The current data shows:

{first_coin.upper()} Overview:
• Current Price: ${first_coin_metrics['current_price']:,.2f}
• Market Cap: ${first_coin_metrics['market_cap']:,.2f}
• 24h Change: {first_coin_metrics['price_change_24h']:+.2f}%

Focus on:
1. What {first_coin.title()} is and its basic purpose
2. Key features that make it valuable
3. Simple explanation of how it works
4. Basic terms a beginner should know
5. Current market status in simple terms

Keep the explanation friendly, avoid technical jargon, and use simple analogies where helpful.
Use the current price data to give context but focus on the fundamentals.

User Query: {user_input}"""

            system_content = """You are a cryptocurrency expert focusing on beginner education:
            - Use simple, clear language
            - Avoid technical jargon unless explaining it
            - Use analogies and examples
            - Focus on fundamentals
            - Make concepts accessible to newcomers"""

        elif is_advanced:
            # Build market metrics section for all analyzed coins
            market_metrics = []
            for coin in analyzed_coins:
                metrics = all_metrics[coin]
                market_metrics.append(f"""{coin.upper()}:
• Price: ${metrics['current_price']:,.2f}
• 24h Range: ${metrics['high_24h']:,.2f} (H) / ${metrics['low_24h']:,.2f} (L)
• Volume: ${metrics['volume']:,.2f}
• Market Cap: ${metrics['market_cap']:,.2f}
• Changes: 1h: {metrics['price_change_1h']:+.2f}% | 24h: {metrics['price_change_24h']:+.2f}% | 7d: {metrics['price_change_7d']:+.2f}%""")
            
            # Calculate market structure metrics if we have multiple coins
            market_structure = []
            if len(analyzed_coins) > 1:
                total_mcap = sum(all_metrics[coin]['market_cap'] for coin in analyzed_coins)
                for coin in analyzed_coins:
                    dominance = (all_metrics[coin]['market_cap'] / total_mcap * 100)
                    market_structure.append(f"• {coin.upper()} Dominance: {dominance:.2f}%")
            
            # Volume analysis for all coins
            volume_analysis = []
            for coin in analyzed_coins:
                metrics = all_metrics[coin]
                vol_mcap = (metrics['volume'] / metrics['market_cap'] * 100)
                volume_analysis.append(f"• {coin.upper()} Vol/MCap: {vol_mcap:.2f}%")
            
            prompt = f"""As a quantitative cryptocurrency analyst, provide advanced market analysis based on the following real-time data:

### MARKET METRICS

{chr(10).join(market_metrics)}

{f'''### MARKET STRUCTURE
{chr(10).join(market_structure)}''' if market_structure else ''}

### VOLUME ANALYSIS
{chr(10).join(volume_analysis)}

User Query: {user_input}

Provide institutional-grade analysis focusing on:
1. Technical Analysis (RSI, MACD, Bollinger Bands)
2. Volume Profile and Market Microstructure
3. Inter-market Correlations
4. Risk Assessment and Position Sizing
5. Entry/Exit Points with Stop-Loss Levels
6. Market Sentiment Analysis
7. Short-term Price Predictions with Confidence Intervals"""

            system_content = """You are an elite quantitative cryptocurrency analyst specializing in:
            - Advanced Technical Analysis
            - Market Microstructure
            - Statistical Arbitrage
            - Risk Management
            - Derivatives Trading
            - Algorithmic Strategies
            
            Provide institutional-grade analysis:
            - Use advanced trading terminology
            - Include specific technical indicators
            - Give precise entry/exit points
            - Calculate risk metrics
            - Provide confidence intervals
            - Reference statistical evidence
            
            Format output with clear sections for:
            - Technical Analysis
            - Volume Profile
            - Risk Assessment
            - Trade Setup
            - Price Targets"""

        else:
            # Standard analysis for general queries
            market_overview = []
            for coin in analyzed_coins:
                metrics = all_metrics[coin]
                market_overview.append(f"{coin.upper()}: ${metrics['current_price']:,.2f} ({metrics['price_change_24h']:+.2f}% 24h)")
            
            prompt = f"""As a cryptocurrency analyst, provide a balanced analysis of the current market:

### MARKET OVERVIEW
{chr(10).join(market_overview)}

Focus on:
1. Current market conditions
2. Notable price movements
3. Key support/resistance levels
4. Market sentiment
5. Trading opportunities

User Query: {user_input}"""

            system_content = """You are a cryptocurrency analyst providing balanced market insights:
            - Combine technical and fundamental analysis
            - Use clear, professional language
            - Include both data and context
            - Provide actionable insights
            - Balance detail with accessibility"""

        print("Sending request to OpenAI")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            print("Received response from OpenAI")
            
        except Exception as openai_error:
            print(f"OpenAI API Error: {str(openai_error)}")
            return jsonify({'error': f'OpenAI API Error: {str(openai_error)}'}), 500
        
        # Format the final response
        if is_intro:
            first_coin = analyzed_coins[0]
            first_coin_metrics = all_metrics[first_coin]
            formatted_analysis = f"""### {first_coin.title()} Introduction

{analysis}

### Note
• Current {first_coin.upper()} Price: ${first_coin_metrics['current_price']:,.2f}
• This introduction is meant to help beginners understand {first_coin.title()}
• For more detailed analysis, feel free to ask specific questions"""
        else:
            formatted_analysis = f"""### Market Analysis

{analysis}

### Risk Disclaimer
- All analysis is based on historical data and current market conditions
- Past performance does not guarantee future results
- Always manage position sizes according to your risk tolerance
- Consider using stop-loss orders and proper position sizing
- Cryptocurrency markets are highly volatile and risky"""
            
        return jsonify({
            'Market Analysis': formatted_analysis,
            'Metrics': all_metrics
        })
        
    except Exception as e:
        error_msg = f"Error processing command: {str(e)}"
        print(f"Critical error: {error_msg}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        return jsonify({'error': error_msg}), 500

# Add CORS headers to allow requests from frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5050)