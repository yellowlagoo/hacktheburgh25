from flask import Flask, jsonify, request
import requests
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
from functools import lru_cache
from datetime import datetime, timedelta
import json
from market_sentiment_analyzer import analyze_market_sentiment, ConsumerDataAnalyzer
from flask_cors import CORS
import re

load_dotenv()

# Initialize OpenAI client with just the API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:5174"]}})

# Error handler for all exceptions
@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    response = {
        "error": str(error),
        "status": "error"
    }
    return jsonify(response), 500

# Cache for storing API responses
price_cache = {}
CACHE_DURATION = 30  # seconds

# Expanded crypto mapping with more coins and common terms
crypto_mapping = {
    # Major coins and their symbols
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'bnb': 'binancecoin',
    'sol': 'solana',
    'xrp': 'ripple',
    'ada': 'cardano',
    'doge': 'dogecoin',
    'dot': 'polkadot',
    'matic': 'polygon',
    'link': 'chainlink',
    'uni': 'uniswap',
    'avax': 'avalanche-2',
    'atom': 'cosmos',
    'ltc': 'litecoin',
    'near': 'near',
    
    # Common terms and variations
    'bitcoin': 'bitcoin',
    'ethereum': 'ethereum',
    'binance': 'binancecoin',
    'solana': 'solana',
    'ripple': 'ripple',
    'cardano': 'cardano',
    'dogecoin': 'dogecoin',
    'polkadot': 'polkadot',
    'polygon': 'polygon',
    'chainlink': 'chainlink',
    'uniswap': 'uniswap',
    'avalanche': 'avalanche-2',
    'cosmos': 'cosmos',
    'litecoin': 'litecoin',
    
    # Parenthetical variations
    'bitcoin (btc)': 'bitcoin',
    'ethereum (eth)': 'ethereum',
    'ripple (xrp)': 'ripple',
    'cardano (ada)': 'cardano',
    'binance coin (bnb)': 'binancecoin',
    'solana (sol)': 'solana',
    'dogecoin (doge)': 'dogecoin',
    'polkadot (dot)': 'polkadot',
    'polygon (matic)': 'polygon',
    'chainlink (link)': 'chainlink',
    'uniswap (uni)': 'uniswap',
    'avalanche (avax)': 'avalanche-2',
    'cosmos (atom)': 'cosmos',
    'litecoin (ltc)': 'litecoin',
    'near (near)': 'near',
    
    # Common phrases and variations
    'btc price': 'bitcoin',
    'eth price': 'ethereum',
    'bitcoin price': 'bitcoin',
    'ethereum price': 'ethereum',
    'price of bitcoin': 'bitcoin',
    'price of ethereum': 'ethereum'
}

# Default coins to analyze if none specified
DEFAULT_COINS = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'ripple', 'cardano', 'dogecoin', 'polkadot', 'polygon', 'chainlink', 'uniswap', 'avalanche-2', 'cosmos', 'litecoin', 'near']
MAX_COINS = 15  # Maximum number of coins to analyze at once

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
        # Convert to lowercase and get mapped ID if exists
        crypto_id = crypto.lower()
        crypto_id = crypto_mapping.get(crypto_id, crypto_id)
        
        print(f"\nFetching price for: {crypto_id}")
        
        # Simplified URL with only essential fields
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true'
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
        
        if crypto_id not in data:
            error_msg = f'No data available for {crypto_id}'
            print(f"Error: {error_msg}")
            return jsonify({'error': error_msg}), 404
        
        coin_data = data[crypto_id]
        result = {
            crypto_id: {
                'usd': coin_data.get('usd', 0),
                'usd_market_cap': coin_data.get('usd_market_cap', 0),
                'usd_24h_vol': coin_data.get('usd_24h_vol', 0),
                'usd_24h_change': coin_data.get('usd_24h_change', 0)
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
    """Extract cryptocurrency symbols from user input with improved handling."""
    query_lower = user_input.lower()
    requested_coins = set()  # Using set to avoid duplicates
    
    print(f"Processing query: {query_lower}")
    
    # First, try to find full names with parenthetical symbols
    # This will match patterns like "Bitcoin (BTC)" or "Ethereum (ETH)"
    full_matches = re.findall(r'([a-zA-Z\s]+)\s*\(([^)]+)\)', query_lower)
    print(f"Found full name matches: {full_matches}")
    
    for full_name, symbol in full_matches:
        clean_symbol = symbol.strip().lower()
        clean_name = full_name.strip().lower()
        # Try the symbol first, then the full name
        if clean_symbol in crypto_mapping:
            requested_coins.add(crypto_mapping[clean_symbol])
            print(f"Added coin from symbol in parentheses: {crypto_mapping[clean_symbol]}")
        elif clean_name in crypto_mapping:
            requested_coins.add(crypto_mapping[clean_name])
            print(f"Added coin from full name: {crypto_mapping[clean_name]}")
    
    # If no full matches found, look for standalone symbols in parentheses
    if not requested_coins:
        symbol_matches = re.findall(r'\(([^)]+)\)', query_lower)
        print(f"Found symbol matches: {symbol_matches}")
        for symbol in symbol_matches:
            clean_symbol = symbol.strip().lower()
            if clean_symbol in crypto_mapping:
                requested_coins.add(crypto_mapping[clean_symbol])
                print(f"Added coin from standalone symbol: {crypto_mapping[clean_symbol]}")
    
    # Look for exact matches of coin names or symbols
    if not requested_coins:
        words = query_lower.split()
        for word in words:
            clean_word = word.strip('.,!?()[]{}').lower()
            if clean_word in crypto_mapping:
                requested_coins.add(crypto_mapping[clean_word])
                print(f"Added coin from word match: {crypto_mapping[clean_word]}")
    
    # If we found specific coins, return them
    if requested_coins:
        print(f"Final coins selected: {list(requested_coins)}")
        return list(requested_coins)[:MAX_COINS]
    
    # If the query mentions crypto but no specific coins found, return an empty list
    crypto_terms = ['cryptocurrency', 'crypto', 'coin', 'token', 'blockchain']
    if any(term in query_lower for term in crypto_terms):
        print("Query contains crypto terms but no specific coins identified")
        return []
    
    # Only use default coins if explicitly requested
    if 'default' in query_lower or 'all coins' in query_lower:
        print("Using default coins as explicitly requested")
        return DEFAULT_COINS
    
    # If we get here, no coins were found and no crypto terms were present
    print("No coins found in query")
    return []

@app.route('/command', methods=['POST'])
def process_command():
    try:
        data = request.get_json()
        command = data.get('command', '').lower()
        is_intro = data.get('is_intro', False)
        
        # Extract coins from command
        coins = extract_coin_symbols(command)
        
        # Collect metrics for all requested coins
        all_metrics = {}
        analyzed_coins = []
        
        for coin in coins:
            try:
                # Get current price and market data
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true&include_last_updated_at=true"
                response = make_request(url)
                
                if response.status_code == 200:
                    data = response.json()
                    if coin in data:
                        coin_data = data[coin]
                        metrics = {
                            'current_price': coin_data['usd'],
                            '24h_volume': coin_data.get('usd_24h_vol', 0),
                            '24h_change': coin_data.get('usd_24h_change', 0),
                            'market_cap': coin_data.get('usd_market_cap', 0),
                            'last_updated': coin_data.get('last_updated_at', 0)
                        }
                        all_metrics[coin] = metrics
                        analyzed_coins.append(coin)
                        print(f"Successfully got data for {coin}")
                    else:
                        print(f"No data available for {coin}")
                else:
                    print(f"Failed to get data for {coin}: {response.status_code}")
            except Exception as e:
                print(f"Error processing {coin}: {str(e)}")
                continue  # Continue with other coins if one fails
        
        if not analyzed_coins:
            return jsonify({
                'error': 'No valid coins to analyze. Try mentioning specific cryptocurrencies like Bitcoin (BTC) or Ethereum (ETH).'
            }), 400
        
        # Get market sentiment analysis
        try:
            consumer_analyzer = ConsumerDataAnalyzer('APU0000717311.csv')
            coffee_sentiment = consumer_analyzer.process_consumer_data()
            sentiment_description = "positive" if coffee_sentiment > 0.6 else "neutral" if coffee_sentiment > 0.4 else "negative"
            sentiment_context = f"\nConsumer spending sentiment is {sentiment_description} (score: {coffee_sentiment:.2f}), based on coffee price trends."
        except Exception as e:
            print(f"Error getting sentiment analysis: {e}")
            sentiment_description = "neutral"
            coffee_sentiment = 0.5
            sentiment_context = "Unable to analyze market sentiment due to an error."

        # Generate analysis using OpenAI
        system_prompt = """You are an expert cryptocurrency analyst and educator. Adapt your analysis style and complexity to match the user's apparent knowledge level and interests, which you should infer from their query style, terminology usage, and specific requests.

Key Principles:
1. Match Technical Depth: Scale technical complexity based on the user's apparent expertise level
2. Maintain Accessibility: Always explain complex concepts when introducing them
3. Progressive Disclosure: Layer information from fundamental to advanced as needed
4. Context Awareness: Reference relevant market metrics and indicators appropriately
5. Educational Value: Weave explanations and insights naturally into analysis

Analysis Framework:
- Adapt technical depth based on query complexity
- Include relevant metrics and indicators for context
- Explain market dynamics at appropriate depth
- Connect analysis to practical implications
- Provide actionable insights scaled to user sophistication"""

        prompt = f"Analyze the following cryptocurrency metrics and market sentiment:\n\n"
        
        # Add time context
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        prompt += f"Analysis Time: {current_time}\n\n"
        
        # Add coin metrics
        for coin in analyzed_coins:
            metrics = all_metrics[coin]
            prompt += f"{coin.title()}:\n"
            prompt += f"- Current Price: ${metrics['current_price']:,.2f}\n"
            prompt += f"- 24h Volume: ${metrics['24h_volume']:,.2f}\n"
            prompt += f"- 24h Change: {metrics['24h_change']:.2f}%\n"
            prompt += f"- Market Cap: ${metrics['market_cap']:,.2f}\n\n"
        
        # Add sentiment analysis
        prompt += f"Market Sentiment Analysis:\n"
        prompt += f"- Overall Sentiment Score: {coffee_sentiment:.2f}\n"
        prompt += f"- Overall Sentiment: {sentiment_description}\n"
        prompt += f"- Sentiment Context: {sentiment_context}\n\n"

        # Add user's original query for context
        prompt += f"User Query: {command}\n\n"
        
        # Add analysis guidance
        prompt += """Based on the user's query style and terminology, provide an appropriately detailed analysis that:
1. Matches the technical depth to their apparent knowledge level
2. Explains any complex concepts introduced
3. Provides relevant context and background as needed
4. Focuses on aspects most relevant to their interests
5. Delivers actionable insights at an appropriate level

Include market metrics, technical analysis, and recommendations that align with the user's demonstrated sophistication level."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        analysis = response.choices[0].message.content
        
        # Format the final response with enhanced structure
        if is_intro:
            first_coin = analyzed_coins[0]
            first_coin_metrics = all_metrics[first_coin]
            formatted_analysis = f"""### {first_coin.title()} Introduction

{analysis}

### Market Sentiment & Technical Indicators
• Overall Sentiment: {sentiment_description.title()} ({coffee_sentiment:.2f})
• Market Confidence: {get_confidence_level(coffee_sentiment).title()}
• Consumer Trends: {sentiment_context}

### Current Market Metrics ({current_time})
• Price: ${first_coin_metrics['current_price']:,.2f}
• 24h Change: {first_coin_metrics['24h_change']:+.2f}%
• Volume: ${first_coin_metrics['24h_volume']:,.2f}
• Market Cap: ${first_coin_metrics['market_cap']:,.2f}

### Risk Assessment
• Market Phase: {get_market_phase({'price_change_24h': first_coin_metrics['24h_change']})}
• Risk Level: {get_risk_level({'price_change_24h': first_coin_metrics['24h_change']})}
• Volatility: {get_volatility_measure({'high_24h': first_coin_metrics['current_price'] * 1.1, 'low_24h': first_coin_metrics['current_price'] * 0.9})}

### Additional Resources
For more detailed analysis, consider exploring:
- Technical indicators and chart patterns
- On-chain metrics and network health
- Development activity and ecosystem growth
- Regulatory developments and compliance updates"""
        else:
            formatted_analysis = f"""### Comprehensive Market Analysis

{analysis}

### Market Sentiment & Technical Context
• Overall Sentiment: {sentiment_description.title()} ({coffee_sentiment:.2f})
• Market Confidence: {get_confidence_level(coffee_sentiment).title()}
• Market Context: {sentiment_context}

### Current Market State ({current_time})
{chr(10).join(f"**{coin.upper()}**\n• Price: ${all_metrics[coin]['current_price']:,.2f}\n• 24h Change: {all_metrics[coin]['24h_change']:+.2f}%\n• Volume: ${all_metrics[coin]['24h_volume']:,.2f}\n• Market Cap: ${all_metrics[coin]['market_cap']:,.2f}" for coin in analyzed_coins)}

### Risk Management Guidelines
- Set clear entry and exit points based on technical levels
- Use appropriate position sizing (1-2% risk per trade)
- Implement stop-loss orders to protect capital
- Consider using trailing stops in trending markets
- Diversify across multiple cryptocurrencies and strategies

### Disclaimer
- Analysis based on current market data and historical patterns
- Past performance does not guarantee future results
- Cryptocurrency markets are highly volatile
- Always conduct your own research (DYOR)
- Never invest more than you can afford to lose"""
            
        return jsonify({
            'Market Analysis': formatted_analysis,
            'Metrics': all_metrics,
            'Sentiment': {
                'score': coffee_sentiment,
                'description': sentiment_description,
                'context': sentiment_context
            }
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

def get_confidence_level(sentiment_score):
    if sentiment_score > 0.7:
        return "very high"
    elif sentiment_score > 0.6:
        return "high"
    elif sentiment_score > 0.4:
        return "moderate"
    elif sentiment_score > 0.3:
        return "low"
    else:
        return "very low"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051, debug=True)