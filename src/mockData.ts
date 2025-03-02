// This file contains mock data for the application
// In a real application, this would be replaced with API calls

export interface Crypto {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  marketCap: number;
  volume24h: number;
  circulatingSupply: number;
}

export const mockCryptos: Crypto[] = [
  {
    id: 'bitcoin',
    name: 'Bitcoin',
    symbol: 'btc',
    price: 57832.41,
    change24h: 2.34,
    marketCap: 1089543256789,
    volume24h: 32456789012,
    circulatingSupply: 18956432
  },
  {
    id: 'ethereum',
    name: 'Ethereum',
    symbol: 'eth',
    price: 2843.12,
    change24h: -0.78,
    marketCap: 342567890123,
    volume24h: 18765432109,
    circulatingSupply: 120543210
  },
  {
    id: 'cardano',
    name: 'Cardano',
    symbol: 'ada',
    price: 1.23,
    change24h: 5.67,
    marketCap: 43210987654,
    volume24h: 2109876543,
    circulatingSupply: 35123456789
  },
  {
    id: 'solana',
    name: 'Solana',
    symbol: 'sol',
    price: 102.45,
    change24h: 8.91,
    marketCap: 38765432109,
    volume24h: 5432109876,
    circulatingSupply: 378654321
  },
  {
    id: 'ripple',
    name: 'XRP',
    symbol: 'xrp',
    price: 0.58,
    change24h: -1.23,
    marketCap: 28765432109,
    volume24h: 1987654321,
    circulatingSupply: 49876543210
  },
  {
    id: 'polkadot',
    name: 'Polkadot',
    symbol: 'dot',
    price: 18.76,
    change24h: 3.45,
    marketCap: 21098765432,
    volume24h: 1098765432,
    circulatingSupply: 1123456789
  },
  {
    id: 'dogecoin',
    name: 'Dogecoin',
    symbol: 'doge',
    price: 0.12,
    change24h: -2.56,
    marketCap: 16789012345,
    volume24h: 987654321,
    circulatingSupply: 139876543210
  },
  {
    id: 'avalanche',
    name: 'Avalanche',
    symbol: 'avax',
    price: 34.21,
    change24h: 4.32,
    marketCap: 12345678901,
    volume24h: 876543210,
    circulatingSupply: 360987654
  }
];

export interface AIResponse {
  summary: string;
  full: string;
}

export const mockAIResponses: Record<string, AIResponse> = {
  weather: {
    summary: "Today's weather forecast shows mild temperatures with a chance of rain.",
    full: "Today's weather forecast indicates mild temperatures ranging from 65°F to 75°F. There's a 40% chance of rain in the afternoon, with winds from the southwest at 5-10 mph. The humidity is expected to be around 65%. Tomorrow should be clearer with slightly higher temperatures."
  },
  help: {
    summary: "I can assist with information, answer questions, or provide guidance on various topics.",
    full: "I'm your AI assistant, designed to help with a wide range of tasks. I can provide information on topics like weather, news, or cryptocurrency prices. I can answer questions about general knowledge, help with simple calculations, or offer guidance on various subjects. Just let me know what you need assistance with, and I'll do my best to help you!"
  }
};