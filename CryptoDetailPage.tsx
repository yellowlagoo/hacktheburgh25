import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown } from 'lucide-react';
import { Link } from 'react-router-dom';
import CryptoChart from '../components/CryptoChart';
import TimeframeSelector from '../components/TimeframeSelector';

interface CryptoDetailPageProps {
  darkMode: boolean;
}

interface CryptoDetail {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  marketCap: number;
  volume24h: number;
  circulatingSupply: number;
}

const CryptoDetailPage: React.FC<CryptoDetailPageProps> = ({ darkMode }) => {
  const { id } = useParams<{ id: string }>();
  const [crypto, setCrypto] = useState<CryptoDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [chartData, setChartData] = useState<{ data: number[], labels: string[] }>({ data: [], labels: [] });
  
  const timeframes = ['1H', '1D', '1W', '1M', '1Y'];

  useEffect(() => {
    // Simulate fetching crypto details
    setTimeout(() => {
      const mockCryptoDetails: Record<string, CryptoDetail> = {
        'bitcoin': {
          id: 'bitcoin',
          name: 'Bitcoin',
          symbol: 'btc',
          price: 57832.41,
          change24h: 2.34,
          marketCap: 1089543256789,
          volume24h: 32456789012,
          circulatingSupply: 18956432
        },
        'ethereum': {
          id: 'ethereum',
          name: 'Ethereum',
          symbol: 'eth',
          price: 2843.12,
          change24h: -0.78,
          marketCap: 342567890123,
          volume24h: 18765432109,
          circulatingSupply: 120543210
        },
        'cardano': {
          id: 'cardano',
          name: 'Cardano',
          symbol: 'ada',
          price: 1.23,
          change24h: 5.67,
          marketCap: 43210987654,
          volume24h: 2109876543,
          circulatingSupply: 35123456789
        },
        'solana': {
          id: 'solana',
          name: 'Solana',
          symbol: 'sol',
          price: 102.45,
          change24h: 8.91,
          marketCap: 38765432109,
          volume24h: 5432109876,
          circulatingSupply: 378654321
        },
        'ripple': {
          id: 'ripple',
          name: 'XRP',
          symbol: 'xrp',
          price: 0.58,
          change24h: -1.23,
          marketCap: 28765432109,
          volume24h: 1987654321,
          circulatingSupply: 49876543210
        },
        'polkadot': {
          id: 'polkadot',
          name: 'Polkadot',
          symbol: 'dot',
          price: 18.76,
          change24h: 3.45,
          marketCap: 21098765432,
          volume24h: 1098765432,
          circulatingSupply: 1123456789
        },
        'dogecoin': {
          id: 'dogecoin',
          name: 'Dogecoin',
          symbol: 'doge',
          price: 0.12,
          change24h: -2.56,
          marketCap: 16789012345,
          volume24h: 987654321,
          circulatingSupply: 139876543210
        },
        'avalanche': {
          id: 'avalanche',
          name: 'Avalanche',
          symbol: 'avax',
          price: 34.21,
          change24h: 4.32,
          marketCap: 12345678901,
          volume24h: 876543210,
          circulatingSupply: 360987654
        }
      };
      
      if (id && mockCryptoDetails[id]) {
        setCrypto(mockCryptoDetails[id]);
      }
      
      setIsLoading(false);
    }, 1500);
  }, [id]);

  useEffect(() => {
    if (!crypto) return;
    
    // Generate mock chart data based on the selected timeframe
    const generateMockChartData = (timeframe: string) => {
      let dataPoints: number[] = [];
      let labels: string[] = [];
      const basePrice = crypto.price;
      const volatility = 0.05; // 5% volatility
      
      switch (timeframe) {
        case '1H':
          // Generate data for each 5 minutes in the last hour
          for (let i = 0; i < 12; i++) {
            const time = new Date();
            time.setMinutes(time.getMinutes() - (11 - i) * 5);
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
            
            const randomChange = (Math.random() - 0.5) * 2 * volatility * basePrice;
            dataPoints.push(basePrice + randomChange * (i / 11));
          }
          break;
          
        case '1D':
          // Generate data for each hour in the last day
          for (let i = 0; i < 24; i++) {
            const time = new Date();
            time.setHours(time.getHours() - (23 - i));
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
            
            const randomChange = (Math.random() - 0.5) * 2 * volatility * basePrice;
            dataPoints.push(basePrice + randomChange * (i / 23));
          }
          break;
          
        case '1W':
          // Generate data for each day in the last week
          for (let i = 0; i < 7; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (6 - i));
            labels.push(date.toLocaleDateString([], { weekday: 'short' }));
            
            const randomChange = (Math.random() - 0.5) * 2 * volatility * 3 * basePrice;
            dataPoints.push(basePrice + randomChange * (i / 6));
          }
          break;
          
        case '1M':
          // Generate data for each 3 days in the last month
          for (let i = 0; i < 10; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (30 - i * 3));
            labels.push(date.toLocaleDateString([], { month: 'short', day: 'numeric' }));
            
            const randomChange = (Math.random() - 0.5) * 2 * volatility * 5 * basePrice;
            dataPoints.push(basePrice + randomChange * (i / 9));
          }
          break;
          
        case '1Y':
          // Generate data for each month in the last year
          for (let i = 0; i < 12; i++) {
            const date = new Date();
            date.setMonth(date.getMonth() - (11 - i));
            labels.push(date.toLocaleDateString([], { month: 'short' }));
            
            const randomChange = (Math.random() - 0.5) * 2 * volatility * 10 * basePrice;
            dataPoints.push(basePrice + randomChange * (i / 11));
          }
          break;
      }
      
      return { data: dataPoints, labels };
    };
    
    setChartData(generateMockChartData(selectedTimeframe));
  }, [selectedTimeframe, crypto]);

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="flex space-x-2">
          <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_infinite]"></div>
          <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.4s_infinite]"></div>
          <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.8s_infinite]"></div>
        </div>
      </div>
    );
  }

  if (!crypto) {
    return (
      <div className={`text-center ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        <p>Cryptocurrency not found.</p>
        <Link to="/crypto" className="text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300">
          Back to Crypto Dashboard
        </Link>
      </div>
    );
  }

  const isPositive = crypto.change24h >= 0;

  return (
    <div className="container mx-auto px-4 py-8">
      <Link to="/crypto" className={`flex items-center mb-6 ${darkMode ? 'text-white' : 'text-gray-800'} hover:text-indigo-600 dark:hover:text-indigo-400`}>
        <ArrowLeft size={20} className="mr-2" />
        Back to Crypto Dashboard
      </Link>
      
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              {crypto.name} ({crypto.symbol.toUpperCase()})
            </h1>
            <p className={`text-2xl font-semibold mt-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              ${crypto.price.toLocaleString()}
            </p>
          </div>
          
          <div className={`flex items-center text-lg ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {isPositive ? <TrendingUp size={24} className="mr-2" /> : <TrendingDown size={24} className="mr-2" />}
            <span>{crypto.change24h.toFixed(2)}%</span>
          </div>
        </div>
      </div>
      
      <div className={`p-6 rounded-lg mb-8 ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
        <TimeframeSelector 
          timeframes={timeframes} 
          selectedTimeframe={selectedTimeframe} 
          onSelect={setSelectedTimeframe}
          darkMode={darkMode}
        />
        
        <div className="h-80">
          <CryptoChart 
            data={chartData.data} 
            labels={chartData.labels} 
            timeframe={selectedTimeframe}
            darkMode={darkMode}
          />
        </div>
      </div>
      
      <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800 text-white' : 'bg-white shadow-md text-gray-800'}`}>
        <h2 className="text-xl font-bold mb-4">Market Stats</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Market Cap</p>
            <p className="text-lg font-semibold">${crypto.marketCap.toLocaleString()}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">24h Trading Volume</p>
            <p className="text-lg font-semibold">${crypto.volume24h.toLocaleString()}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Circulating Supply</p>
            <p className="text-lg font-semibold">{crypto.circulatingSupply.toLocaleString()} {crypto.symbol.toUpperCase()}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CryptoDetailPage;