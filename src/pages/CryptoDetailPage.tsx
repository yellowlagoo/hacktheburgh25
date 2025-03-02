import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { ArrowLeft, TrendingUp, TrendingDown } from 'lucide-react';
import { Link } from 'react-router-dom';
import CryptoChart from '../components/CryptoChart';
import TimeframeSelector from '../components/TimeframeSelector';
import CommandInput from '../components/CommandInput';

interface CryptoDetailPageProps {
  darkMode: boolean;
}

interface CryptoDetail {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  change1h: number;
  marketCap: number;
  volume24h: number;
  high24h: number;
  low24h: number;
}

interface Analysis {
  analysis: string;
  metrics: {
    current_price: number;
    market_cap: number;
    volume: number;
    high_24h: number;
    low_24h: number;
    price_change: number;
    price_change_7d: number;
    price_change_30d: number;
  };
}

const CryptoDetailPage: React.FC<CryptoDetailPageProps> = ({ darkMode }) => {
  const { id } = useParams<{ id: string }>();
  const [crypto, setCrypto] = useState<CryptoDetail | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');
  const [chartData, setChartData] = useState<{ data: number[], labels: string[] }>({ data: [], labels: [] });
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  const timeframes = ['1H', '1D', '1W', '1M', '1Y'];

  useEffect(() => {
    const fetchCryptoDetails = async () => {
      try {
        if (!id) return;
        
        const response = await fetch(`http://localhost:5050/price/${id}`);
        const data = await response.json();
        
        if (data.error) {
          console.error(`Error fetching ${id}:`, data.error);
          setIsLoading(false);
          return;
        }

        const coinData = data[id];
        setCrypto({
          id,
          name: id.charAt(0).toUpperCase() + id.slice(1),
          symbol: id.substring(0, 3).toUpperCase(),
          price: coinData.usd,
          change24h: coinData.usd_24h_change,
          change1h: coinData.usd_1h_change,
          marketCap: coinData.usd_market_cap,
          volume24h: coinData.usd_24h_vol,
          high24h: coinData.usd_24h_high,
          low24h: coinData.usd_24h_low
        });
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching crypto details:', error);
        setIsLoading(false);
      }
    };

    fetchCryptoDetails();
    const interval = setInterval(fetchCryptoDetails, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [id]);

  useEffect(() => {
    if (!crypto) return;
    
    // Generate mock chart data based on the selected timeframe
    const generateChartData = (timeframe: string) => {
      let dataPoints: number[] = [];
      let labels: string[] = [];
      const basePrice = crypto.price;
      const volatility = Math.abs(crypto.change24h / 100); // Use actual 24h change for volatility
      
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
    
    setChartData(generateChartData(selectedTimeframe));
  }, [selectedTimeframe, crypto]);

  const handleCommand = async (command: string) => {
    if (!id) return;
    
    setIsAnalyzing(true);
    try {
      const response = await fetch('http://localhost:5050/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command,
          crypto_id: id,
        }),
      });
      
      const data = await response.json();
      if (data.error) {
        console.error('Error:', data.error);
        return;
      }
      
      setAnalysis(data);
    } catch (error) {
      console.error('Error processing command:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

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
              {crypto.name} ({crypto.symbol})
            </h1>
            <p className={`text-2xl font-semibold mt-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              ${crypto.price.toLocaleString()}
            </p>
          </div>
          
          <div className={`flex items-center text-lg ${crypto.change24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {crypto.change24h >= 0 ? <TrendingUp size={24} className="mr-2" /> : <TrendingDown size={24} className="mr-2" />}
            <span>{crypto.change24h.toFixed(2)}%</span>
          </div>
        </div>
      </div>
      
      <div className={`p-6 rounded-lg mb-8 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white shadow-md text-gray-800'}`}>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">1h Change</p>
            <p className={`text-lg font-semibold ${crypto.change1h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {crypto.change1h.toFixed(2)}%
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">24h High</p>
            <p className="text-lg font-semibold">${crypto.high24h.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">24h Low</p>
            <p className="text-lg font-semibold">${crypto.low24h.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">24h Volume</p>
            <p className="text-lg font-semibold">${crypto.volume24h.toLocaleString()}</p>
          </div>
        </div>
      </div>
      
      <div className="mb-8">
        <CommandInput
          darkMode={darkMode}
          onSubmit={handleCommand}
          isLoading={isAnalyzing}
        />
        
        {analysis && (
          <div className={`mt-4 p-6 rounded-lg ${darkMode ? 'bg-gray-800 text-white' : 'bg-white shadow-md text-gray-800'}`}>
            <h2 className="text-xl font-bold mb-4">Analysis</h2>
            <div className="prose max-w-none dark:prose-invert">
              {analysis.analysis.split('\n').map((paragraph, index) => (
                <p key={index} className="mb-4">{paragraph}</p>
              ))}
            </div>
          </div>
        )}
      </div>
      
      <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800 text-white' : 'bg-white shadow-md text-gray-800'}`}>
        <TimeframeSelector 
          timeframes={timeframes} 
          selectedTimeframe={selectedTimeframe} 
          onSelect={setSelectedTimeframe}
          darkMode={darkMode}
        />
        
        <div className="h-80 mt-4">
          <CryptoChart 
            data={chartData.data} 
            labels={chartData.labels} 
            timeframe={selectedTimeframe}
            darkMode={darkMode}
          />
        </div>
      </div>
    </div>
  );
};

export default CryptoDetailPage;