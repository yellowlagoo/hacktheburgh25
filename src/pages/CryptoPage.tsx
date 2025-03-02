import React, { useState, useEffect } from 'react';
import CryptoCard from '../components/CryptoCard';

interface CryptoPageProps {
  darkMode: boolean;
}

interface Crypto {
  id: string;
  name: string;
  symbol: string;
  current_price: number;
  price_change_24h: number;
  price_change_1h: number;
  market_cap: number;
  total_volume: number;
  high_24h: number;
  low_24h: number;
}

const CryptoPage: React.FC<CryptoPageProps> = ({ darkMode }) => {
  const [cryptos, setCryptos] = useState<Crypto[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchCryptos = async () => {
      try {
        const cryptoIds = ['bitcoin', 'ethereum', 'cardano', 'solana', 'ripple', 'polkadot', 'dogecoin', 'avalanche'];
        const cryptoData = await Promise.all(
          cryptoIds.map(async (id) => {
            const response = await fetch(`http://localhost:5050/price/${id}`);
            const data = await response.json();
            
            if (data.error) {
              console.error(`Error fetching ${id}:`, data.error);
              return null;
            }

            const coinData = data[id];
            return {
              id,
              name: id.charAt(0).toUpperCase() + id.slice(1),
              symbol: id.substring(0, 3).toUpperCase(),
              current_price: coinData.usd,
              price_change_24h: coinData.usd_24h_change,
              price_change_1h: coinData.usd_1h_change || 0,
              market_cap: coinData.usd_market_cap,
              total_volume: coinData.usd_24h_vol,
              high_24h: coinData.usd_24h_high || 0,
              low_24h: coinData.usd_24h_low || 0
            };
          })
        );

        // Filter out any null values from failed requests
        const validData = cryptoData.filter((data): data is Crypto => data !== null);
        setCryptos(validData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching crypto data:', error);
        setIsLoading(false);
      }
    };

    fetchCryptos();
    // Update data every minute
    const interval = setInterval(fetchCryptos, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`container mx-auto px-4 py-8 ${darkMode ? 'dark' : ''}`}>
      <div className="text-center mb-8">
        <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
          Cryptocurrency Dashboard
        </h1>
        <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Real-time market prices and trends
        </p>
      </div>

      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_infinite]"></div>
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.4s_infinite]"></div>
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.8s_infinite]"></div>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {cryptos.map((crypto) => (
            <CryptoCard
              key={crypto.id}
              id={crypto.id}
              name={crypto.name}
              symbol={crypto.symbol}
              price={crypto.current_price}
              change24h={crypto.price_change_24h}
              change1h={crypto.price_change_1h}
              marketCap={crypto.market_cap}
              volume={crypto.total_volume}
              high24h={crypto.high_24h}
              low24h={crypto.low_24h}
              darkMode={darkMode}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default CryptoPage;