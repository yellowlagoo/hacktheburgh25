import React, { useState, useEffect } from 'react';
import CryptoCard from '../components/CryptoCard';

interface CryptoPageProps {
  darkMode: boolean;
}

interface Crypto {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
}

const CryptoPage: React.FC<CryptoPageProps> = ({ darkMode }) => {
  const [cryptos, setCryptos] = useState<Crypto[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate fetching crypto data
    setTimeout(() => {
      const mockCryptos: Crypto[] = [
        { id: 'bitcoin', name: 'Bitcoin', symbol: 'btc', price: 57832.41, change24h: 2.34 },
        { id: 'ethereum', name: 'Ethereum', symbol: 'eth', price: 2843.12, change24h: -0.78 },
        { id: 'cardano', name: 'Cardano', symbol: 'ada', price: 1.23, change24h: 5.67 },
        { id: 'solana', name: 'Solana', symbol: 'sol', price: 102.45, change24h: 8.91 },
        { id: 'ripple', name: 'XRP', symbol: 'xrp', price: 0.58, change24h: -1.23 },
        { id: 'polkadot', name: 'Polkadot', symbol: 'dot', price: 18.76, change24h: 3.45 },
        { id: 'dogecoin', name: 'Dogecoin', symbol: 'doge', price: 0.12, change24h: -2.56 },
        { id: 'avalanche', name: 'Avalanche', symbol: 'avax', price: 34.21, change24h: 4.32 }
      ];
      
      setCryptos(mockCryptos);
      setIsLoading(false);
    }, 1500);
  }, []);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Cryptocurrency Dashboard</h1>
        <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Current market prices and trends
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
              price={crypto.price}
              change24h={crypto.change24h}
              darkMode={darkMode}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default CryptoPage;