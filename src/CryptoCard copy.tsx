import React from 'react';
import { Link } from 'react-router-dom';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface CryptoCardProps {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  darkMode: boolean;
}

const CryptoCard: React.FC<CryptoCardProps> = ({ id, name, symbol, price, change24h, darkMode }) => {
  const isPositive = change24h >= 0;

  return (
    <Link to={`/crypto/${id}`}>
      <div className={`p-4 rounded-lg transition-transform hover:scale-105 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white shadow-md text-gray-800'}`}>
        <div className="flex justify-between items-center">
          <div>
            <h3 className="font-bold">{name}</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">{symbol.toUpperCase()}</p>
          </div>
          <div className={`flex items-center ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
            {isPositive ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
            <span className="ml-1">{change24h.toFixed(2)}%</span>
          </div>
        </div>
        <p className="mt-2 text-lg font-semibold">${price.toLocaleString()}</p>
      </div>
    </Link>
  );
};

export default CryptoCard;