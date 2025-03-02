import React from 'react';
import { Link } from 'react-router-dom';

interface CryptoCardProps {
  id: string;
  name: string;
  symbol: string;
  price: number;
  change24h: number;
  change1h: number;
  marketCap: number;
  volume: number;
  high24h: number;
  low24h: number;
  darkMode: boolean;
}

const CryptoCard: React.FC<CryptoCardProps> = ({
  id,
  name,
  symbol,
  price,
  change24h,
  change1h,
  marketCap,
  volume,
  high24h,
  low24h,
  darkMode,
}) => {
  return (
    <Link to={`/crypto/${id}`}>
      <div
        className={`p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
        }`}
      >
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-xl font-semibold">{name}</h2>
            <p className={`text-sm uppercase ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {symbol}
            </p>
          </div>
        </div>
        <div className="space-y-2">
          <p className="text-2xl font-bold">${price.toLocaleString()}</p>
          <div className="flex space-x-4">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">1h</p>
              <p
                className={`text-sm ${
                  change1h >= 0 ? 'text-green-500' : 'text-red-500'
                }`}
              >
                {change1h >= 0 ? '↑' : '↓'} {Math.abs(change1h).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">24h</p>
              <p
                className={`text-sm ${
                  change24h >= 0 ? 'text-green-500' : 'text-red-500'
                }`}
              >
                {change24h >= 0 ? '↑' : '↓'} {Math.abs(change24h).toFixed(2)}%
              </p>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <p className="text-gray-500 dark:text-gray-400">24h High</p>
                <p>${high24h.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-gray-500 dark:text-gray-400">24h Low</p>
                <p>${low24h.toLocaleString()}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
};

export default CryptoCard;