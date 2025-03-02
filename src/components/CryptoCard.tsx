import React from 'react';
import { Link } from 'react-router-dom';
import { TrendingUp, TrendingDown, DollarSign, BarChart2, Clock } from 'lucide-react';

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
  const formatNumber = (num: number) => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return `$${num.toLocaleString()}`;
  };

  return (
    <Link to={`/crypto/${id}`}>
      <div
        className={`p-6 rounded-xl transition-all duration-300 hover:scale-[1.02] ${
          darkMode
            ? 'bg-gray-800 text-white hover:bg-gray-700'
            : 'bg-white text-gray-900 hover:bg-gray-50 shadow-lg'
        }`}
      >
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold">{name}</h2>
            <p className={`text-sm uppercase mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {symbol}
            </p>
          </div>
          <div className={`flex items-center ${change24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {change24h >= 0 ? <TrendingUp size={24} /> : <TrendingDown size={24} />}
            <span className="ml-1 font-semibold">{change24h.toFixed(2)}%</span>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center">
            <DollarSign size={20} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
            <span className="text-2xl font-bold ml-1">{price.toLocaleString()}</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Market Cap</p>
              <p className="font-semibold">{formatNumber(marketCap)}</p>
            </div>
            <div>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>24h Volume</p>
              <p className="font-semibold">{formatNumber(volume)}</p>
            </div>
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex items-center">
                  <Clock size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
                  <p className={`text-sm ml-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>1h</p>
                </div>
                <p className={`text-sm font-semibold ${change1h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {change1h >= 0 ? '↑' : '↓'} {Math.abs(change1h).toFixed(2)}%
                </p>
              </div>
              <div>
                <div className="flex items-center">
                  <BarChart2 size={16} className={darkMode ? 'text-gray-400' : 'text-gray-500'} />
                  <p className={`text-sm ml-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Range</p>
                </div>
                <p className="text-sm font-semibold">
                  ${low24h.toLocaleString()} - ${high24h.toLocaleString()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Link>
  );
};

export default CryptoCard;