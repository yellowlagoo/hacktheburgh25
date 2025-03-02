import React, { useState, KeyboardEvent } from 'react';
import { Command } from 'lucide-react';

interface CommandInputProps {
  darkMode: boolean;
  onSubmit: (command: string) => void;
  isLoading: boolean;
}

interface MetricsData {
  current_price: number;
  market_cap: number;
  volume: number;
  high_24h: number;
  low_24h: number;
  price_change: number;
  price_change_7d: number;
  price_change_30d: number;
  ath: number;
  ath_change_percentage: number;
  atl: number;
  atl_change_percentage: number;
}

interface AnalysisResponse {
  'Market Analysis': string;
  'Key Metrics': MetricsData;
  'Focus Areas': string[];
}

const CommandInput: React.FC<CommandInputProps> = ({ darkMode, onSubmit, isLoading }) => {
  const [command, setCommand] = useState('');
  const [response, setResponse] = useState<AnalysisResponse | null>(null);

  const handleKeyDown = async (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isLoading && command.trim()) {
      try {
        const result = await fetch('http://localhost:5050/command', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            command: command,
            crypto_id: 'bitcoin', // You can make this dynamic based on the current page
          }),
        });

        const data = await result.json();
        if (data.error) {
          console.error('Error:', data.error);
          return;
        }

        setResponse(data);
        onSubmit(command);
        setCommand('');
      } catch (error) {
        console.error('Error processing command:', error);
      }
    }
  };

  const formatMetric = (value: number, isPercentage = false) => {
    return isPercentage
      ? `${value.toFixed(2)}%`
      : `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  return (
    <div className={`space-y-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
      <div className="relative">
        <Command
          size={20}
          className={`absolute left-3 top-1/2 transform -translate-y-1/2 ${
            darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}
        />
        <input
          type="text"
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about market trends, price predictions, or trading insights..."
          className={`w-full pl-10 pr-4 py-2 rounded-lg border ${
            darkMode
              ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400'
              : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
          } focus:outline-none focus:ring-2 focus:ring-indigo-500`}
          disabled={isLoading}
        />
        {isLoading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
          </div>
        )}
      </div>

      {response && (
        <div className={`mt-4 space-y-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
            <h3 className="text-lg font-semibold mb-2">Market Analysis</h3>
            <div className="prose max-w-none dark:prose-invert">
              {response['Market Analysis'].split('\n').map((paragraph, index) => (
                <p key={index} className="mb-2">{paragraph}</p>
              ))}
            </div>
          </div>

          <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
            <h3 className="text-lg font-semibold mb-2">Key Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Current Price</p>
                <p className="font-semibold">{formatMetric(response['Key Metrics'].current_price)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">24h Change</p>
                <p className={`font-semibold ${response['Key Metrics'].price_change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {formatMetric(response['Key Metrics'].price_change, true)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">7d Change</p>
                <p className={`font-semibold ${response['Key Metrics'].price_change_7d >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {formatMetric(response['Key Metrics'].price_change_7d, true)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Market Cap</p>
                <p className="font-semibold">{formatMetric(response['Key Metrics'].market_cap)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">24h Volume</p>
                <p className="font-semibold">{formatMetric(response['Key Metrics'].volume)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">ATH</p>
                <p className="font-semibold">{formatMetric(response['Key Metrics'].ath)}</p>
              </div>
            </div>
          </div>

          <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
            <h3 className="text-lg font-semibold mb-2">Analysis Focus</h3>
            <div className="flex flex-wrap gap-2">
              {response['Focus Areas'].map((area, index) => (
                <span
                  key={index}
                  className={`px-3 py-1 rounded-full text-sm ${
                    darkMode
                      ? 'bg-gray-700 text-gray-200'
                      : 'bg-gray-200 text-gray-800'
                  }`}
                >
                  {area}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CommandInput; 