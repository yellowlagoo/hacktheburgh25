import React from 'react';

interface TimeframeSelectorProps {
  timeframes: string[];
  selectedTimeframe: string;
  onSelect: (timeframe: string) => void;
  darkMode: boolean;
}

const TimeframeSelector: React.FC<TimeframeSelectorProps> = ({ 
  timeframes, 
  selectedTimeframe, 
  onSelect,
  darkMode
}) => {
  return (
    <div className="flex space-x-2 mb-4">
      {timeframes.map((timeframe) => (
        <button
          key={timeframe}
          onClick={() => onSelect(timeframe)}
          className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
            selectedTimeframe === timeframe
              ? 'bg-indigo-600 text-white'
              : darkMode 
                ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          {timeframe}
        </button>
      ))}
    </div>
  );
};

export default TimeframeSelector;