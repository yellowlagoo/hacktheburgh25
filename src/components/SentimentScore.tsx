import React from 'react';

interface SentimentScoreProps {
  score: number;
  darkMode: boolean;
}

const SentimentScore: React.FC<SentimentScoreProps> = ({ score, darkMode }) => {
  const getScoreColor = () => {
    if (score >= 0.7) {
      return 'text-green-500';
    } else if (score <= 0.4 && score > 0) {
      return 'text-red-500';
    } else {
      return darkMode ? 'text-white' : 'text-gray-800';
    }
  };

  // Convert score to percentage for display
  const displayScore = Math.round(score * 100);

  return (
    <div className="fixed top-4 left-4 z-50">
      <div className={`p-2 rounded-md ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
        <p className="text-sm font-medium">Sentiment Score</p>
        <p className={`text-2xl font-bold ${getScoreColor()}`}>
          {score > 0 ? `${displayScore}%` : '-'}
        </p>
      </div>
    </div>
  );
};

export default SentimentScore;