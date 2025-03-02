import React, { useState, useEffect } from 'react';
import { Volume2 } from 'lucide-react';

interface AIBallProps {
  prompt: string;
  fullResponse: string;
  sentimentScore: number;
  isLoading: boolean;
  darkMode: boolean;
}

const AIBall: React.FC<AIBallProps> = ({ prompt, fullResponse, sentimentScore, isLoading, darkMode }) => {
  const [showFullResponse, setShowFullResponse] = useState(false);
  const [isShaking, setIsShaking] = useState(false);

  useEffect(() => {
    if (prompt && !isLoading) {
      setIsShaking(true);
      const timer = setTimeout(() => {
        setIsShaking(false);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [prompt, isLoading]);

  const getBallColor = () => {
    if (sentimentScore >= 7) {
      return 'shadow-[0_0_30px_15px_rgba(34,197,94,0.6)]';
    } else if (sentimentScore <= 4 && sentimentScore > 0) {
      return 'shadow-[0_0_30px_15px_rgba(239,68,68,0.6)]';
    } else {
      return '';
    }
  };

  const speakResponse = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(fullResponse);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center mt-8">
      <div
        className={`relative w-64 h-64 rounded-full flex items-center justify-center text-center p-6 transition-all duration-300 ${
          darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'
        } ${getBallColor()} ${isShaking ? 'animate-[shake_0.5s_ease-in-out_infinite]' : ''}`}
        style={{ boxShadow: sentimentScore > 0 ? '' : 'none' }}
      >
        {isLoading ? (
          <div className="flex space-x-2">
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_infinite]"></div>
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.4s_infinite]"></div>
            <div className="w-3 h-3 rounded-full bg-gray-400 dark:bg-gray-600 animate-[pulse_1.2s_ease-in-out_0.8s_infinite]"></div>
          </div>
        ) : (
          <p className="text-sm overflow-auto max-h-full">{prompt}</p>
        )}
      </div>

      {prompt && !isLoading && (
        <div className="mt-4 flex flex-col items-center">
          <button
            onClick={() => setShowFullResponse(!showFullResponse)}
            className="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
          >
            {showFullResponse ? 'Hide Details' : 'Read More Info'}
          </button>
          
          <button
            onClick={speakResponse}
            className="mt-2 flex items-center px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors"
          >
            <Volume2 size={16} className="mr-2" />
            Listen
          </button>
        </div>
      )}

      {showFullResponse && (
        <div className={`mt-4 p-4 rounded-md max-w-2xl ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800 shadow-md'}`}>
          <p>{fullResponse}</p>
        </div>
      )}
    </div>
  );
};

export default AIBall;