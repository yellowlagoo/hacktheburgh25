import React, { useState, useEffect } from 'react';
import { Volume2 } from 'lucide-react';

interface AIBallProps {
  prompt: string;
  fullResponse: string;
  sentimentScore: number;
  isLoading: boolean;
  darkMode: boolean;
  sentimentContext?: string;
}

const AIBall: React.FC<AIBallProps> = ({ prompt, fullResponse, sentimentScore, isLoading, darkMode, sentimentContext }) => {
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

  const formatSection = (section: string) => {
    const lines = section.trim().split('\n');
    const title = lines[0].replace(/^[#\s]*/, '').replace(/^###\s*/, '').trim();
    const content = lines.slice(1).join('\n');
    return { title, content };
  };

  const formatContent = (content: string) => {
    const lines = content.trim().split('\n');
    return lines.map((line, index) => {
      const uniqueKey = `line-${index}-${line.substring(0, 20)}`;
      
      // Remove markdown headers and clean the line
      const cleanedLine = line.replace(/^#+\s*/, '').trim();
      
      if (!cleanedLine) {
        return null;
      }
      
      // Handle subsections (marked with **)
      if (cleanedLine.startsWith('**')) {
        return <h3 key={uniqueKey} className="text-lg font-semibold mt-4 mb-2">{cleanedLine.replace(/\*\*/g, '')}</h3>;
      }
      
      // Handle bullet points
      if (cleanedLine.startsWith('•') || cleanedLine.startsWith('-')) {
        const bulletContent = cleanedLine.startsWith('•') ? cleanedLine.substring(1) : cleanedLine.substring(2);
        const [label, value] = bulletContent.split(':').map(s => s.trim());
        
        if (value) {
          return (
            <div key={uniqueKey} className="flex justify-between items-center py-1">
              <span className="text-gray-500 dark:text-gray-400">{label}</span>
              <span className="font-medium">{value}</span>
            </div>
          );
        }
        return <li key={uniqueKey} className="ml-4 mb-2">{bulletContent.trim()}</li>;
      }
      
      // Handle horizontal rules
      if (cleanedLine.startsWith('---')) {
        return <hr key={uniqueKey} className="my-4 border-gray-200 dark:border-gray-700" />;
      }
      
      // Handle section headers (previously marked with ##)
      if (line.startsWith('#')) {
        return <h3 key={uniqueKey} className="text-lg font-semibold mt-6 mb-3">{cleanedLine}</h3>;
      }
      
      // Regular paragraph
      return <p key={uniqueKey} className="mb-2">{cleanedLine}</p>;
    }).filter(Boolean); // Remove null elements
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
          <div className="text-sm overflow-auto max-h-full">
            <p className="font-semibold mb-2">Market Analysis</p>
            <p className={`${sentimentScore >= 0.6 ? 'text-green-500' : sentimentScore <= 0.4 ? 'text-red-500' : 'text-gray-500'}`}>
              {prompt || 'Ask me about crypto market analysis'}
            </p>
            {sentimentContext && (
              <p className="mt-2 text-xs opacity-75">
                {sentimentContext}
              </p>
            )}
          </div>
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

      {showFullResponse && fullResponse && (
        <div className={`mt-4 p-6 rounded-lg max-w-4xl w-full ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800 shadow-lg'}`}>
          {fullResponse.split('###').map((section, sectionIndex) => {
            if (!section.trim()) return null;
            const { title, content } = formatSection(section);
            const sectionKey = `section-${sectionIndex}-${title}`;
            return (
              <div key={sectionKey} className="mb-8 last:mb-0">
                {title && (
                  <h2 className="text-xl font-bold mb-4 pb-2 border-b border-gray-200 dark:border-gray-700">
                    {title}
                  </h2>
                )}
                <div className="space-y-2">
                  {formatContent(content)}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default AIBall;