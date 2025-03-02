import React, { useState } from 'react';
import ChatInput from '../components/ChatInput';
import AIBall from '../components/AIBall';
import SentimentScore from '../components/SentimentScore';

interface AIAssistantProps {
  darkMode: boolean;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ darkMode }) => {
  const [prompt, setPrompt] = useState('');
  const [fullResponse, setFullResponse] = useState('');
  const [sentimentScore, setSentimentScore] = useState(0);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = (message: string) => {
    setIsLoading(true);
    
    // Simulate AI processing
    setTimeout(() => {
      // Generate a mock response based on the user's message
      const response = generateMockResponse(message);
      
      // Generate a mock sentiment score between 1 and 10
      const score = Math.floor(Math.random() * 10) + 1;
      
      setPrompt(response.summary);
      setFullResponse(response.full);
      setSentimentScore(score);
      setIsLoading(false);
    }, 2000);
  };

  const generateMockResponse = (message: string) => {
    // This is a simple mock response generator
    // In a real application, this would be replaced with an actual AI service
    const topics = {
      weather: {
        summary: "Today's weather forecast shows mild temperatures with a chance of rain.",
        full: "Today's weather forecast indicates mild temperatures ranging from 65°F to 75°F. There's a 40% chance of rain in the afternoon, with winds from the southwest at 5-10 mph. The humidity is expected to be around 65%. Tomorrow should be clearer with slightly higher temperatures."
      },
      crypto: {
        summary: "Bitcoin is showing strong upward momentum, while Ethereum remains stable.",
        full: "Bitcoin has shown significant upward momentum over the past 24 hours, with a 5.2% increase bringing it to $58,432. Trading volume is up 15% compared to the weekly average. Ethereum has remained relatively stable at around $2,850, with only minor fluctuations. Analysts suggest this divergence may be due to recent regulatory news affecting the markets differently."
      },
      help: {
        summary: "I can assist with information, answer questions, or provide guidance on various topics.",
        full: "I'm your AI assistant, designed to help with a wide range of tasks. I can provide information on topics like weather, news, or cryptocurrency prices. I can answer questions about general knowledge, help with simple calculations, or offer guidance on various subjects. Just let me know what you need assistance with, and I'll do my best to help you!"
      }
    };

    // Default response if no keywords match
    let response = {
      summary: "I've processed your request and have some information for you.",
      full: "I've analyzed your request and prepared some information that might be helpful. Your query was about: '" + message + "'. If you need more specific information, please provide additional details about what you're looking for, and I'll be happy to assist further."
    };

    // Check for keywords in the message
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes('weather')) {
      response = topics.weather;
    } else if (lowerMessage.includes('crypto') || lowerMessage.includes('bitcoin') || lowerMessage.includes('ethereum')) {
      response = topics.crypto;
    } else if (lowerMessage.includes('help') || lowerMessage.includes('assist')) {
      response = topics.help;
    }

    return response;
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)]">
      <SentimentScore score={sentimentScore} darkMode={darkMode} />
      
      <div className="text-center mb-8">
        <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>AI Assistant</h1>
        <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Ask me anything and I'll do my best to help you
        </p>
      </div>
      
      <AIBall 
        prompt={prompt} 
        fullResponse={fullResponse} 
        sentimentScore={sentimentScore} 
        isLoading={isLoading}
        darkMode={darkMode}
      />
      
      <ChatInput 
        onSubmit={handleSubmit} 
        isLoading={isLoading}
        darkMode={darkMode}
      />
    </div>
  );
};

export default AIAssistant;