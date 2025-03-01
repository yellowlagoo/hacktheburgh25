import React, { useState } from 'react';
import ChatInput from '../components/ChatInput';
import VoiceInput from '../components/VoiceInput';
import AIBall from '../components/AIBall';
import SentimentScore from '../components/SentimentScore';

interface AIAssistantProps {
  darkMode: boolean;
}

const AIAssistant: React.FC<AIAssistantProps> = ({ darkMode }) => {
  const [prompt, setPrompt] = useState('');
  const [fullResponse, setFullResponse] = useState('');
  const [sentimentScore, setSentimentScore] = useState(0);
  const [sentimentContext, setSentimentContext] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const handleSubmit = async (message: string) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('http://localhost:5050/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command: message,
        }),
      });

      const data = await response.json();
      console.log('Response data:', data); // Debug log
      
      if (data.error) {
        console.error('Error:', data.error);
        setPrompt('Sorry, there was an error processing your request.');
        setFullResponse('Please try again later.');
        return;
      }

      // Get sentiment data
      const sentiment = data.Sentiment || {};
      setSentimentScore(sentiment.score || 0);
      setSentimentContext(sentiment.context || '');

      // Get the first section as the summary
      const marketAnalysis = data['Market Analysis'] || '';
      console.log('Market Analysis:', marketAnalysis); // Debug log
      
      const sections = marketAnalysis.split('###');
      const summary = sections[1] ? sections[1].split('\n')[1] : marketAnalysis.split('\n')[0];
      
      setPrompt(summary || 'Analysis received');
      setFullResponse(marketAnalysis);
    } catch (error) {
      console.error('Error processing command:', error);
      setPrompt('Sorry, there was an error processing your request.');
      setFullResponse('Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceInput = (text: string) => {
    console.log('Voice input received:', text);
    handleSubmit(text);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)]">
      <SentimentScore score={sentimentScore} darkMode={darkMode} />
      
      <div className="text-center mb-8">
        <h1 className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>AI Assistant</h1>
        <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Ask me about cryptocurrency trends and market analysis
        </p>
      </div>
      
      <AIBall 
        prompt={prompt} 
        fullResponse={fullResponse} 
        sentimentScore={sentimentScore}
        sentimentContext={sentimentContext}
        isLoading={isLoading}
        darkMode={darkMode}
      />
      
      <div className="w-full max-w-2xl mx-auto mt-8 space-y-4">
        <div className="flex justify-center">
          <VoiceInput
            onVoiceInput={handleVoiceInput}
            isListening={isListening}
            setIsListening={setIsListening}
            darkMode={darkMode}
            lastResponse={fullResponse}
          />
        </div>
        
        <ChatInput 
          onSubmit={handleSubmit} 
          isLoading={isLoading}
          darkMode={darkMode}
        />
      </div>
    </div>
  );
};

export default AIAssistant;