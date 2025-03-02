import React, { useState } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSubmit: (message: string) => void;
  isLoading: boolean;
  darkMode: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSubmit, isLoading, darkMode }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSubmit(message);
      setMessage('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mt-8 max-w-2xl mx-auto">
      <div className={`flex items-center p-2 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white shadow-md'}`}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask the AI assistant..."
          className={`flex-grow p-2 outline-none ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'}`}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !message.trim()}
          className={`ml-2 p-2 rounded-full ${
            isLoading || !message.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          } transition-colors`}
        >
          <Send size={20} />
        </button>
      </div>
    </form>
  );
};

export default ChatInput;