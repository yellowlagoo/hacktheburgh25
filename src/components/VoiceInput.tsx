import React, { useState, useEffect } from 'react';
import { Mic, MicOff, Volume2, VolumeX } from 'lucide-react';

interface VoiceInputProps {
  onVoiceInput: (text: string) => void;
  isListening: boolean;
  setIsListening: (isListening: boolean) => void;
  darkMode: boolean;
  lastResponse: string;
}

const VoiceInput: React.FC<VoiceInputProps> = ({
  onVoiceInput,
  isListening,
  setIsListening,
  darkMode,
  lastResponse,
}) => {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [speechSupported] = useState('SpeechRecognition' in window || 'webkitSpeechRecognition' in window);
  const [synthesis] = useState<SpeechSynthesis | null>(window.speechSynthesis);

  useEffect(() => {
    // Initialize speech recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onresult = (event: any) => {
        const transcript = Array.from(event.results)
          .map((result: any) => result[0])
          .map((result: any) => result.transcript)
          .join('');

        if (event.results[0].isFinal) {
          onVoiceInput(transcript);
          setIsListening(false);
        }
      };

      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };

      if (isListening) {
        recognition.start();
      }

      return () => {
        recognition.stop();
      };
    }
  }, [isListening, onVoiceInput, setIsListening]);

  const toggleListening = () => {
    if (!speechSupported) {
      alert('Speech recognition is not supported in your browser.');
      return;
    }
    setIsListening(!isListening);
  };

  const speakResponse = () => {
    if (!synthesis) {
      alert('Speech synthesis is not supported in your browser.');
      return;
    }

    if (isSpeaking) {
      synthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(lastResponse);
    utterance.onend = () => setIsSpeaking(false);
    setIsSpeaking(true);
    synthesis.speak(utterance);
  };

  return (
    <div className="flex space-x-2">
      <button
        onClick={toggleListening}
        className={`p-2 rounded-full transition-colors ${
          darkMode
            ? isListening
              ? 'bg-red-600 text-white'
              : 'bg-gray-700 text-white hover:bg-gray-600'
            : isListening
            ? 'bg-red-500 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        }`}
        title={isListening ? 'Stop listening' : 'Start voice input'}
      >
        {isListening ? <MicOff size={20} /> : <Mic size={20} />}
      </button>
      
      <button
        onClick={speakResponse}
        disabled={!lastResponse}
        className={`p-2 rounded-full transition-colors ${
          darkMode
            ? isSpeaking
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-700 text-white hover:bg-gray-600'
            : isSpeaking
            ? 'bg-indigo-500 text-white'
            : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
        } ${!lastResponse && 'opacity-50 cursor-not-allowed'}`}
        title={isSpeaking ? 'Stop speaking' : 'Speak response'}
      >
        {isSpeaking ? <VolumeX size={20} /> : <Volume2 size={20} />}
      </button>
    </div>
  );
};

export default VoiceInput; 