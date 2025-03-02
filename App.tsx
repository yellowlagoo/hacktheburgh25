import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Moon, Sun } from 'lucide-react';
import Navbar from './components/Navbar';
import AIAssistant from './pages/AIAssistant';
import CryptoPage from './pages/CryptoPage';
import CryptoDetailPage from './pages/CryptoDetailPage';

function App() {
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'dark bg-gray-900' : 'bg-gray-100'}`}>
      <Router>
        <div className="fixed top-4 right-4 z-50">
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-full ${darkMode ? 'bg-gray-700 text-yellow-300' : 'bg-white text-gray-800 shadow-md'}`}
            aria-label="Toggle dark mode"
          >
            {darkMode ? <Sun size={20} /> : <Moon size={20} />}
          </button>
        </div>
        <Navbar />
        <div className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<AIAssistant darkMode={darkMode} />} />
            <Route path="/crypto" element={<CryptoPage darkMode={darkMode} />} />
            <Route path="/crypto/:id" element={<CryptoDetailPage darkMode={darkMode} />} />
          </Routes>
        </div>
      </Router>
    </div>
  );
}

export default App;