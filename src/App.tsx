import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Moon, Sun } from "lucide-react";
import Navbar from "./components/Navbar";
import AIAssistant from "./pages/AIAssistant";
import CryptoPage from "./pages/CryptoPage";
import CryptoDetailPage from "./pages/CryptoDetailPage";

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle("dark");
  };

  return (
    <Router>
      <div className={`min-h-screen ${isDarkMode ? "dark" : ""}`}>
        <div className="dark:bg-gray-900 min-h-screen">
          <nav className="bg-white dark:bg-gray-800 border-b dark:border-gray-700">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex">
                  <Navbar />
                </div>
                <button
                  onClick={toggleDarkMode}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  {isDarkMode ? (
                    <Sun className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                  ) : (
                    <Moon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                  )}
                </button>
              </div>
            </div>
          </nav>
          <Routes>
            <Route path="/" element={<CryptoPage darkMode={isDarkMode} />} />
            <Route path="/crypto/:id" element={<CryptoDetailPage darkMode={isDarkMode} />} />
            <Route path="/ai-assistant" element={<AIAssistant darkMode={isDarkMode} />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;