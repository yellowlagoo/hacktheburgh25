import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bot, BarChart2 } from 'lucide-react';

const Navbar: React.FC = () => {
  const location = useLocation();

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <Bot className="h-8 w-8 text-indigo-600 dark:text-indigo-400" />
              <span className="ml-2 text-xl font-bold text-gray-800 dark:text-white">AI Assistant</span>
            </Link>
          </div>
          <div className="flex space-x-4">
            <Link
              to="/"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/'
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-indigo-100 dark:hover:bg-gray-700'
              }`}
            >
              Assistant
            </Link>
            <Link
              to="/crypto"
              className={`px-3 py-2 rounded-md text-sm font-medium flex items-center ${
                location.pathname.includes('/crypto')
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-700 dark:text-gray-300 hover:bg-indigo-100 dark:hover:bg-gray-700'
              }`}
            >
              <BarChart2 className="h-4 w-4 mr-1" />
              Crypto
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;