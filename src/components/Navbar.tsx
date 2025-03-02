import React from "react";
import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <div className="flex space-x-8">
      <Link
        to="/"
        className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white px-3 py-2 text-sm font-medium"
      >
        Crypto Dashboard
      </Link>
      <Link
        to="/ai-assistant"
        className="text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white px-3 py-2 text-sm font-medium"
      >
        AI Assistant
      </Link>
    </div>
  );
};

export default Navbar; 