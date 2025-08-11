import React, { useState } from 'react';
import { MessagesSquare, Loader2 } from 'lucide-react';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSendMessage = async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, isUser: true };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { text: data.answer, isUser: false };
      setMessages(prevMessages => [...prevMessages, botMessage]);
    } catch (e) {
      console.error('Error:', e);
      setError('An error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4 antialiased">
      <div className="w-full max-w-2xl bg-white rounded-xl shadow-lg flex flex-col h-[80vh]">
        <header className="bg-blue-600 rounded-t-xl text-white p-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <MessagesSquare size={24} />
            One Piece RAG Chatbot
          </h1>
        </header>

        <div className="flex-1 p-6 overflow-y-auto space-y-4">
          {messages.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-gray-500">
              <MessagesSquare size={48} className="mb-2" />
              <p>Start a conversation by typing a question.</p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div
              key={index}
              className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[70%] p-3 rounded-xl shadow-md ${
                  msg.isUser
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-gray-200 text-gray-800 rounded-bl-none'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-[70%] p-3 rounded-xl bg-gray-200 text-gray-800 flex items-center gap-2">
                <Loader2 size={20} className="animate-spin" />
                <span className="text-sm">Bot is thinking...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="text-red-500 text-center mt-4">
              {error}
            </div>
          )}
        </div>

        <div className="p-4 bg-gray-100 rounded-b-xl border-t border-gray-200 flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 p-3 rounded-xl border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Ask a question about the Straw Hat Pirates..."
            disabled={isLoading}
          />
          <button
            onClick={handleSendMessage}
            className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-400"
            disabled={isLoading}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
