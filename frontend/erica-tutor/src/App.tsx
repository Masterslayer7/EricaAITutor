import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, User, Bot, RefreshCw, Eraser } from 'lucide-react';
import axios from 'axios';

// --- Types ---
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  isError?: boolean; // Optional property to indicate error messages
}

// --- Hooks ---

// Simple hook to load the 'marked' library from CDN for reliable markdown parsing without build steps
const useMarked = () => {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if ((window as any).marked) {
      setIsLoaded(true);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.async = true;
    script.onload = () => setIsLoaded(true);
    document.body.appendChild(script);

    return () => {
      // Clean up script if needed, though usually fine to leave for single page apps
    };
  }, []);

  return isLoaded;
};

// --- Components ---

// Markdown Renderer Component
const MarkdownMessage = ({ content }: { content: string }) => {
  const isMarkedLoaded = useMarked();
  const [htmlContent, setHtmlContent] = useState('');

  useEffect(() => {
    if (isMarkedLoaded && (window as any).marked) {
      // Configure marked for security (sanitize) and breaks
      (window as any).marked.setOptions({
        gfm: true,
        breaks: true,
      });
      setHtmlContent((window as any).marked.parse(content));
    } else {
      setHtmlContent(content); // Fallback to plain text while loading
    }
  }, [content, isMarkedLoaded]);

  if (!isMarkedLoaded) return <div className="animate-pulse">Loading renderer...</div>;

  return (
    <div 
      className="prose prose-sm md:prose-base prose-slate max-w-none dark:prose-invert prose-p:leading-relaxed prose-pre:bg-slate-800 prose-pre:text-slate-50"
      dangerouslySetInnerHTML={{ __html: htmlContent }} 
    />
  );
};

// 2. Main App Component
export default function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "# Hello, I'm Erica.\nI'm here to help you learn. You can ask me questions, and I'll reply with **rich text**, `code snippets`, or organized lists.\n\n* **Concept Explanations**\n* **Code Review**\n* **Study Plans**\n\nHow can I help you today?",
      timestamp: Date.now(),
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMsg]);
    const currentQuestion = input;
    setInput('');
    setIsTyping(true);

    // MOCK RESPONSE - Replace this with your actual API call to Erica
    try {
      const response = await axios.post(
        "http://localhost:5000/ask",
        { question: currentQuestion }, // Payload matches your old code
        { responseType: "text" }
      );

      // 3. Add Assistant Response
      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data, // Using the data directly as you requested
        timestamp: Date.now(),
      };
      
      setMessages(prev => [...prev, botMsg]);

    } catch (err) {
      console.error(err);
      
      // 4. Handle Errors Gracefully in Chat
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "**Error:** Failed to connect to the backend. Is your Flask server running on port 5000?",
        timestamp: Date.now(),
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
      setMessages([messages[0]]); // Keep welcome message
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-900 overflow-hidden">
      
      {/* Header */}
      <header className="flex-none bg-white/80 backdrop-blur-md border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-200">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
              Erica Tutor
            </h1>
            <p className="text-xs text-slate-500 font-medium">Personalized Learning Assistant</p>
          </div>
        </div>
        <button 
          onClick={clearChat}
          className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors"
          title="Clear Chat"
        >
          <Eraser className="w-5 h-5" />
        </button>
      </header>

      {/* Chat Area */}
      <main 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 scroll-smooth bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-fixed"
      >
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div 
              key={msg.id} 
              className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              {/* Avatar */}
              <div className={`flex-none w-8 h-8 md:w-10 md:h-10 rounded-full flex items-center justify-center shadow-sm mt-1 ${
                msg.role === 'user' 
                  ? 'bg-slate-800 text-white' 
                  : 'bg-white text-indigo-600 border border-indigo-100'
              }`}>
                {msg.role === 'user' ? <User className="w-5 h-5" /> : <Bot className="w-6 h-6" />}
              </div>

              {/* Message Bubble */}
              <div className={`flex-1 max-w-[85%] md:max-w-[80%] rounded-2xl px-5 py-4 shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-slate-800 text-white rounded-tr-none' 
                  : 'bg-white border border-slate-100 rounded-tl-none shadow-md'
              }`}>
                {msg.role === 'assistant' ? (
                  <MarkdownMessage content={msg.content} />
                ) : (
                  <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
                )}
              </div>
            </div>
          ))}

          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex gap-4">
              <div className="flex-none w-8 h-8 md:w-10 md:h-10 rounded-full bg-white border border-indigo-100 flex items-center justify-center shadow-sm mt-1">
                <RefreshCw className="w-5 h-5 text-indigo-500 animate-spin" />
              </div>
              <div className="bg-white border border-slate-100 rounded-2xl rounded-tl-none px-4 py-3 shadow-md flex items-center gap-1">
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-100"></span>
                <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce delay-200"></span>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Input Area */}
      <footer className="flex-none bg-white border-t border-slate-200 p-4 md:p-6">
        <div className="max-w-3xl mx-auto relative">
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 to-purple-500/5 rounded-2xl -m-2 blur-xl pointer-events-none"></div>
          <div className="relative flex items-end gap-2 bg-white rounded-xl border border-slate-300 focus-within:border-indigo-500 focus-within:ring-4 focus-within:ring-indigo-500/10 shadow-sm transition-all p-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask Erica a question..."
              className="flex-1 max-h-32 min-h-[50px] bg-transparent border-none focus:ring-0 p-3 text-slate-800 placeholder:text-slate-400 resize-none font-medium"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isTyping}
              className="flex-none p-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white rounded-lg transition-all active:scale-95 shadow-md shadow-indigo-200"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <div className="text-center mt-2">
            <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
              Markdown Supported
            </p>
          </div>
        </div>
      </footer>

    </div>
  );
}