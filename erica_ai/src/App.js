import React, { useState } from "react";
import axios from "axios";
import Header from "./components/Header";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setAnswer("");

    try {
      const response = await axios.post(
        "http://localhost:5000/ask",
        { question },
        { responseType: "text" }
      );
      setAnswer(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to get an answer. Check console for details.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: "black", minHeight: "100vh", color: "white" }}>
      
      <nav> 
        <a href="https://github.com/Masterslayer7/EricaAITutor" target="_blank" rel="noopener noreferrer" style={{ color: "white", textDecoration: "underline", padding: "10px", display: "inline-block" }} onMouseOver={(e) => (e.target.style.backgroundColor = "#333")} onMouseOut={(e) => (e.target.style.backgroundColor = "#1f1f1f")}>
          Erica AI Tutor - GitHub
        </a>
      </nav>

      {/* Centered content */}
      <div style={{ maxWidth: "600px", margin: "0 auto", paddingTop: "20px" }}>
        <h1>Erica AI Tutor</h1>
        <h3>By: Yug Patel & Eric Zhang</h3>

        <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
          <div style={{ display: "flex", gap: "10px" }}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask Erica a question..."
              style={{ flex: 1, padding: "10px", fontSize: "16px" }}
            />
            <button
              type="submit"
              style={{ padding: "10px 20px", fontSize: "16px" }}
            >
              Ask
            </button>
          </div>
        </form>

        {loading && <p>Loading answer...</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}
        {answer && (
          <div
            className="prose prose-lg max-w-none bg-white text-black shadow-md rounded-lg p-6"
            dangerouslySetInnerHTML={{ __html: answer }}
          />
        )}
      </div>
    </div>
  );
}

export default App;
