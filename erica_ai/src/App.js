import React, { useState } from "react";
import axios from "axios";

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
    <div style={{ maxWidth: "600px", margin: "50px auto", fontFamily: "sans-serif" }}>
      <h1>Erica AI Tutor</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask Erica a question..."
          style={{ width: "100%", padding: "10px", fontSize: "16px" }}
        />
        <button type="submit" style={{ marginTop: "10px", padding: "10px 20px", fontSize: "16px" }}>
          Ask
        </button>
      </form>

      {loading && <p>Loading answer...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {answer && (
        <div style={{ background: "#f2f2f2", padding: "15px", borderRadius: "5px" }}>
          <h3>Erica's Answer:</h3>
          <div className="text-gray-700 whitespace-pre-wrap">
            {answer}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
