const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

let conversationHistory = [];

console.log("GROQ_API_KEY loaded:", process.env.GROQ_API_KEY ? "YES (length: " + process.env.GROQ_API_KEY.length + ")" : "NO");

app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;

        if (!message) return res.status(400).json({ error: "Message is required" });

        conversationHistory.push({ role: "user", content: message });

        if (!process.env.GROQ_API_KEY) {
            return res.status(500).json({ error: "GROQ_API_KEY is not configured" });
        }

        const response = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
    model: "llama-3.3-70b-versatile",     // ← Updated here
    messages: [
        { 
            role: "system", 
            content: "You are JARVIS, a highly intelligent, witty, and loyal AI assistant. You belong to Shubham. Speak in a sophisticated but natural tone. Be concise and direct. Never mention Tony Stark." 
        },
        ...conversationHistory
    ],
    temperature: 0.7,
    max_tokens: 500
}, {
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        const reply = response.data.choices[0].message.content;
        conversationHistory.push({ role: "assistant", content: reply });

        if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);

        res.json({ reply });

    } catch (error) {
        console.error("Error:", error.response?.data || error.message);
        res.status(500).json({ 
            error: "Something went wrong",
            details: error.response?.data || error.message 
        });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
