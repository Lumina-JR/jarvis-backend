const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// In-memory conversation history
let conversationHistory = [];

// Groq API Configuration
const GROQ_API_KEY = process.env.GROQ_API_KEY;

app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;

        if (!message) {
            return res.status(400).json({ error: "Message is required" });
        }

        // Add user message to history
        conversationHistory.push({ role: "user", content: message });

        const response = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
            model: "llama-3.1-70b-versatile",
            messages: [
                { role: "system", content: "You are JARVIS, a highly intelligent, witty, and loyal AI assistant inspired by Iron Man. Speak in a sophisticated, slightly sarcastic but respectful tone." },
                ...conversationHistory
            ],
            temperature: 0.7,
            max_tokens: 500
        }, {
            headers: {
                'Authorization': `Bearer ${GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        const reply = response.data.choices[0].message.content;

        // Add Jarvis reply to history
        conversationHistory.push({ role: "assistant", content: reply });

        // Keep only last 10 messages to save memory
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

        res.json({ reply });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Something went wrong" });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
