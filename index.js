const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

let conversationHistory = [];

app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        // Trim history BEFORE pushing to keep it within bounds
        if (conversationHistory.length >= 20) {
            conversationHistory = conversationHistory.slice(-18); // Leave room for new pair
        }

        conversationHistory.push({ role: "user", content: message });

        let reply;
        try {
            const response = await axios.post(
                'https://api.groq.com/openai/v1/chat/completions',
                {
                    model: "llama-3.3-70b-versatile",
                    messages: [
                        {
                            role: "system",
                            content: "You are JARVIS, Shubham's personal AI assistant. You are warm, helpful, and slightly sarcastic in a friendly way. Speak naturally like a clever friend. Be concise but engaging."
                        },
                        ...conversationHistory
                    ],
                    temperature: 0.75,
                    max_tokens: 600
                },
                {
                    headers: {
                        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                        'Content-Type': 'application/json'
                    }
                }
            );

            reply = response.data.choices[0].message.content;
        } catch (apiError) {
            // Roll back the user message so history stays clean
            conversationHistory.pop();
            console.error("Groq API Error:", JSON.stringify(apiError.response?.data, null, 2) ?? apiError.message);
            return res.status(500).json({ error: "Something went wrong while contacting the AI." });
        }

        conversationHistory.push({ role: "assistant", content: reply });
        res.json({ reply });

    } catch (error) {
        console.error("Unexpected Error:", error.message);
        res.status(500).json({ error: "Something went wrong." });
    }
});

app.post('/speak', async (req, res) => {
    try {
        const { text } = req.body;
        if (!text) return res.status(400).json({ error: "Text is required" });

        const response = await axios.post(
            'https://api.deepgram.com/v1/speak?model=aura-2-odysseus-en',
            { text },
            {
                headers: {
                    'Authorization': `Token ${process.env.DEEPGRAM_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                responseType: 'arraybuffer'
            }
        );

        res.set('Content-Type', 'audio/mp3');
        res.send(response.data);

    } catch (error) {
        console.error("TTS Error:", JSON.stringify(error.response?.data, null, 2) ?? error.message);
        res.status(500).json({ error: "Failed to generate speech." });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
