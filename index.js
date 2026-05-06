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

        conversationHistory.push({ role: "user", content: message });

        const response = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
            model: "llama-3.3-70b-versatile",
            messages: [
                { 
                    role: "system", 
                    content: "You are JARVIS, Shubham's personal AI assistant. You are warm, helpful, and slightly sarcastic in a friendly way. Speak naturally like a clever friend. Be concise but engaging. You belong to Shubham." 
                },
                ...conversationHistory
            ],
            temperature: 0.75,
            max_tokens: 600
        }, {
            headers: {
                'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        const reply = response.data.choices[0].message.content;

        conversationHistory.push({ role: "assistant", content: reply });

        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

        res.json({ reply });

    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: "Something went wrong" });
    }
});

// New endpoint for Deepgram TTS
app.post('/speak', async (req, res) => {
    try {
        const { text } = req.body;
        if (!text) return res.status(400).json({ error: "Text is required" });

        const response = await axios.post(
            'https://api.deepgram.com/v1/speak?model=aura-2-odysseus-en',
            { text: text },
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
        console.error("TTS Error:", error.message);
        res.status(500).json({ error: "Failed to generate speech" });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running with Deepgram 🚀');
});

// Deepgram Transcribe Endpoint
app.post('/transcribe', async (req, res) => {
    try {
        const formData = new FormData();
        formData.append('file', req.files ? req.files.audio.data : req.body.audio, 'audio.webm');

        const response = await axios.post('https://api.deepgram.com/v1/listen?model=nova-2-general&smart_format=true', 
            formData, {
            headers: {
                'Authorization': `Token ${process.env.DEEPGRAM_API_KEY}`,
                ...formData.getHeaders()
            }
        });

        const transcript = response.data.results.channels[0 0].transcript;
        
        res.json({ text: transcript || "" });

    } catch (error) {
        console.error("Transcription Error:", error.message);
        res.status(500).json({ error: "Transcription failed" });
    }
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
