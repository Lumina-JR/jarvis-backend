const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Install this package: npm install duck-duck-scrape
const { DuckDuckGoSearch } = require('duck-duck-scrape');

let conversationHistory = [];

app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        conversationHistory.push({ role: "user", content: message });

        let searchResults = "";

        // If the question seems to need current information, search
        if (/news|today|current|latest|weather|stock|price|what.*happened|who.*won/i.test(message)) {
            try {
                const search = await DuckDuckGoSearch.search(message, { safeSearch: DuckDuckGoSearch.SafeSearchType.OFF });
                searchResults = "\n\nRecent information: " + search.results.slice(0, 2).map(r => r.title + ": " + r.description).join("\n");
            } catch (e) {
                console.log("Search failed");
            }
        }

        const response = await axios.post('https://api.groq.com/openai/v1/chat/completions', {
            model: "llama-3.3-70b-versatile",
            messages: [
                { 
                    role: "system", 
                    content: "You are JARVIS, Shubham's personal AI assistant. You are warm, helpful, and slightly sarcastic in a friendly way. Speak naturally like a clever friend. Be concise but engaging. You belong to Shubham." 
                },
                ...conversationHistory,
                { role: "user", content: message + searchResults }
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

        if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);

        res.json({ reply });

    } catch (error) {
        console.error(error.message);
        res.status(500).json({ error: "Something went wrong" });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
