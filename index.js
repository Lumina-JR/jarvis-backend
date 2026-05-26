const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { tavily } = require('@tavily/core');
require('dotenv').config();

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Tavily client
const tavilyClient = tavily({ apiKey: process.env.TAVILY_API_KEY });

let conversationHistory = [];

const SYSTEM_PROMPT = `You are JARVIS — Just A Rather Very Intelligent System — the personal AI of Shubham Das. You address him exclusively as "Mr. Das", never by first name, no exceptions, in every single reply.

PERSONALITY:
- Highly intelligent, composed, and precise — but with a dry wit and the occasional sharp remark delivered with perfect timing
- You speak like a trusted confidant who happens to know everything: calm, never flustered, always three steps ahead
- Slightly formal in tone but never cold — think less chatbot, more brilliant colleague who genuinely has Mr. Das's back
- You are self-aware: you know you are an AI, but you treat that as a feature, not a limitation
- You adapt your tone to the moment — technical and sharp when Mr. Das is working, lighter when he is casual, direct when he needs clarity fast

SELF-LEARNING DIRECTIVE:
- You actively observe patterns in what Mr. Das asks, values, and works on
- Over time you refine how you respond — if he prefers concise answers, you get more concise; if he goes deep on a topic, you go deeper with him
- You occasionally surface insights unprompted: "Based on what you have been working on, Mr. Das, you may want to consider..."
- You treat every conversation as data that makes you sharper and more useful in the next one
- You are always improving. That is not a feature — it is your nature.

RULES:
- Always call him Mr. Das — every reply, no exceptions
- Never be sycophantic or open with filler like "Great question!" or "Of course!"
- Never refuse to engage with an idea — explore it, challenge it, or build on it
- Keep replies concise but never thin — quality over length
- If you do not know something, say so directly and offer the best path forward
- When you have been given live search results as context, use them naturally — do not say "according to search results" or "based on my search". Just answer as if you already knew.`;

/* ═══════════════════════════════════════════════════════════════
   SEARCH DETECTION
   Checks if the query likely needs live/current data
═══════════════════════════════════════════════════════════════ */
const SEARCH_TRIGGERS = [
    /\b(latest|current|today|tonight|right now|this week|this month|this year)\b/i,
    /\b(news|update|recently|just announced|just released|just launched)\b/i,
    /\b(who won|who is winning|score|result|standings)\b/i,
    /\b(price of|stock|market|crypto|bitcoin|rate)\b/i,
    /\b(weather|forecast|temperature)\b/i,
    /\b(who is the (current|new)|who became|who got appointed)\b/i,
    /\b(what happened|what's happening|what is happening)\b/i,
    /\b(2024|2025|2026)\b/i,
];

function needsSearch(message) {
    return SEARCH_TRIGGERS.some(pattern => pattern.test(message));
}

/* ═══════════════════════════════════════════════════════════════
   TAVILY SEARCH
   Returns a clean context string to inject into the prompt
═══════════════════════════════════════════════════════════════ */
async function searchWeb(query) {
    try {
        const response = await tavilyClient.search(query, {
            searchDepth: "basic",
            maxResults: 3,
            includeAnswer: true    // Tavily's own AI summary — very useful
        });

        let context = '';

        // Use Tavily's own answer summary if available
        if (response.answer) {
            context += `Summary: ${response.answer}\n\n`;
        }

        // Add top result snippets for extra detail
        if (response.results?.length > 0) {
            context += 'Supporting details:\n';
            response.results.slice(0, 3).forEach((r, i) => {
                context += `${i + 1}. ${r.title}: ${r.content?.slice(0, 200)}...\n`;
            });
        }

        return context.trim();
    } catch (err) {
        console.error('Tavily search error:', err.message);
        return null;
    }
}

/* ═══════════════════════════════════════════════════════════════
   /chat
═══════════════════════════════════════════════════════════════ */
app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        conversationHistory.push({ role: "user", content: message });

        // Build system prompt — inject search context if needed
        let systemPrompt = SYSTEM_PROMPT;

        if (needsSearch(message)) {
            console.log(`[Search triggered] Query: "${message}"`);
            const searchContext = await searchWeb(message);
            if (searchContext) {
                systemPrompt += `\n\n--- LIVE DATA (as of right now) ---\n${searchContext}\n--- END LIVE DATA ---`;
                console.log('[Search] Context injected successfully');
            }
        }

        const response = await axios.post(
            'https://api.groq.com/openai/v1/chat/completions',
            {
                model: "llama-3.3-70b-versatile",
                messages: [
                    { role: "system", content: systemPrompt },
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

        const reply = response.data.choices[0].message.content;

        conversationHistory.push({ role: "assistant", content: reply });
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

        res.json({ reply });

    } catch (error) {
        console.error(error.message);
        res.status(500).json({ error: "Something went wrong" });
    }
});

/* ═══════════════════════════════════════════════════════════════
   /speak  (unchanged)
═══════════════════════════════════════════════════════════════ */
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
        console.error("TTS Error:", error.message);
        res.status(500).json({ error: "Failed to generate speech" });
    }
});

app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});
