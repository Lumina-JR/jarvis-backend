const express  = require('express');
const cors     = require('cors');
const axios    = require('axios');
const fs       = require('fs');
const path     = require('path');
require('dotenv').config();

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

/* ═══════════════════════════════════════════════════════════════
   PERSISTENT MEMORY  — stored in memory.json on disk
   Survives Render cold starts as long as the file isn't wiped.
   NOTE: Render's free tier uses an ephemeral filesystem — the
   file resets on every new deploy. For truly permanent memory
   across deploys, swap the file functions below with a free
   database like MongoDB Atlas or Render's own Postgres.
   Instructions at the bottom of this file.
═══════════════════════════════════════════════════════════════ */
const MEMORY_FILE = path.join(__dirname, 'memory.json');

function loadMemory() {
    try {
        if (fs.existsSync(MEMORY_FILE)) {
            const raw = fs.readFileSync(MEMORY_FILE, 'utf8');
            return JSON.parse(raw);
        }
    } catch (e) {
        console.error('Failed to load memory:', e.message);
    }
    // Default empty memory structure
    return {
        facts:   [],          // ["Shubham likes chess", "Shubham is learning ML"]
        history: []           // last N conversation turns
    };
}

function saveMemory(memory) {
    try {
        fs.writeFileSync(MEMORY_FILE, JSON.stringify(memory, null, 2), 'utf8');
    } catch (e) {
        console.error('Failed to save memory:', e.message);
    }
}

// Load persisted memory on startup
let memory = loadMemory();

// In-session conversation history — rebuilt from persisted history on load
// so JARVIS remembers the last conversation even after a cold start
let conversationHistory = memory.history || [];

/* ═══════════════════════════════════════════════════════════════
   MEMORY EXTRACTION
   After each JARVIS reply, scan for things worth remembering
   and add them to memory.facts so the system prompt stays rich.
═══════════════════════════════════════════════════════════════ */
const MEMORY_TRIGGERS = [
    /my name is (.+)/i,
    /i am (.+)/i,
    /i('m| am) (\d+ years old|working on|learning|building|studying|from|in) (.+)/i,
    /i (?:love|hate|like|dislike|prefer|enjoy) (.+)/i,
    /remember that (.+)/i,
    /don'?t forget (.+)/i,
    /note that (.+)/i,
    /my (.+?) is (.+)/i,
];

function extractFacts(message) {
    const newFacts = [];
    for (const pattern of MEMORY_TRIGGERS) {
        const match = message.match(pattern);
        if (match) {
            const fact = message.trim();
            if (!memory.facts.includes(fact)) {
                newFacts.push(fact);
            }
        }
    }
    return newFacts;
}

function buildSystemPrompt() {
    const factBlock = memory.facts.length > 0
        ? `\n\nThings you remember about Mr. Das:\n${memory.facts.map(f => `- ${f}`).join('\n')}`
        : '';

    return `You are JARVIS — Just A Rather Very Intelligent System — the personal AI of Shubham Das. You address him exclusively as "Mr. Das", never by first name, no exceptions, in every single reply.

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
- If you do not know something, say so directly and offer the best path forward${factBlock}`;
}

/* ═══════════════════════════════════════════════════════════════
   /chat
═══════════════════════════════════════════════════════════════ */
app.post('/chat', async (req, res) => {
    try {
        const { message } = req.body;
        if (!message) return res.status(400).json({ error: "Message is required" });

        // Extract and store any memorable facts from user message
        const newFacts = extractFacts(message);
        if (newFacts.length > 0) {
            memory.facts.push(...newFacts);
            // Cap facts at 50 so the system prompt doesn't balloon
            if (memory.facts.length > 50) memory.facts = memory.facts.slice(-50);
        }

        conversationHistory.push({ role: "user", content: message });

        const response = await axios.post(
            'https://api.groq.com/openai/v1/chat/completions',
            {
                model: "llama-3.3-70b-versatile",
                messages: [
                    { role: "system", content: buildSystemPrompt() },
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

        // Keep last 20 turns in active context
        if (conversationHistory.length > 20) {
            conversationHistory = conversationHistory.slice(-20);
        }

        // Persist history + facts to disk after every turn
        memory.history = conversationHistory;
        saveMemory(memory);

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

/* ═══════════════════════════════════════════════════════════════
   /transcribe  — Deepgram STT (for the mic in jarvis.html)
═══════════════════════════════════════════════════════════════ */
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage() });

app.post('/transcribe', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: "No audio file received" });

        const response = await axios.post(
            'https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&language=en',
            req.file.buffer,
            {
                headers: {
                    'Authorization': `Token ${process.env.DEEPGRAM_API_KEY}`,
                    'Content-Type': req.file.mimetype || 'audio/webm'
                }
            }
        );

        const transcript = response.data?.results?.channels?.[0]
            ?.alternatives?.[0]?.transcript || '';

        res.json({ transcript });

    } catch (error) {
        console.error("Transcribe Error:", error.message);
        res.status(500).json({ error: "Transcription failed" });
    }
});

/* ═══════════════════════════════════════════════════════════════
   /memory  — read or clear memory (optional debug endpoints)
═══════════════════════════════════════════════════════════════ */
app.get('/memory', (req, res) => {
    res.json({ facts: memory.facts, turns: conversationHistory.length });
});

app.delete('/memory', (req, res) => {
    memory = { facts: [], history: [] };
    conversationHistory = [];
    saveMemory(memory);
    res.json({ message: "Memory cleared." });
});

/* ═══════════════════════════════════════════════════════════════
   ROOT
═══════════════════════════════════════════════════════════════ */
app.get('/', (req, res) => {
    res.send('Jarvis Backend is Running 🚀');
});

app.listen(PORT, () => {
    console.log(`Jarvis Backend running on port ${PORT}`);
});

/*
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  UPGRADING TO PERMANENT MEMORY (survives redeploys)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Render's free tier wipes the filesystem on every new deploy.
  To keep memory permanently, swap the file-based functions
  with MongoDB Atlas (free tier):

  1. npm install mongoose
  2. Add MONGODB_URI to your Render environment variables
  3. Replace loadMemory() / saveMemory() with mongoose calls:

  const mongoose = require('mongoose');
  mongoose.connect(process.env.MONGODB_URI);

  const MemorySchema = new mongoose.Schema({
      key:   { type: String, default: 'jarvis' },
      facts: [String],
      history: Array
  });
  const Memory = mongoose.model('Memory', MemorySchema);

  async function loadMemory() {
      return await Memory.findOne({ key: 'jarvis' })
          || { facts: [], history: [] };
  }
  async function saveMemory(data) {
      await Memory.findOneAndUpdate(
          { key: 'jarvis' },
          { facts: data.facts, history: data.history },
          { upsert: true }
      );
  }
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
*/
