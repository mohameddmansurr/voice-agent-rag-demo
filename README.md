# Voice Agent with RAG (Deepgram + Groq + Cartesia)

A real-time voice AI assistant built with **LiveKit Agents**. It listens via **Deepgram**, thinks via **Groq (Llama 3.1)**, speaks via **Cartesia**, and uses a local **RAG (FAISS)** system to answer technical questions.

## ⚠️ Architecture Decision Record
**Note on Model Selection:**
The original requirement specified **Gemini Multimodal Live API**. During development, the Google Cloud Free Tier quota for real-time streaming was exceeded (`Error 1011`), which halted development.

To ensure a functional, demo-ready prototype could be delivered on time, I migrated the architecture to a **Modular Voice Pipeline**:
* **STT:** Deepgram (High accuracy, low latency)
* **LLM:** Groq Llama 3.1 8B Instant (Sub-second inference, strict tool use)
* **TTS:** Cartesia (Fastest TTS available)

This demonstrates the flexibility of the LiveKit framework to swap backend providers without changing frontend logic.

## 🚀 Features
- **Real-time Pipeline:** Sub-second voice-to-voice interaction.
- **RAG Integration:** Vector search (FAISS) to answer specific questions (e.g., "What is RAG?", "Tell me about Basil").
- **Intelligent Tool Use:** The agent automatically detects when to query the knowledge base.
- **Windows Optimized:** Built using `uv` for easy dependency management on Windows.

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- [LiveKit CLI](https://docs.livekit.io/home/cli/setup/) (Optional)

### 1. Backend Setup
```bash
# Clone the repo
git clone [https://github.com/mohameddmansurr/voice-agent-rag-demo.git](https://github.com/mohameddmansurr/voice-agent-rag-demo.git)
cd voice-agent-rag-demo

# Install dependencies
pip install uv
uv sync

```

### 2. Environment Configuration

Create a `.env.local` file in the root directory:

```env
# LiveKit Keys (from cloud.livekit.io)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret

# Pipeline Keys (Free Tiers)
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
CARTESIA_API_KEY=your_cartesia_key

```

### 3. Run the Agent

```bash
uv run agent.py dev

```

### 4. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm run dev

```

Open [http://localhost:3000](https://www.google.com/search?q=http://localhost:3000) to connect.

## 🧩 Code Structure

* `agent.py`: Main entry point. Defines the `Deepgram -> Groq -> Cartesia` pipeline and the `lookup_knowledge` tool.
* `rag.py`: Handles vector embeddings (`sentence-transformers`) and similarity search (`faiss-cpu`).
* `frontend/`: Next.js web interface using LiveKit Components.

```
