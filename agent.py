import logging
import os
from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    AgentSession,
)
from livekit.agents import Agent
from livekit.agents.llm import function_tool

from livekit.plugins import deepgram
from livekit.plugins import openai
from livekit.plugins import cartesia
from livekit.plugins import silero

from rag import RAGEngine

load_dotenv(".env.local")
logger = logging.getLogger("voice-agent")

# 1. Initialize RAG
rag_engine = RAGEngine()

# 2. Define the RAG Tool
@function_tool
async def lookup_knowledge(query: str) -> str:
    """
    Search the knowledge base for technical questions about LiveKit, RAG, or this project.
    """
    logger.info(f"RAG Search: {query}")
    return rag_engine.search(query)

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 3. Define the Pipeline Components
    # Hearing (STT) - Deepgram
    stt_component = deepgram.STT()

    # Thinking (LLM) - Groq 
    llm_component = openai.LLM(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant", 
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # Speaking (TTS) - Cartesia
    tts_component = cartesia.TTS()
    
    # Interruptions (VAD) - Silero
    vad_component = silero.VAD.load()

    # 4. Create the Agent
    agent = Agent(
        instructions=(
            "You are a helpful voice assistant. "
            "You have a tool called 'lookup_knowledge' to answer technical questions. "
            "IMPORTANT: If the user says 'RAG', 'Rag', or 'Greg', they mean 'Retrieval-Augmented Generation'. "
            "Use the 'lookup_knowledge' tool for these queries. "
            "Keep your answers concise and conversational."),
        tools=[lookup_knowledge]
    )

    # 5. Create the Session
    session = AgentSession(
        vad=vad_component,
        stt=stt_component,
        llm=llm_component,
        tts=tts_component
    )

    # 6. Start
    await session.start(room=ctx.room, agent=agent)
    
    # Send Greeting
    await session.say("Hello! How can I help you today?", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))