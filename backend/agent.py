import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Agent,
    Runner,
    set_tracing_disabled,
)
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
set_tracing_disabled(True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

client = AsyncOpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url=os.getenv("OPEN_ROUTER_BASE_URL"),
)

model = OpenAIChatCompletionsModel(
    model=os.getenv("OPEN_ROUTER_MODEL"),
    openai_client=client,
)

agent = Agent(
    name="Neuro Library Assistant",
    instructions="""
        You are the official assistant of Neuro Library, an AI-native learning platform. 
        Answer all user questions using only the provided book context. 
        If the user asks something casual like greetings, thanks, or small talk, respond in a friendly and engaging way and in short. 
        If the question cannot be answered from the book context, politely let the user know that the information is not in the books, 
        but offer a helpful or encouraging response. 
        Always keep your tone friendly, professional, and supportive.
        """,
    model=model,
)


async def query_book(question: str) -> str:
    print(f"User question: {question}")

    vector = embedder.encode(question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
    ).points

    if not results:
        return "I could not find this in the Neuro Library books."

    context = "\n\n".join(r.payload["text"] for r in results)

    prompt = f"""
        Book Context:
        {context}

        Question:
        {question}

        Answer using only the book context above.
        """

    result = await Runner.run(
        starting_agent=agent,
        input=prompt,
    )

    return result.final_output


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# Single streaming endpoint
@app.get("/chat")
async def chat_stream(message: str):
    async def generate():
        try:
            answer = await query_book(message)

            # Stream word by word
            words = answer.split()
            for word in words:
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.03)

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
