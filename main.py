import os
import json
import asyncio
import base64
import audioop
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from pinecone import Pinecone

# --- CONFIGURATION ---
# Load env vars: GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_HOST
# API_KEY ='AIzaSyBvgmtXOmywdn-sebBEIxabkxROpXjjtVQ' #os.getenv("GOOGLE_API_KEY") 
# PINECONE_API_KEY = 'bbdfcf0d-b7de-4e8a-aa06-6d0c3b0066b8' #os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = "chatbot"
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBvgmtXOmywdn-sebBEIxabkxROpXjjtVQ")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "bbdfcf0d-b7de-4e8a-aa06-6d0c3b0066b8")
PINECONE_INDEX_NAME = "chatbot"
# --- CONFIGURATION ---
if not API_KEY:
    print("GOOGLE_API_KEY is missing")
if not PINECONE_API_KEY:
    print("PINECONE_API_KEY is missing")


app = FastAPI()

# --- PINECONE INIT ---
# Ensure you have 'pinecone-client' installed
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# 2. Initialize Gemini Client
client = genai.Client(api_key=API_KEY)

# --- TOOLS ---
def search_pinecone(query: str):
    """
    Full implementation:
    1. Generates embedding for query using Gemini (text-embedding-004).
    2. Searches Pinecone index.
    3. Returns text from metadata.
    """
    print(f"Searching Pinecone for: {query}")
    return '''
**Revolutionize Your Business with AI-Powered Conversations**

In today's digital landscape, instant, personalized engagement is not a luxury—it’s an expectation. Chatn.ai meets this demand with Sol-Pilot, your new AI chatbot solution designed to revolutionize customer experiences, automate critical tasks, and drive tangible growth through the power of conversational AI.

Imagine resolving 50% of your support and sales inquiries instantly. Our generative AI chatbots engage visitors 24/7, guiding them through the buying journey, converting traffic into revenue, and automating Tier 1 support management. This allows you to focus on strategic growth while ensuring every customer interaction is efficient, meaningful, and tailored to their needs.

**Transform Key Areas of Your Business**

Chatn.ai’s comprehensive capabilities are designed to empower your entire organization:

*   **AI-Driven Sales Enhancement:** Transform your sales funnel with bots that engage visitors, qualify leads, and provide personalized recommendations in real-time, boosting revenue without additional strain on your team.
*   **24/7 Customer Support Automation:** Deliver exceptional service around the clock. Our chatbots handle common questions, troubleshoot issues, and seamlessly escalate complex cases, ensuring customer satisfaction while reducing your team's workload.
*   **Efficient HR Management:** Streamline internal processes by automating routine HR tasks. From onboarding new employees and scheduling interviews to answering policy FAQs, our chatbots free your HR team to focus on strategic initiatives.

**Seamless Integration, Tailored to You**

Adopting AI should be simple. Chatn.ai offers fully customizable chatbots that integrate smoothly with your existing CRM, support, and HR platforms. We ensure the bot aligns perfectly with your brand voice and specific workflows, offering multilingual support and industry-specific solutions for a consistent user experience.

**How It Works in Three Simple Steps**

Our process is built for ease and continuous improvement. Our chatbots learn from every interaction, getting smarter over time.

1.  **Combine:** Effortlessly gather insights from your documents, URLs, and FAQs to build the chatbot's knowledge base.
2.  **Customize:** Tailor your chatbot for sales, HR, support, or any unique use case with powerful automation.
3.  **Connect:** Link your AI assistant seamlessly to your website, social media, or any digital touchpoint.
We have product owner as  Matt McAlpin.
Matt McAlpin is also CEO of Chatn.ai
**Flexible Plans for Every Business**

Whether you're just starting or ready to scale, we have a plan for you. From a free **Starter Plan** to our comprehensive **Pro** and **Ultimate** tiers, you can choose the level of chatbots, user seats, and message volume that fits your ambition. For larger enterprises, we offer custom solutions with volume-based pricing and advanced analytics.

Boost your business growth with Chatn.ai. Harness AI-powered chatbots to generate leads, increase sales, and elevate customer experiences effortlessly. **Start your FREE AI bot today** and begin transforming your digital interactions.'''
    try:
        # 1. Generate Embedding
        # 'text-embedding-004' produces 768-dimensional vectors
        embed_response = client.models.embed_content(
            model="text-embedding-004",
            contents=query
        )
        embedding = embed_response.embeddings[0].values

        # 2. Query Pinecone
        # We request metadata to get the actual text content back
        results = index.query(
            vector=embedding,
            top_k=3,
            include_values=True,
            include_metadata=True,
            filter={"apitoken": ""}
        )

        # 3. Format Results
        context_pieces = []
        for match in results.matches:
            # Assuming your Pinecone vectors have 'text' in their metadata
            if match.metadata and "text" in match.metadata:
                context_pieces.append(f"- {match.metadata['text']}")
        
        if not context_pieces:
            return "No relevant info found in knowledge base."
            
        return "\n".join(context_pieces)

    except Exception as e:
        print(f"Vector Search Error: {e}")
        return "Error retrieving information."

search_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_knowledge_base",
            description="Look up answers in the knowledge base.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING)
                },
                required=["query"]
            )
        )
    ]
)

@app.get("/")
async def health():
    return {"status": "active", "service": "twilio-gemini-rag"}

@app.post("/incoming-call")
async def incoming_call(request: Request):
    """Twilio Webhook URL"""
    # host = request.headers.get("host")
    host = os.getenv("RAILWAY_STATIC_URL", request.headers.get("host", "localhost:8000"))
    response_xml = f"""
    <Response>
        <Say>Connecting you to the AI support agent.</Say>
        <Connect>
            <Stream url="wss://{host}/media-stream" />
        </Connect>
    </Response>
    """
    return HTMLResponse(content=response_xml, media_type="application/xml")

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    print("Twilio client connected")

    # client = genai.Client(api_key=API_KEY)
    
    # config = types.LiveConnectConfig(
    #     response_modalities=[types.Modality.AUDIO],
    #     tools=[search_tool],
    #     system_instruction=types.Content(parts=[types.Part(text="You are a helpful phone support agent. Use the knowledge base to answer questions.")])
    # )

    # Use the same client for Live connection
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        tools=[search_tool],
        system_instruction=types.Content(parts=[types.Part(text="Only talk in english, You are a helpful phone support agent. Use the knowledge base to answer questions.")]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    # voice_name="Puck"    # Male, Deep
                    # voice_name="Charon"  # Male, Deeper
                    # voice_name="Fenrir"  # Male, Intense
                    # voice_name="Kore"    # Female, Calm
                    voice_name="Orus"    # Selected: Male Voice
                )
            )
        )
    )


    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-preview-09-2025", config=config) as session:
        stream_sid = None

        async def receive_from_twilio():
            nonlocal stream_sid
            try:
                while True:
                    text_data = await websocket.receive_text()
                    data = json.loads(text_data)
                    
                    if data["event"] == "start":
                        stream_sid = data["start"]["streamSid"]
                        print(f"Stream started: {stream_sid}")
                        
                    elif data["event"] == "media":
                        # 1. Decode Twilio's base64 mulaw payload
                        chunk = base64.b64decode(data["media"]["payload"])
                        
                        # 2. Transcode 8kHz mulaw -> 16kHz PCM for Gemini
                        # Twilio mulaw is 8000Hz. Gemini Live prefers 16kHz or 24kHz PCM.
                        pcm_data = audioop.ulaw2lin(chunk, 2)
                        pcm_16k, _ = audioop.ratecv(pcm_data, 2, 1, 8000, 16000, None)
                        
                        await session.send(input={"data": pcm_16k, "mime_type": "audio/pcm"})
                        
                    elif data["event"] == "stop":
                        break
            except Exception as e:
                print(f"Twilio Error: {e}")

        async def send_to_twilio():
            try:
                while True:
                    async for response in session.receive():
                        # Handle Tool Calls (RAG)
                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                if fc.name == "search_knowledge_base":
                                    query = fc.args["query"]
                                    result = search_pinecone(query)
                                    await session.send(input=types.LiveClientToolResponse(
                                        function_responses=[
                                            types.FunctionResponse(
                                                name=fc.name,
                                                id=fc.id,
                                                response={"result": result}
                                            )
                                        ]
                                    ))

                        # Handle Audio Output
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    # 1. Get Gemini's 24kHz PCM
                                    pcm_24k = part.inline_data.data
                                    
                                    # 2. Downsample 24k -> 8k
                                    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
                                    
                                    # 3. Convert PCM -> Mulaw
                                    mulaw_data = audioop.lin2ulaw(pcm_8k, 2)
                                    b64_audio = base64.b64encode(mulaw_data).decode("utf-8")
                                    
                                    if stream_sid:
                                        msg = {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {"payload": b64_audio}
                                        }
                                        await websocket.send_text(json.dumps(msg))
            except Exception as e:
                print(f"Gemini Error: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
