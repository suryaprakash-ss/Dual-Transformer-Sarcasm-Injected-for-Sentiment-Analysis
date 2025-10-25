from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from cores import DTS2

app = FastAPI(title="DTS² Backend API")

# load a trained model
print("📦 Loading trained DTS² model...")  
model = DTS2()
print("✅ Model loaded successfully.")

# Allow frontend (Streamlit) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class TextInput(BaseModel):
    text: str

# Endpoint for analysis
@app.post("/analyze/", response_model=dict)
async def analyze(input: TextInput):
    text = input.text.strip()
    if not text:
        return {"error": "Empty input text"}
    
    # Call the hidden LLM internally
    return model.call_model(text)
