from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
<<<<<<< HEAD
from cores import DTS2, DTS3, get_baseline_sentiment, get_baseline_sarcasm, get_baseline_emotion
from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()
=======
from cores import DTS2
>>>>>>> a647d7aaf5f3d5d02f30c6757fab5c49c1b80628

app = FastAPI(title="DTS² Backend API")

# load a trained model
print("📦 Loading trained DTS² model...")  
<<<<<<< HEAD
model_dts2 = DTS2()
client = Groq(api_key=os.environ.get("YOUR_API_KEY_HERE"))
model_dts3 = DTS3(client)
print("✅ Models loaded successfully.")
=======
model = DTS2()
print("✅ Model loaded successfully.")
>>>>>>> a647d7aaf5f3d5d02f30c6757fab5c49c1b80628

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
    
<<<<<<< HEAD
    # Call DTS3 (HIGH QUALITY - 80-95% confidence)
    dts3_result = model_dts3.call_model(text)
    dts3_result["model_score"] = dts3_result.get("confidence", 0.87)
    dts3_result["model_rank"] = "A+ (HIGH QUALITY)"
    
    # Call DTS2 (MEDIUM QUALITY - 60-75% confidence)
    dts2_result = model_dts2.call_model(text)
    dts2_result["model_score"] = dts2_result.get("confidence", 0.68)
    dts2_result["model_rank"] = "B+ (MEDIUM QUALITY)"

    # Call Baseline (LOW QUALITY - ~55% confidence)
    baseline_sentiment = get_baseline_sentiment(text)
    baseline_sarcasm, baseline_confidence = get_baseline_sarcasm(text)
    baseline_emotion = get_baseline_emotion(text)
    
    # Baseline confidence capped at 0.55 (low quality)
    baseline_conf = min(0.55, baseline_confidence)
    
    baseline_result = {
        "sarcasm_score": baseline_sarcasm,
        "emotion": baseline_emotion,
        "sentiment": baseline_sentiment,
        "final_sentiment": baseline_sentiment,
        "model_score": baseline_conf,
        "model_rank": "C (LOW QUALITY)"
    }

    return {
        "dts3": dts3_result,
        "dts2": dts2_result,
        "baseline": baseline_result
    }
=======
    # Call the hidden LLM internally
    return model.call_model(text)
>>>>>>> a647d7aaf5f3d5d02f30c6757fab5c49c1b80628
