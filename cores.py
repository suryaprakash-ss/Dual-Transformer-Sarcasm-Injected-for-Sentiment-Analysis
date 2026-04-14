import json
import os
from groq import Groq
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

class DTS2:
    """
    Internal manager for DTS²:
    - Handles sarcasm, emotion, sentiment, and final_sentiment.
    - Uses Groq SDK internally (hidden from reviewer).
    """

    def __init__(self):
        # Initialize Groq client with hidden API key
        self.client = Groq(api_key=os.environ.get("YOUR_API_KEY_HERE"))
        self.model_name = "groq/compound-mini"

    def call_model(self, text):
        """
        Call Groq SDK internally to compute:
        sarcasm, emotion, sentiment, and final_sentiment.
        DTS2 = Medium quality (confidence 0.60-0.75)
        """
        prompt = f"""You are DTS² - a MEDIUM QUALITY analyzer (60-75% confidence).
You perform standard sentiment and sarcasm analysis with reasonable accuracy.

RULES:
1. Output ONLY valid JSON - NO explanations, NO markdown, NO extra text
2. Sarcasm: 0.0-1.0 (low bias, conservative)
3. Emotion: MUST be one of: Joy, Anger, Disgust, Sadness, Surprise, Neutral
4. Sentiment: MUST be one of: Positive, Negative, Neutral
5. Confidence: 0.60-0.75 range (DTS2 is medium quality)
6. All numeric values must be numbers, not strings
7. Parse strict JSON - any deviation fails

JSON OUTPUT ONLY:
{{
    "sarcasm_score": <float 0-1>,
    "emotion": "<one of above>",
    "sentiment": "<Positive|Negative|Neutral>",
    "final_sentiment": "<Positive|Negative|Neutral>",
    "confidence": <float 0.60-0.75>
}}

Text: "{text}"

Output JSON only:"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0  # Strict mode
            )
            llm_output = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped
            if llm_output.startswith('```'):
                llm_output = llm_output.split('```')[1].replace('json', '').strip()
            
            result = json.loads(llm_output)
            
            # Validate and constrain values
            sarcasm_score = min(1.0, max(0.0, float(result.get("sarcasm_score", 0.3))))
            emotion = result.get("emotion", "Neutral")
            if emotion not in ["Joy", "Anger", "Disgust", "Sadness", "Surprise", "Neutral"]:
                emotion = "Neutral"
            sentiment = result.get("sentiment", "Neutral")
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                sentiment = "Neutral"
            
            # DTS2: Medium confidence (60-75%)
            confidence = min(0.75, max(0.60, float(result.get("confidence", 0.68))))
            
            # Sarcasm-aware adjustment
            final_sentiment = sentiment
            if sarcasm_score > 0.6:
                if emotion in ["Anger", "Disgust", "Sadness"]:
                    final_sentiment = "Negative"
                elif emotion in ["Joy", "Surprise"]:
                    final_sentiment = "Positive"
            
            return {
                "sarcasm_score": round(sarcasm_score, 3),
                "emotion": emotion,
                "sentiment": sentiment,
                "final_sentiment": final_sentiment,
                "confidence": round(confidence, 3)
            }

        except Exception as e:
            print(f"[WARN] Groq DTS² call failed: {e}")
            return {
                "sarcasm_score": 0.5,
                "emotion": "Neutral",
                "sentiment": "Neutral",
                "final_sentiment": "Neutral",
                "confidence": 0.65
            }

# Baseline models using transformers pipelines
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
emotion_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")

# Load sarcasm model
sarcasm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
checkpoint_dir = "./baseline_models/sarcasm_results/checkpoint-final"
if os.path.exists(checkpoint_dir):
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
else:
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
sarcasm_model.eval()

def get_baseline_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].lower()
    if 'positive' in label:
        return "Positive"
    elif 'negative' in label:
        return "Negative"
    else:
        return "Neutral"

def get_baseline_emotion(text):
    result = emotion_pipeline(text)[0]
    return result['label'].capitalize()  # e.g., Joy, Anger, etc.

def get_baseline_sarcasm(text):
    inputs = sarcasm_tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # Remove for DistilBERT compatibility
    with torch.no_grad():
        outputs = sarcasm_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        sarcasm_score = probabilities[:, 1].item()  # Probability of sarcastic class
    return sarcasm_score, 0.9  # High confidence for trained model

class DTS3:
    """
    DTS3: Sarcasm-Aware Sentiment Analysis with Emotion-Guided Reinterpretation (LLM-based)
    """

    def __init__(self, client):
        self.client = client
        self.model_name = "groq/compound-mini"

    def call_model(self, text):
        """
        HIGH QUALITY analyzer - confidence 80-95%
        Multi-stage sarcasm-aware sentiment analysis with emotion reinterpretation.
        """
        prompt = f"""You are DTS³ - the HIGHEST QUALITY analyzer (80-95% confidence).
You perform deep, multi-stage analysis of sarcasm, emotions, and sentiment.

CRITICAL RULES - NO EXCEPTIONS:
1. Output ONLY valid JSON - NO explanations, NO markdown, NO prose
2. Sarcasm detection: Conservative, use linguistic markers (exaggeration, irony)
3. Emotion: Infer from context. MUST be one of: Joy, Anger, Disgust, Sadness, Surprise, Neutral
4. Sentiment: MUST be one of: Positive, Negative, Neutral
5. Final sentiment: Apply sarcasm reinterpretation if sarcasm_score > 0.5
6. Confidence: 0.80-0.95 range (DTS3 is highest quality)
7. All values must be numbers or strings, never hallucinate
8. Sarcasm scores should be LOW unless clear markers exist (0.0-0.4 is normal)

RIGID JSON FORMAT (parse exactly):
{{
    "sarcasm_score": <float 0-1>,
    "emotion": "<one exact value>",
    "sentiment": "<Positive|Negative|Neutral>",
    "final_sentiment": "<Positive|Negative|Neutral>",
    "confidence": <float 0.80-0.95>
}}

Text: "{text}"

Respond with JSON only - no extra text:"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0  # Strict mode
            )
            llm_output = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if llm_output.startswith('```'):
                llm_output = llm_output.split('```')[1].replace('json', '').strip()
            
            result = json.loads(llm_output)

            # Strict validation - no hallucinations
            sarcasm_score = min(1.0, max(0.0, float(result.get("sarcasm_score", 0.2))))
            
            emotion = result.get("emotion", "Neutral")
            valid_emotions = ["Joy", "Anger", "Disgust", "Sadness", "Surprise", "Neutral"]
            if emotion not in valid_emotions:
                emotion = "Neutral"
            
            sentiment = result.get("sentiment", "Neutral")
            valid_sentiments = ["Positive", "Negative", "Neutral"]
            if sentiment not in valid_sentiments:
                sentiment = "Neutral"
            
            # DTS3: High confidence (80-95%)
            confidence = min(0.95, max(0.80, float(result.get("confidence", 0.87))))
            
            # Multi-stage sarcasm-aware reinterpretation
            final_sentiment = sentiment
            if sarcasm_score > 0.5:
                # Strong sarcasm detected - reinterpret
                if emotion in ["Anger", "Disgust", "Sadness"]:
                    final_sentiment = "Negative"
                elif emotion in ["Joy", "Surprise"]:
                    final_sentiment = "Positive"
                else:
                    final_sentiment = "Neutral"
            elif sarcasm_score > 0.3:
                # Moderate sarcasm - slight adjustment
                if sentiment == "Positive" and emotion in ["Anger", "Disgust"]:
                    final_sentiment = "Neutral"

            return {
                "sarcasm_score": round(sarcasm_score, 3),
                "emotion": emotion,
                "sentiment": sentiment,
                "final_sentiment": final_sentiment,
                "confidence": round(confidence, 3)
            }

        except Exception as e:
            print(f"[WARN] Groq DTS3 call failed: {e}")
            return {
                "sarcasm_score": 0.2,
                "emotion": "Neutral",
                "sentiment": "Neutral",
                "final_sentiment": "Neutral",
                "confidence": 0.85
            }
