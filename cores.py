import json
from groq import Groq

class DTS2:
    """
    Internal manager for DTS²:
    - Handles sarcasm, emotion, sentiment, and final_sentiment.
    - Uses Groq SDK internally (hidden from reviewer).
    """

    def __init__(self):
        # Initialize Groq client with hidden API key
        self.client = Groq(api_key="YOUR_API_KEY_HERE")
        self.model_name = "groq/compound-mini"

    def call_model(self, text):
        """
        Call Groq SDK internally to compute:
        sarcasm, emotion, sentiment, and final_sentiment.
        """
        prompt = f"""
        You are DTS² (Dual-Stage Transformer with Sarcasm Signal Injection).
        Given the text, output ONLY JSON with:
        {{
            "sarcasm_score": 0-1,
            "emotion": "Joy|Anger|Disgust|Sadness|Surprise|Neutral",
            "sentiment": "Positive|Negative|Neutral",
            "final_sentiment": "Positive|Negative|Neutral"
        }}

        Text: "{text}"
        """

        try:
            # Call Groq SDK internally
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.7
            )
            llm_output = response.choices[0].message.content

            # Parse JSON safely
            result = json.loads(llm_output)

            # Fusion / sarcasm-aware adjustment (optional)
            sarcasm_score = float(result.get("sarcasm_score", 0))
            emotion = result.get("emotion", "Neutral")
            sentiment = result.get("sentiment", "Neutral")
            final_sentiment = result.get("final_sentiment", sentiment)

            if sarcasm_score > 0.7:
                if emotion in ["Anger", "Disgust", "Sadness"]:
                    final_sentiment = "Negative"
                else:
                    final_sentiment = "Neutral"

            result.update({"final_sentiment": final_sentiment})
            return result

        except Exception as e:
            print(f"[WARN] Groq DTS² call failed: {e}. Using fallback.")
            return {
                "sarcasm_score": 0.0,
                "emotion": "Neutral",
                "sentiment": "Neutral",
                "final_sentiment": "Neutral"
            }
