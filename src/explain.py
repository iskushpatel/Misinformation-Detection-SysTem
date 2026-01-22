import config
import google.generativeai as genai

# Configure the AI Model
try:
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
    AI_AVAILABLE = True
except Exception as e:
    print(f"⚠️ API WARNING: Could not connect to Gemini. Using offline mode. Error: {e}")
    AI_AVAILABLE = False

def generate_explanation(claim, retrieval_results):
    
    print(f"\n--- ANALYZING CLAIM: '{claim}' ---")
    
    if not retrieval_results:
        return "NO DATA: No verified evidence found in our database to support or debunk this."

    top_hit = retrieval_results[0]
    try:
        score = top_hit.score
        payload = top_hit.payload
    except AttributeError:
        if hasattr(top_hit[0], 'score'):
             score = top_hit[0].score
             payload = top_hit[0].payload
        else:
             return "SYSTEM ERROR: Data format mismatch."
    if score < config.SIMILARITY_THRESHOLD:
        return (f"UNCERTAIN: We found similar concepts, but the match score ({score:.2f}) "
                f"is too low to safely verify this claim.")

    if AI_AVAILABLE:
        try:
            prompt = f"""
            You are FactChk, a professional fact checker.
            
            USER CLAIM: "{claim}"
            
            VERIFIED EVIDENCE DATABASE:
            - Fact: "{payload['text']}"
            - Verdict: {payload['verdict']}
            - Source: {payload['source']} (Credibility: {payload['credibility_score']})
            
            TASK:
            Write a helpful, 2-sentence response to the user.
            1. Clearly state if the claim is True, False, or Misleading based *only* on the evidence.
            2. Explain the correct context naturally.
            3. Mention the source.
            
            DO NOT make up information. Stick strictly to the evidence provided.
            """
        
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"GenAI Error: {e}")
            pass

    # --- Offline backup ---
    explanation = []
    explanation.append(f"VERDICT: {payload['verdict']}")
    explanation.append(f"EVIDENCE: {payload['text']}")
    explanation.append(f"SOURCE: {payload['source']}")
    return "\n".join(explanation)