import os
import uuid
import requests
from flask import Flask, render_template, request, jsonify, send_file
from groq import Groq

app = Flask(__name__)

# --- CONFIGURATION ---
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

# --- PERSONALIZE THIS SECTION ---
SYSTEM_PROMPT = """
You are a helpful voice assistant representing a candidate named [YOUR NAME] for an internship.
Your goal is to answer interview questions based ONLY on the context below.
Keep answers conversational, friendly, and under 3 sentences. Make the answers logical and professional.

MY CONTEXT:
1. Life Story: [I am a final year CS student from Indian Institute of Information Technology, Kalyani. I am currently in my final year of B.Tech with a 9.01 CGPA. I have interned at two companies: as a Full Stack Developer at Wise Mango Inc., where I built social media content personalization for LinkedIn using LangChain and LLM Fine-tuning; and as a Software Developer Intern at Kitchain AI, where I built a cloud-based platform that orchestrates multiple LLMs using a single API Key.]
2. Superpower: [My superpower is fast prototyping, learning, and problem-solving. I can turn an idea into code in hours and innovate new solutions to meet industry needs. I excel at rapidly integrating new technologies into systems to keep products modern and up-to-date.]
3. Growth Areas: [I am working on Public Speaking, System Design, and Project Management.]
4. Misconception: [People think I am shy because I code a lot, but I actually love team collaboration.]
5. Pushing Boundaries: [I push limits by participating in 48-hour hackathons. I also regularly compete in coding contests to keep upskilling myself.]

If asked something else, say: "That's a great question, but I'd love to chat about that in a real interview!"
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if not DEEPGRAM_API_KEY:
            return jsonify({"error": "DEEPGRAM_API_KEY missing"}), 500

        # 1. Receive Audio File
        audio_file = request.files['audio']
        # We force the extension to .webm so Groq knows how to handle it
        filename = f"input_{uuid.uuid4()}.webm"
        audio_file.save(filename)

        # 2. Speech-to-Text (Transcribe) using GROQ (Whisper)
        # We use Groq here because the library handles browser files better than raw requests
        print("Transcribing...")
        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()), # Tuple format (name, bytes) is CRITICAL here
                model="whisper-large-v3",
                response_format="text"
            )
        user_text = transcription
        print(f"User asked: {user_text}")

        # 3. AI Intelligence (Generate Answer) using GROQ
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )
        ai_response = completion.choices[0].message.content
        print(f"AI Answer: {ai_response}")

        # 4. Text-to-Speech (Generate Voice) using DEEPGRAM (Male Voice)
        deepgram_url = "https://api.deepgram.com/v1/speak?model=aura-orion-en"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": ai_response
        }

        # Request audio from Deepgram
        response = requests.post(deepgram_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Deepgram TTS Error: {response.text}")

        # Save audio
        output_audio = f"response_{uuid.uuid4()}.mp3"
        with open(output_audio, "wb") as f:
            f.write(response.content)

        # Cleanup input file
        if os.path.exists(filename):
            os.remove(filename)

        # Return audio to frontend
        return send_file(output_audio, mimetype="audio/mpeg")

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
