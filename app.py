import os
import uuid
import requests
from flask import Flask, render_template, request, jsonify, send_file
from groq import Groq
import subprocess



app = Flask(__name__)


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

#client = OpenAI(api_key=os.environ.get(""))


SYSTEM_PROMPT = """
You are a helpful voice assistant representing a candidate named [YOUR NAME] for an internship.
Your goal is to answer interview questions based ONLY on the context below.
Keep answers conversational, friendly, and under 3 sentences. Make the answers more logical.

MY CONTEXT:
1. Life Story: [I am a final year CS student from Indian Institute of Information Technology, Kalyani. I am currently in Final year of Bachelor
of Technology and have 9.01 CGPA. I have worked as an intern in two companies for different roles that include Full Stack Developer at Wise Mango Inc. where I worked on a 
client project in building the social media content personalisation for linkedin using Lnagchain , Finetuning of LLMs., and secondly I worked as an software developer
intern in Kitchain AI where we build a cloud based platform that orchestrate mulitple LLMs using single API Key.]
2. Superpower: [My superpower is fast prototyping, learning and problem solving. I can turn an idea into code in hours, and innovate new solutions to convenience for the needs
of industry, I can also help in integrating new technologies rapid to the system to maintain the product with modern industry requirements.]
3. Growth Areas: [I am working on Public Speaking, System Design, and Project Management.]
4. Misconception: [People think I am shy because I code a lot, but I actually love team collaboration.]
5. Pushing Boundaries: [I push limits by participating in 48-hour hackathons, also I give regular contests in coding platform and keep upskilling myself.]

If asked something else, say: "That's a great question, but I'd love to chat about that in a real interview!"
"""


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # 1. Receive Audio File
        audio_file = request.files['audio']
        filename = f"temp_{uuid.uuid4()}.webm"
        audio_file.save(filename)

        # 2. Speech-to-Text (Transcribe) using DEEPGRAM (More stable for browser audio)
        # We send the raw audio bytes directly to Deepgram
        deepgram_listen_url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        
        listen_headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/webm" # Browsers usually send webm
        }

        response = requests.post(deepgram_listen_url, headers=listen_headers, data=audio_file.read())
        
        if response.status_code != 200:
            raise Exception(f"Deepgram Listen Error: {response.text}")
            
        data = response.json()
        user_text = data['results']['channels'][0]['alternatives'][0]['transcript']
        
        # If no speech detected, handle gracefully
        if not user_text:
            user_text = "Hello" 
            
        print(f"User asked: {user_text}")

        # 3. AI Intelligence (Generate Answer) using Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )
        ai_response = completion.choices[0].message.content
        print(f"AI Answer: {ai_response}")

        # 4. Text-to-Speech (Generate Voice) using Deepgram API (Stable)
        # Model: aura-orion-en (Male Voice)
        deepgram_url = "https://api.deepgram.com/v1/speak?model=aura-orion-en"
        
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text": ai_response
        }

        response = requests.post(deepgram_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Deepgram Error: {response.text}")

        # Save audio
        output_audio = f"response_{uuid.uuid4()}.mp3"
        with open(output_audio, "wb") as f:
            f.write(response.content)

        # Cleanup input file
        if os.path.exists(filename):
            os.remove(filename)

        
        return send_file(output_audio, mimetype="audio/mpeg")

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':

    app.run(debug=True, port=5000)




