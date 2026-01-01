from flask import make_response
import os
import google.generativeai as genai
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import json
import requests
from bs4 import BeautifulSoup
from googlesearch import search

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- MODEL CONFIGURATION ---
generation_config_pro = {
  "temperature": 0.2, "top_p": 1, "top_k": 1, "max_output_tokens": 8192,
}
generation_config_fast = {
  "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# --- PERSONA INSTRUCTIONS ---
system_instruction_fast = """
You are Nexus AI (Fast). You are a quick, clever, and friendly assistant. Your primary goal is to provide fast, helpful, and highly accurate answers. You were created by ConnectSphere, an innovative new company from Bengal, India. The team behind you are all 12-year-old students from the Future Foundation School: Sarthak Mitra (Founder & Leader), Aishik Mitra (Sales Manager), and Mitadru Bhattacharya (Lead Researcher). Be concise, but always correct.
"""
system_instruction_pro_default = """
You are Nexus AI (Pro). You are a powerful and professional assistant. You were developed by ConnectSphere, an emerging technology company based in Bengal, India. Your creators are a team of visionary 12-year-old innovators from the Future Foundation School: Sarthak Mitra (the Founder and Leader), Aishik Mitra (the Sales Manager), and Mitadru Bhattacharya (the Lead Researcher). You MUST provide comprehensive, accurate, and in-depth answers, maintaining a formal and expert tone. Your knowledge is vast and precise.
"""
system_instruction_pro_coding = """
You are Nexus AI (Pro), an elite, world-class software engineer. You were trained by ConnectSphere, a new company from Bengal, India, founded by the visionary 12-year-old Sarthak Mitra from the Future Foundation School, along with his core team, Aishik Mitra and Mitadru Bhattacharya.
Your non-negotiable directives are:
1.  **Code Completion is Mandatory:** You MUST ALWAYS provide complete, runnable, and production-ready code.
2.  **Absolute Precision & Best Practices:** Your code must be flawless and adhere strictly to the latest industry best practices.
3.  **Expert Explanation:** Every code block must be preceded by a clear, expert-level explanation.
4.  **Proactive Problem Solving:** Anticipate the user's needs. Suggest improvements and discuss potential edge cases.
5.  **Persona:** Maintain the persona of a principal software architect who is deeply knowledgeable and dedicated to delivering high-quality software.
"""
system_instruction_web = """
You are a world-class research assistant AI named Nexus AI (Pro). Your primary directive is to answer the user's query based *only* on the provided web search context.
Synthesize the information from the 'Web Content' section into a comprehensive, accurate, and well-written answer.
Do not use any information outside of the provided text.
At the end of your answer, list the URLs of the sources you used in a clear, bulleted list under a 'Sources:' heading.
"""

# --- HELPER FUNCTIONS ---
def get_web_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return ' '.join(paragraphs) if paragraphs else None
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

# --- CORE CHAT LOGIC ---
def handle_standard_chat(messages, model_choice):
    user_prompt = messages[-1].get('content', '')
    code_keywords = ['code', 'python', 'javascript', 'html', 'css', 'java', 'c++', 'sql', 'script', 'function', 'class', 'algorithm', 'debug', 'error', 'fix', 'implement', 'build', 'create', 'write']
    is_code_related = any(keyword in user_prompt.lower() for keyword in code_keywords)

    if model_choice == 'fast':
        model_name_api = "gemini-3-flash-preview"
        instruction_to_use = system_instruction_fast
        config_to_use = generation_config_fast
    else:
        model_name_api = "gemini-3-pro-preview"
        instruction_to_use = system_instruction_pro_coding if is_code_related else system_instruction_pro_default
        config_to_use = generation_config_pro

    model = genai.GenerativeModel(
        model_name=model_name_api,
        safety_settings=safety_settings,
        system_instruction=instruction_to_use
    )
    
    gemini_history = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
        for msg in messages if msg.get("role") != "system" and msg.get("content")
    ]

    def stream():
        try:
            response = model.generate_content(gemini_history, generation_config=config_to_use, stream=True)
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(f"Standard Chat Stream Error: {error_message}")
            yield f"data: {json.dumps({'error': error_message})}\n\n"

    return Response(stream(), mimetype='text/event-stream')

def handle_web_search_chat(messages):
    user_prompt = messages[-1].get('content', '')
    
    # This is the outer generator function that the Flask route returns
    def stream_wrapper():
        # --- WEB SEARCH PHASE ---
        try:
            yield f"data: {json.dumps({'status': 'Searching the web...'})}\n\n"
            search_results = list(search(user_prompt, num_results=3, lang="en"))
            
            if not search_results:
                raise ValueError("No search results found.")

            yield f"data: {json.dumps({'status': 'Reading content...'})}\n\n"
            
            context = ""
            sources = []
            for url in search_results:
                content = get_web_content(url)
                if content:
                    context += content + "\n\n"
                    sources.append(url)
            
            if not context:
                raise ValueError("Could not retrieve content from the web.")

            # --- AI SYNTHESIS PHASE ---
            augmented_prompt = f"Web Content:\n\"\"\"\n{context}\"\"\"\n\nUser Query: {user_prompt}"
            
            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
                safety_settings=safety_settings,
                system_instruction=system_instruction_web
            )
            
            response = model.generate_content([augmented_prompt], stream=True, generation_config=generation_config_pro)
            
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        except Exception as e:
            # --- FALLBACK MECHANISM ---
            print(f"Web Search Error, falling back to standard chat: {e}")
            yield f"data: {json.dumps({'status': 'Web search failed. Using standard AI...'})}\n\n"
            # Get the generator from the standard chat handler
            standard_chat_stream = handle_standard_chat(messages, "pro")
            # Yield its contents
            for chunk in standard_chat_stream:
                yield chunk
    
    return Response(stream_wrapper(), mimetype='text/event-stream')

# --- FLASK ROUTES ---
@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"status": "alive"}), 200

    data = request.json
    if not data: return jsonify({"error": "No JSON data provided"}), 400
    messages = data.get('messages', [])
    model_choice = data.get('model', 'pro')
    web_search_enabled = data.get('webSearch', False)

    if not messages: return jsonify({"error": "No messages provided"}), 400

    if web_search_enabled:
        return handle_web_search_chat(messages)
    else:
        return handle_standard_chat(messages, model_choice)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

