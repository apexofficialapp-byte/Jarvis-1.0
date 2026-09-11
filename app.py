import os
import time
import threading
from flask import Flask, render_template_string, request, jsonify
from playwright.sync_api import sync_playwright

app = Flask(__name__)

USER_DATA_DIR = os.path.join(os.getcwd(), "shared_ai_browser_profile")
browser_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JARVIS - AI Shared Bridge</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white flex flex-col items-center justify-center h-screen">
    <div class="bg-gray-800 p-8 rounded-2xl shadow-xl w-96 text-center">
        <h1 class="text-2xl font-bold mb-4 text-cyan-400">JARVIS Dashboard</h1>
        <textarea id="prompt" class="w-full p-3 bg-gray-700 rounded-lg text-white mb-4" rows="4" placeholder="Enter prompt..."></textarea>
        <button onclick="sendPrompt()" class="bg-cyan-500 hover:bg-cyan-600 px-4 py-2 rounded-lg font-bold w-full">Send to AI</button>
        <p id="response" class="mt-4 text-gray-300 text-sm"></p>
    </div>
    <script>
        async function sendPrompt() {
            const prompt = document.getElementById('prompt').value;
            document.getElementById('response').innerText = "JARVIS is thinking...";
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: prompt, target: 'chatgpt'})
            });
            const data = await res.json();
            document.getElementById('response').innerText = data.reply;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt")
    target = data.get("target", "chatgpt")
    
    with browser_lock:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch_persistent_context(
                    user_data_dir=USER_DATA_DIR,
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"]
                )
                page = browser.pages[0] if browser.pages else browser.new_page()
                
                if target == "chatgpt":
                    page.goto("https://chatgpt.com/", timeout=60000)
                    time.sleep(3)
                    # Selector logic here based on your working code
                
                browser.close()
                return jsonify({"reply": "Automation executed successfully!"})
        except Exception as e:
            return jsonify({"reply": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)