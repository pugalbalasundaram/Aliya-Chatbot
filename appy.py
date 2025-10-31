from pymongo import MongoClient
from flask import Flask, request, jsonify, send_from_directory, render_template_string, render_template, redirect, session
import ollama
import os
import json
import uuid
import time
import re
import psutil
import subprocess
from duckduckgo_search import DDGS
from jinja2 import Template
from fpdf import FPDF
from datetime import datetime
from dotenv import load_dotenv
from flask import session
from collections import defaultdict
from datetime import datetime

# ✅ Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY") or "fallback_secret_key"

# ✅ Load Mongo URI from .env
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# ✅ Define DB and collections
db = client["formDB"]
forms_collection = db["forms"]
submissions_collection = db["submissions"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FORM_FOLDER = os.path.join(os.getcwd(),BASE_DIR, 'forms')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'form_templates')
SUBMISSION_FOLDER = os.path.join(os.getcwd(),BASE_DIR, 'submissions')
JSON_FOLDER = os.path.join(BASE_DIR, 'json_submissions')
forms = list(forms_collection.find())

os.makedirs(FORM_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
os.makedirs(SUBMISSION_FOLDER, exist_ok=True)
os.makedirs(JSON_FOLDER, exist_ok=True)

MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
FORM_LOG_FILE = os.path.join(BASE_DIR, "form_logs.json")

THEMES = {
    "1": "professional",
    "2": "formal",
    "3": "school",
    "4": "creative",
    "5": "job"
}

# Count forms per day
form_creation_per_day = defaultdict(int)
for form in forms:
    if "created_at" in form:
        date = form["created_at"].strftime("%Y-%m-%d")
        form_creation_per_day[date] += 1

# Sort by date
sorted_form_data = dict(sorted(form_creation_per_day.items()))

def load_json(filename):
    return json.load(open(filename)) if os.path.exists(filename) else {}

def save_json_file(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_memory(): return load_json(MEMORY_FILE)
def save_memory(memory): save_json_file(MEMORY_FILE, memory)
def load_chat_history(): return load_json(CHAT_HISTORY_FILE)
def save_chat_history(history): save_json_file(CHAT_HISTORY_FILE, history)
def load_form_logs(): return load_json(FORM_LOG_FILE)
def save_form_logs(logs): save_json_file(FORM_LOG_FILE, logs)

def search_web(query):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(r['body'])
    return "\n".join(results)

def has_enough_memory(required_gb=1.0):
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    return available_gb >= required_gb

def get_ollama_backend():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("✅ GPU found — using CUDA backend")
            return "cuda"
        else:
            print("⚠️ No GPU detected — using CPU fallback")
            return "cpu"
    except Exception as e:
        print(f"⚠️ Error detecting GPU: {e}")
        return "cpu"

def ask_llama(prompt, model="llama3"):
    try:
        if not has_enough_memory():
            return "❌ Not enough system memory."
        backend = get_ollama_backend()
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"backend": backend})
        return response['message']['content']
    except Exception as e:
        return f"❌ Error: {str(e)}"
    
def enhance_prompt(user_input):
    return f"""
You are an expert at generating user-friendly and accessible HTML forms.

Your output must follow these rules strictly:
- Output only **raw HTML** — no markdown, explanation, or code blocks.
- Start with a descriptive <h2> title (e.g., "Job Application Form" or "Student Registration").
- Optionally include a short paragraph below the heading explaining the form’s purpose.
- Always include a hidden input like <input type="hidden" name="form_id" value="<random_id>">.
- Use proper <label>, <input>, <select>, <textarea> with placeholder text where appropriate.
- All user-input fields **must have** the `required` attribute.
- Group related fields with <div class="section"> or <fieldset> for visual clarity.
- Do not add any <button> — the system will inject a custom styled submit button.
- Do NOT include any text like “Here’s the form” or use backticks.

Now generate the HTML layout based on this user request:
\"{user_input}\"
"""

def detect_theme_ai(prompt):
    analysis_prompt = f"""
Classify the style of the following HTML form prompt into one of these themes:
[professional, formal, school, creative, colorful, minimal, dark, job].

Return only one word — the closest matching theme.
Prompt: "{prompt}"
"""
    reply = ask_llama(analysis_prompt).strip().lower()
    if reply in ["professional", "formal", "school", "creative", "minimal", "colorful", "dark", "job"]:
        return reply
    return "professional"

def get_template_by_theme(theme):
    file_path = os.path.join(TEMPLATE_FOLDER, f"{theme}.html")
    return open(file_path, "r", encoding="utf-8").read() if os.path.exists(file_path) else None

def generate_form_html(user_prompt):
    import uuid, re, os
    from datetime import datetime
    from jinja2 import Template

    # Detect theme using AI
    theme = detect_theme_ai(user_prompt)
    base_template = get_template_by_theme(theme) or get_template_by_theme("professional")
    if not base_template:
        return "❌ No theme templates found."

    form_id = str(uuid.uuid4())[:8]

    # Enhance prompt for Codellama
    prompt = enhance_prompt(user_prompt).replace("<random_id>", form_id)
    raw_html = ask_llama(prompt, model="codellama")

    # Clean up AI response: remove markdown/code blocks and <form> nesting
    def clean_html(raw):
        html = re.sub(r"(?s)```.*?```", "", raw)  # Remove code blocks
        html = re.sub(r"</?form[^>]*>", "", html, flags=re.IGNORECASE)  # Remove <form> tags
        html = re.sub(r'<input[^>]*(type=["\']?(submit|button|reset)["\']?)[^>]*>', '', html, flags=re.IGNORECASE)  # Remove submit buttons
        return html.strip()

    cleaned = clean_html(raw_html)

    # ✅ Inject required attributes
    def add_required_fields(html):
        html = re.sub(r'(<input[^>]+type="(text|email|date|number|file)")', r'\1 required', html)
        html = re.sub(r'(<textarea)', r'\1 required', html)
        html = re.sub(r'(<select)', r'\1 required', html)
        return html

    required_ready = add_required_fields(cleaned)

    # Inject <form> and styled submit button
    def wrap_with_form(body):
        return f"""
        <form action="/submit" method="POST" enctype="multipart/form-data">
            <input type="hidden" name="form_id" value="{form_id}">
            {body}
            <div class="text-center mt-6">
                <button type="submit" class="bg-gradient-to-r from-pink-500 to-orange-400 hover:from-pink-600 hover:to-orange-500 text-white font-bold py-3 px-8 rounded-full shadow-lg transition duration-300 transform hover:scale-105">
                    🚀 Submit Form
                </button>
            </div>
        </form>
        """

    full_form = wrap_with_form(required_ready)

    # Render template with the final form
    final_html = Template(base_template).render(content=full_form, script="""
    <script>
      document.querySelectorAll("input, textarea, select").forEach((el, i, arr) => {
        el.addEventListener("keydown", function(e) {
          if (e.key === "Enter") {
            e.preventDefault();
            const next = arr[i + 1];
            if (next) next.focus();
          }
        });
      });
    </script>
    """)

    # Save the file
    file_path = os.path.join(FORM_FOLDER, f"form_{form_id}.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    # Save to DB
    forms_collection.insert_one({
        "form_id": form_id,
        "prompt": user_prompt,
        "theme": theme,
        "generated_file": f"form_{form_id}.html",
        "timestamp": datetime.now()
    })

    return f"http://localhost:5000/forms/form_{form_id}.html"

def count_submissions():
    submission_path = os.path.join(os.getcwd(), "submissions")
    if not os.path.exists(submission_path):
        return 0
    return sum([len(files) for _, _, files in os.walk(submission_path)])

def count_forms():
    form_path = os.path.join(os.getcwd(), "forms")
    if not os.path.exists(form_path):
        return 0
    return sum([len(files) for _, _, files in os.walk(form_path)])

def save_submission_as_pdf(form_id, submitted_data):
    from fpdf import FPDF
    from datetime import datetime
    import os

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, f"Form Submission ID: {form_id}", ln=True, align="C")
            self.ln(10)

        def field_row(self, label, value):
            self.set_font("Arial", "B", 12)
            self.cell(50, 10, f"{label}:", border=0)
            self.set_font("Arial", "", 12)
            self.multi_cell(0, 10, value, border=0)
            self.ln(2)

    pdf = PDF()
    pdf.add_page()

    for key, value in submitted_data.items():
        safe_key = str(key).strip() if key else "Unknown Field"
        safe_value = str(value).strip() if value else "N/A"
        pdf.field_row(safe_key, safe_value)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"form_{form_id}_{timestamp}.pdf"
    output_path = os.path.join(SUBMISSION_FOLDER, filename)
    pdf.output(output_path)

    print(f"✅ PDF saved: {output_path}")
    return output_path

@app.route("/reset", methods=["POST"])
def reset():
    # Clear all collections
    db.submissions.delete_many({})
    db.forms.delete_many({})
    
    # Clear folders
    for folder in [FORM_FOLDER, SUBMISSION_FOLDER, JSON_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            except Exception as e:
                print(f"❌ Error clearing {file_path}: {e}")

    return "✅ All data and files cleared successfully."

@app.route("/forms/<filename>")
def serve_form(filename):
    return send_from_directory(FORM_FOLDER, filename)

@app.route("/submit", methods=["POST"])
def handle_submit():
    form_id = request.form.get("form_id", str(uuid.uuid4())[:8])
    submitted_data = request.form.to_dict()

    # Handle file upload (if any)
    if 'assignment' in request.files:
        file = request.files['assignment']
        file_path = os.path.join(SUBMISSION_FOLDER, file.filename)
        file.save(file_path)
        submitted_data['uploaded_file'] = file.filename

    # Save to MongoDB
    submissions_collection.insert_one({
        "form_id": form_id,
        "submitted_data": submitted_data,
        "timestamp": datetime.now()
    })

    # Save PDF
    save_submission_as_pdf(form_id, submitted_data)

    # Save JSON
    with open(os.path.join(JSON_FOLDER, f"{form_id}.json"), "w") as f:
        json.dump(submitted_data, f, indent=4)

    # ✅ Acknowledgement message
    return f"""
    <div style='font-family: Arial, sans-serif; text-align: center; padding: 2rem;'>
      <h2 style='color: green;'>✅ Submission Successful!</h2>
      <p>Your form was submitted and saved successfully.</p>
      <p>Form ID: <b>{form_id}</b></p>
      <a href='/chat' style='margin-top: 20px; display: inline-block; color: blue;'>Back to Home</a>
    </div>
    """

@app.route("/save-json", methods=["POST"])
def save_json():
    try:
        data = request.get_json()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"user_data_{timestamp}.json"
        filepath = os.path.join(JSON_FOLDER, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return "✅ Data saved successfully as JSON"
    except Exception as e:
        return f"❌ Error saving JSON: {str(e)}", 500
    

@app.route('/')
def index1():
    return redirect('/chat')
    
@app.route("/chat")
def index():
    if "chat_history" not in session:
        session["chat_history"] = []

    else:
        redirect("/chat")
    return render_template("chat.html", chat_history=session["chat_history"])


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_input = data.get("message", "")

    if "chat_history" not in session:
        session["chat_history"] = []

    # Append user's message
    session["chat_history"].append({"role": "user", "content": user_input})

    # Handle AI/form response
    if re.search(r'\b(form|registration|feedback|contact|application|survey)\b', user_input.lower()):
        link = generate_form_html(user_input)
        reply = f"✅ Your form is ready: <a href='{link}' target='_blank'>{link}</a>"
    else:
        reply = ask_llama(user_input, model="llama3")

    # Append AI reply
    session["chat_history"].append({"role": "ai", "content": reply})
    session.modified = True

    return jsonify({"reply": reply})


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # ✅ Replace with real authentication logic
        if email.endswith("hlo123@gmail.com") and password == "admin123":
            session["user"] = email
            return redirect("/dashboard")
        else:
            return "❌ Invalid credentials"

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    forms = list(forms_collection.find())  # moved inside
    form_count = len(forms)
    response_count = submissions_collection.count_documents({})
    active_count = form_count

    # Count forms per day again (fresh)
    form_creation_per_day = defaultdict(int)
    for form in forms:
        if "created_at" in form:
            date = form["created_at"].strftime("%Y-%m-%d")
            form_creation_per_day[date] += 1
    sorted_form_data = dict(sorted(form_creation_per_day.items()))

    recent_forms = sorted(forms, key=lambda x: x.get("timestamp", datetime.min), reverse=True)[:10]
    
    total_submissions = count_submissions()

    form_counts = {}
    if os.path.exists(FORM_FOLDER):
        for form_name in os.listdir(FORM_FOLDER):
            form_path = os.path.join(FORM_FOLDER, form_name)
            if os.path.isdir(form_path):
                count = len([
                    f for f in os.listdir(form_path)
                    if os.path.isfile(os.path.join(form_path, f)) and f.endswith(".html")
                ])
                form_counts[form_name] = count

    labels = list(form_counts.keys())
    data = list(form_counts.values())

    form_submits = {}
    if os.path.exists(SUBMISSION_FOLDER):
        for form_name in os.listdir(SUBMISSION_FOLDER):
            form_path = os.path.join(SUBMISSION_FOLDER, form_name)
            if os.path.isdir(form_path):
                count = len([
                    f for f in os.listdir(form_path)
                    if os.path.isfile(os.path.join(form_path, f)) and f.endswith(".pdf")
                ])
                form_submits[form_name] = count

    S_labels = list(form_submits.keys())
    S_data = list(form_submits.values())

    return render_template("dashboard.html",
                           total_responses=response_count,
                           forms=recent_forms,
                           form_count=form_count,
                           form_creation_labels=list(sorted_form_data.keys()),
                           form_creation_counts=list(sorted_form_data.values()),
                           response_count=response_count,
                           active_count=active_count,
                           user=session["user"],
                           submissions=total_submissions,
                           labels=labels, data=data,
                           S_labels=S_labels, S_data=S_data)

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/chat")

def update_memory(user_input, memory):
    if "my name is" in user_input.lower():
        memory["name"] = user_input.split("my name is")[-1].strip().split()[0]
    return memory

def chat():
    print("🤖 Hello! I’m your AI Assistant. I can chat or create forms based on your input.")
    memory = load_memory()
    chat_history = load_chat_history()

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye! Session ended.")
            break

        if re.search(r'\b(form|registration|feedback|contact|application|survey)\b', user_input.lower()):
            print("📝 It sounds like you want a form. Let’s choose a style.")
            print("🎨 Choose a style theme:")
            for key, value in THEMES.items():
                print(f"{key}: {value}")
            style_input = input("Enter style number: ")
            style = THEMES.get(style_input, "professional")

            full_prompt = f"Please generate a {style} style HTML form. Details: {user_input}"
            print("[Info] Generating form...")
            link = generate_form_html(full_prompt, style)
            print(f"✅ Your form is ready: {link}\n")
            continue

        chat_history.append({"role": "user", "content": user_input})
        memory = update_memory(user_input, memory)
        response = ask_llama(user_input, model="llama3")
        chat_history.append({"role": "assistant", "content": response})

        print(f"AI: {response}")

        save_memory(memory)
        save_chat_history(chat_history)

if __name__ == "__main__":
    app.run(debug=True) 

@app.after_request
def remove_frame_options(response):
    response.headers.pop("X-Frame-Options", None)  # Allow iframe embedding
    return response
