import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from app import app as application

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
MODEL_NAME = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    GEMINI_API_KEY = "AIzaSyBQO4cb0B_fDkofjnVg-FlnUafAG8hvdKI"
    genai.configure(api_key=GEMINI_API_KEY)


def _extract_text_from_obj(obj):
    """Recursively extract plain text from SDK objects or JSON-like structures."""
    if obj is None:
        return None
    # Plain string
    if isinstance(obj, str):
        return obj
    # If it's a list, join extracted parts
    if isinstance(obj, list):
        parts = [p for p in ( _extract_text_from_obj(x) for x in obj ) if p]
        return "\n".join(parts) if parts else None
    # Objects from google.genai often have .text or .content or to_dict()
    try:
        # common text attribute
        if hasattr(obj, 'text') and isinstance(getattr(obj, 'text'), str):
            return getattr(obj, 'text')
        if hasattr(obj, 'content'):
            return _extract_text_from_obj(getattr(obj, 'content'))
        if hasattr(obj, 'candidates'):
            return _extract_text_from_obj(getattr(obj, 'candidates'))
        # to_dict -> search for text-like fields
        if hasattr(obj, 'to_dict'):
            d = obj.to_dict()
            return _extract_text_from_obj(d)
    except Exception:
        pass
    # dict-like
    if isinstance(obj, dict):
        for key in ('text', 'content', 'output', 'response', 'candidates'):
            if key in obj:
                return _extract_text_from_obj(obj[key])
        # fallback: try values
        for v in obj.values():
            t = _extract_text_from_obj(v)
            if t:
                return t
    # Fallback: string conversion
    try:
        return str(obj)
    except Exception:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Set GEMINI_API_KEY environment variable.'}), 500
    
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Generate content using Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)