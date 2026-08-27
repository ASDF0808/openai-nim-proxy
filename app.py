import os
import time
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
def home():
    return jsonify({"status": "active", "message": "Render Gemini Proxy Ready"}), 200

@app.route('/v1', methods=['GET', 'POST', 'OPTIONS'])
def check_v1():
    return jsonify({"status": "active", "message": "Render Gemini Proxy v1 Ready"}), 200

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.json or {}
    api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    if not api_key:
        return jsonify({"error": "API Key is missing"}), 401

    client = genai.Client(api_key=api_key)
    messages = data.get('messages', [])
    is_stream = data.get('stream', False)
    
    req_model = data.get('model', '').strip().lower()
    
    # 모델 매핑 로직
    if '3.7' in req_model:
        selected_model = 'gemini-3.7-flash'
    else:
        selected_model = req_model if req_model else 'gemini-3.5-flash-lite'

    prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
    
    config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]
    )

    if is_stream:
        def generate():
            try:
                response = client.models.generate_content_stream(
                    model=selected_model,
                    contents=prompt,
                    config=config
                )
                for chunk in response:
                    if chunk.text:
                        chunk_data = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": selected_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": chunk.text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate(), mimetype='text/event-stream')
    else:
        try:
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=config
            )
            reply_text = response.text if response.text else "..."

            return jsonify({
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": selected_model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)