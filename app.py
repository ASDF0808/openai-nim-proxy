import os
import time
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app)

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
    
    # 모델 선택 매핑
    if '3.7' in req_model:
        selected_model = 'gemini-3.7-flash'
    elif '3.6' in req_model:
        selected_model = 'gemini-3.6-flash'
    elif '3.5' in req_model or '3.5-lite' in req_model:
        selected_model = 'gemini-3.5-flash-lite'
    elif '3.1' in req_model or '3.1-lite' in req_model:
        selected_model = 'gemini-3.1-flash-lite'
    elif 'gemma' in req_model or '31b' in req_model:
        selected_model = 'gemma-4-31b-it'
    else:
        selected_model = req_model if req_model else 'gemini-3.5-flash-lite'

    prompt = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])
    
    # 안전 필터 최대한 해제
    config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
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
                has_content = False
                for chunk in response:
                    if chunk.text:
                        has_content = True
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
                
                # 만약 검열 등으로 아무 텍스트도 나오지 않은 경우
                if not has_content:
                    fail_data = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": selected_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "[Gemini 필터에 의해 답변이 차단되었습니다. Janitor 모델명을 gemma로 변경해 보세요.]"},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(fail_data)}\n\n"

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
            
            # 응답 텍스트 파싱 및 예외 처리
            reply_text = ""
            try:
                reply_text = response.text
            except Exception:
                pass

            if not reply_text:
                reply_text = "[Gemini 필터에 의해 답변이 차단되었습니다. Janitor 모델명을 gemma로 변경해 보세요.]"

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