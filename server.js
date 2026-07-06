// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Settings
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping
const MODEL_MAPPING = {
    'gpt-4o': 'deepseek-v4-flash'
};

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE
    });
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy'
    });
});

// OpenAI models endpoint
app.get('/v1/models', (req, res) => {
    const models = Object.keys(MODEL_MAPPING).map(model => ({
        id: model,
        object: 'model',
        created: Date.now(),
        owned_by: 'nvidia-nim-proxy'
    }));

    res.json({
        object: 'list',
        data: models
    });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
    try {
        const {
            model,
            messages,
            temperature,
            max_tokens,
            stream
        } = req.body;

        let nimModel = MODEL_MAPPING[model];

        if (!nimModel) {
            const modelLower = (model || '').toLowerCase();

            if (
                modelLower.includes('gpt-4') ||
                modelLower.includes('claude-opus') ||
                modelLower.includes('405b')
            ) {
                nimModel = 'meta/llama-3.1-405b-instruct';
            } else if (
                modelLower.includes('claude') ||
                modelLower.includes('gemini') ||
                modelLower.includes('70b')
            ) {
                nimModel = 'meta/llama-3.1-70b-instruct';
            } else {
                nimModel = 'meta/llama-3.1-8b-instruct';
            }
        }

        const nimRequest = {
            model: nimModel,
            messages,
            temperature: temperature ?? 0.6,
            max_tokens: max_tokens ?? 4096,
            extra_body: ENABLE_THINKING_MODE
                ? {
                      chat_template_kwargs: {
                          thinking: true
                      }
                  }
                : undefined,
            stream: stream || false
        };

        console.log('========== NVIDIA REQUEST ==========');
        console.log('Model:', nimModel);
        console.log('API Base:', NIM_API_BASE);
        console.log(
            'API Key:',
            NIM_API_KEY
                ? NIM_API_KEY.substring(0, 10) + '...'
                : 'MISSING'
        );
        console.log('====================================');

        const response = await axios.post(
            `${NIM_API_BASE}/chat/completions`,
            nimRequest,
            {
                headers: {
                    Authorization: `Bearer ${NIM_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                responseType: stream ? 'stream' : 'json'
            }
        );

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');

            response.data.pipe(res);

            response.data.on('error', err => {
                console.error('Stream error:', err);
                res.end();
            });
        } else {
            const openaiResponse = {
                id: `chatcmpl-${Date.now()}`,
                object: 'chat.completion',
                created: Math.floor(Date.now() / 1000),
                model,
                choices: response.data.choices.map(choice => {
                    let fullContent =
                        choice.message?.content || '';

                    if (
                        SHOW_REASONING &&
                        choice.message?.reasoning_content
                    ) {
                        fullContent =
                            '<think>\n' +
                            choice.message.reasoning_content +
                            '\n</think>\n\n' +
                            fullContent;
                    }

                    return {
                        index: choice.index,
                        message: {
                            role: choice.message.role,
                            content: fullContent
                        },
                        finish_reason:
                            choice.finish_reason
                    };
                }),
                usage:
                    response.data.usage || {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0
                    }
            };

            res.json(openaiResponse);
        }
    } catch (error) {
        console.error('====================');
        console.error(
            'Proxy error status:',
            error.response?.status
        );
        console.error(
            'Proxy error data:',
            JSON.stringify(
                error.response?.data,
                null,
                2
            )
        );
        console.error(
            'Proxy error message:',
            error.message
        );
        console.error('====================');

        res.status(error.response?.status || 500).json({
            error: {
                message:
                    error.response?.data?.error?.message ||
                    error.message ||
                    'Internal server error',
                type: 'api_error',
                code:
                    error.response?.status || 500
            }
        });
    }
});

// Some clients use /chat/completions
app.post('/chat/completions', (req, res, next) => {
    req.url = '/v1/chat/completions';
    app._router.handle(req, res, next);
});

// Catch all
app.all('*', (req, res) => {
    res.status(404).json({
        error: {
            message: `Endpoint ${req.path} not found`,
            type: 'invalid_request_error',
            code: 404
        }
    });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(
        `OpenAI to NVIDIA NIM Proxy running on port ${PORT}`
    );
    console.log(
        `Health check: http://localhost:${PORT}/health`
    );
    console.log(
        `Reasoning display: ${
            SHOW_REASONING
                ? 'ENABLED'
                : 'DISABLED'
        }`
    );
    console.log(
        `Thinking mode: ${
            ENABLE_THINKING_MODE
                ? 'ENABLED'
                : 'DISABLED'
        }`
    );
    console.log(
        'API KEY:',
        NIM_API_KEY
            ? NIM_API_KEY.substring(0, 10) + '...'
            : 'MISSING'
    );
});