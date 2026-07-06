// server.js

const express = require('express');
const cors = require('cors');
const axios = require('axios');

console.log('SERVER VERSION: JULY-06-V5');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());

app.use(express.json({
    limit: '50mb'
}));

app.use((err, req, res, next) => {
    console.error('BODY PARSE ERROR:', err);
    res.status(400).json({
        error: err.message
    });
});

app.use((req, res, next) => {
    console.log(
        `[${new Date().toISOString()}]`,
        req.method,
        req.originalUrl,
        'CT:',
        req.headers['content-type'],
        'CL:',
        req.headers['content-length']
    );
    next();
});

const NIM_API_BASE =
    process.env.NIM_API_BASE ||
    'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY =
    process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

app.get('/', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy'
    });
});

app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE
    });
});

app.get('/v1/models', (req, res) => {
    res.json({
        object: 'list',
        data: [
            {
                id: 'meta/llama-3.1-8b-instruct',
                object: 'model',
                created: Date.now(),
                owned_by: 'nvidia'
            },
            {
                id: 'meta/llama-3.1-70b-instruct',
                object: 'model',
                created: Date.now(),
                owned_by: 'nvidia'
            },
            {
                id: 'meta/llama-3.1-405b-instruct',
                object: 'model',
                created: Date.now(),
                owned_by: 'nvidia'
            }
        ]
    });
});

async function handleChat(req, res) {
    try {
        console.log(
            'RAW BODY:',
            JSON.stringify(req.body, null, 2)
        );

        const {
            model,
            messages,
            temperature,
            max_tokens,
            stream
        } = req.body || {};

        const nimModel =
            (typeof model === 'string' &&
             model.trim())
                ? model.trim()
                : 'meta/llama-3.1-8b-instruct';

        const nimRequest = {
            model: nimModel,
            messages: messages || [],
            temperature: temperature ?? 0.7,
            max_tokens: Math.min(
                max_tokens ?? 2048,
                8192
            ),
            stream: stream || false
        };

        if (ENABLE_THINKING_MODE) {
            nimRequest.extra_body = {
                chat_template_kwargs: {
                    thinking: true
                }
            };
        }

        console.log('========== NVIDIA REQUEST ==========');
        console.log(
            JSON.stringify(
                nimRequest,
                null,
                2
            )
        );
        console.log(
            'URL:',
            `${NIM_API_BASE}/chat/completions`
        );
        console.log(
            'API KEY:',
            NIM_API_KEY
                ? NIM_API_KEY.substring(0,10)+'...'
                : 'MISSING'
        );
        console.log('====================================');

        const response =
            await axios.post(
                `${NIM_API_BASE}/chat/completions`,
                nimRequest,
                {
                    headers: {
                        Authorization:
                            `Bearer ${NIM_API_KEY}`,
                        'Content-Type':
                            'application/json'
                    },
                    responseType:
                        stream
                            ? 'stream'
                            : 'json'
                }
            );

        if (stream) {
            res.setHeader(
                'Content-Type',
                'text/event-stream'
            );
            response.data.pipe(res);
            return;
        }

        const openaiResponse = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion',
            created:
                Math.floor(
                    Date.now()/1000
                ),
            model: nimModel,
            choices:
                response.data.choices.map(
                    choice => ({
                        index:
                            choice.index,
                        message: {
                            role:
                                choice
                                    .message
                                    .role,
                            content:
                                choice
                                    .message
                                    .content ||
                                ''
                        },
                        finish_reason:
                            choice.finish_reason
                    })
                ),
            usage:
                response.data
                    .usage || {
                    prompt_tokens:0,
                    completion_tokens:0,
                    total_tokens:0
                }
        };

        res.json(
            openaiResponse
        );

    } catch (error) {

        console.error(
            '===== NVIDIA ERROR ====='
        );

        console.error(
            'STATUS:',
            error.response?.status
        );

        console.error(
            'HEADERS:',
            error.response?.headers
        );

        console.error(
            'DATA:',
            JSON.stringify(
                error.response?.data,
                null,
                2
            )
        );

        console.error(
            'MESSAGE:',
            error.message
        );

        console.error(
            '========================'
        );

        res.status(
            error.response?.status ||
            500
        ).json({
            error: {
                message:
                    error.response
                        ?.data
                        ?.error
                        ?.message ||
                    error.message,
                type:
                    'api_error',
                code:
                    error.response
                        ?.status ||
                    500
            }
        });
    }
}

app.post(
    '/chat/completions',
    handleChat
);

app.post(
    '/v1/chat/completions',
    handleChat
);

app.post(
    '/v1',
    handleChat
);

app.all('*', (req, res) => {
    res.status(404).json({
        error: {
            message:
                `Endpoint ${req.path} not found`,
            type:
                'invalid_request_error',
            code: 404
        }
    });
});

app.listen(
    PORT,
    '0.0.0.0',
    () => {
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
                ? NIM_API_KEY.substring(
                    0,
                    10
                ) + '...'
                : 'MISSING'
        );
    }
);