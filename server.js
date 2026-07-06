// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

console.log('SERVER VERSION: JULY-06-V4');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
    console.log(
        `[${new Date().toISOString()}]`,
        req.method,
        req.originalUrl
    );
    next();
});

// NVIDIA settings
const NIM_API_BASE =
    process.env.NIM_API_BASE ||
    'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY =
    process.env.NIM_API_KEY;

const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Root
app.get('/', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy'
    });
});

// Health
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE
    });
});

// Models
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

// Main handler
async function handleChat(req, res) {
    try {
        const {
            model,
            messages,
            temperature,
            max_tokens,
            stream
        } = req.body;

        const nimModel =
            typeof model === 'string' &&
            model.trim()
                ? model.trim()
                : 'meta/llama-3.1-8b-instruct';

        console.log(
            '========== NVIDIA REQUEST =========='
        );
        console.log(
            'Requested model:',
            model
        );
        console.log(
            'Using model:',
            nimModel
        );
        console.log(
            'API Base:',
            NIM_API_BASE
        );
        console.log(
            'API Key:',
            NIM_API_KEY
                ? NIM_API_KEY.substring(0,10)
                    + '...'
                : 'MISSING'
        );
        console.log(
            '===================================='
        );

        const nimRequest = {
            model: nimModel,
            messages: messages,
            temperature:
                temperature ?? 0.6,
            max_tokens:
                Math.min(
                    max_tokens ?? 2048,
                    8192
                ),
            stream:
                stream || false,
            extra_body:
                ENABLE_THINKING_MODE
                    ? {
                          chat_template_kwargs:
                              {
                                  thinking:
                                      true
                              }
                      }
                    : undefined
        };

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

        // STREAM
        if (stream) {
            res.setHeader(
                'Content-Type',
                'text/event-stream'
            );
            res.setHeader(
                'Cache-Control',
                'no-cache'
            );
            res.setHeader(
                'Connection',
                'keep-alive'
            );

            response.data.pipe(res);
            return;
        }

        // NORMAL
        const openaiResponse = {
            id:
                `chatcmpl-${Date.now()}`,
            object:
                'chat.completion',
            created:
                Math.floor(
                    Date.now()/1000
                ),
            model:
                nimModel,
            choices:
                response.data.choices.map(
                    choice => {
                        let content =
                            choice
                                .message
                                ?.content ||
                            '';

                        if (
                            SHOW_REASONING &&
                            choice
                                .message
                                ?.reasoning_content
                        ) {
                            content =
                                '<think>\n' +
                                choice
                                    .message
                                    .reasoning_content +
                                '\n</think>\n\n' +
                                content;
                        }

                        return {
                            index:
                                choice.index,
                            message: {
                                role:
                                    choice
                                        .message
                                        .role,
                                content
                            },
                            finish_reason:
                                choice.finish_reason
                        };
                    }
                ),
            usage:
                response.data
                    .usage || {
                    prompt_tokens:
                        0,
                    completion_tokens:
                        0,
                    total_tokens:
                        0
                }
        };

        res.json(
            openaiResponse
        );

    } catch (error) {

        console.error(
            '===================='
        );
        console.error(
            'Proxy error status:',
            error.response
                ?.status
        );
        console.error(
            'Proxy error data:',
            JSON.stringify(
                error.response
                    ?.data,
                null,
                2
            )
        );
        console.error(
            'Proxy error message:',
            error.message
        );
        console.error(
            '===================='
        );

        res.status(
            error.response
                ?.status || 500
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

// OpenAI endpoint
app.post(
    '/v1/chat/completions',
    handleChat
);

// Janitor AI endpoint
app.post(
    '/v1',
    handleChat
);

// Alternate endpoint
app.post(
    '/chat/completions',
    handleChat
);

// Catch all
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