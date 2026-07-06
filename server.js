// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Request logger
app.use((req, res, next) => {
    console.log(
        `[${new Date().toISOString()}]`,
        req.method,
        req.originalUrl
    );
    next();
});

// NVIDIA NIM API configuration
const NIM_API_BASE =
    process.env.NIM_API_BASE ||
    'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Settings
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy'
    });
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'OpenAI to NVIDIA NIM Proxy',
        reasoning_display: SHOW_REASONING,
        thinking_mode: ENABLE_THINKING_MODE
    });
});

// Models endpoint
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

        // Use model exactly as provided by Janitor AI
        const nimModel =
            (typeof model === 'string' &&
                model.trim())
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
                ? NIM_API_KEY.substring(0, 10) +
                      '...'
                : 'MISSING'
        );
        console.log(
            '===================================='
        );

        const nimRequest = {
            model: nimModel,
            messages,
            temperature:
                temperature ?? 0.6,
            max_tokens: Math.min(
                max_tokens ?? 2048,
                8192
            ),
            extra_body:
                ENABLE_THINKING_MODE
                    ? {
                          chat_template_kwargs:
                              {
                                  thinking: true
                              }
                      }
                    : undefined,
            stream: stream || false
        };

        const response = await axios.post(
            `${NIM_API_BASE}/chat/completions`,
            nimRequest,
            {
                headers: {
                    Authorization:
                        `Bearer ${NIM_API_KEY}`,
                    'Content-Type':
                        'application/json'
                },
                responseType: stream
                    ? 'stream'
                    : 'json'
            }
        );

        // Streaming
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

            let buffer = '';

            response.data.on(
                'data',
                chunk => {
                    buffer +=
                        chunk.toString();

                    const lines =
                        buffer.split('\n');

                    buffer =
                        lines.pop() ||
                        '';

                    lines.forEach(
                        line => {
                            if (
                                line.startsWith(
                                    'data: '
                                )
                            ) {
                                if (
                                    line.includes(
                                        '[DONE]'
                                    )
                                ) {
                                    res.write(
                                        line +
                                            '\n'
                                    );
                                    return;
                                }

                                try {
                                    const data =
                                        JSON.parse(
                                            line.slice(
                                                6
                                            )
                                        );

                                    if (
                                        data
                                            .choices?.[0]
                                            ?.delta
                                            ?.reasoning_content &&
                                        !SHOW_REASONING
                                    ) {
                                        delete data
                                            .choices[0]
                                            .delta
                                            .reasoning_content;
                                    }

                                    res.write(
                                        `data: ${JSON.stringify(
                                            data
                                        )}\n\n`
                                    );
                                } catch {
                                    res.write(
                                        line +
                                            '\n'
                                    );
                                }
                            }
                        }
                    );
                }
            );

            response.data.on(
                'end',
                () => res.end()
            );

            response.data.on(
                'error',
                err => {
                    console.error(
                        'Stream error:',
                        err
                    );
                    res.end();
                }
            );
        }

        // Non-streaming
        else {
            const openaiResponse = {
                id: `chatcmpl-${Date.now()}`,
                object:
                    'chat.completion',
                created: Math.floor(
                    Date.now() / 1000
                ),
                model,
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
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0
                    }
            };

            res.json(
                openaiResponse
            );
        }
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
});

// Alias endpoint
app.post(
    '/chat/completions',
    (req, res, next) => {
        req.url =
            '/v1/chat/completions';
        app._router.handle(
            req,
            res,
            next
        );
    }
);
app.post('/v1', async (req, res) => {
    req.url = '/v1/chat/completions';
    app._router.handle(req, res);
});

// Catch all
app.all('*', (req, res) => {
    res.status(404).json({
        error: {
            message: `Endpoint ${req.path} not found`,
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