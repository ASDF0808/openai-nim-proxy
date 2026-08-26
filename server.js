// server.js
// OpenAI ↔ NVIDIA NIM Proxy
// Render / Railway compatible
//
// Features:
// - OpenAI-compatible API
// - NVIDIA NIM /v1/chat/completions
// - Streaming support
// - Non-streaming support
// - Model mapping
// - Direct NVIDIA model support
// - Optional CLIENT_AUTH_KEY
// - 50MB request body limit
// - NVIDIA error diagnostics
// - Startup model validation
//
// IMPORTANT:
// No automatic fallback is used.
// If a model returns 404/410/etc., that exact error is returned.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();

const PORT = process.env.PORT || 3000;

// ============================================================
// CONFIGURATION
// ============================================================

const NIM_API_BASE =
    process.env.NIM_API_BASE ||
    'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Optional.
// If this is set, clients must send:
// Authorization: Bearer YOUR_CLIENT_AUTH_KEY
//
// If it is NOT set, proxy authentication is disabled.
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING =
    process.env.SHOW_REASONING === 'true';

const ENABLE_THINKING_MODE =
    process.env.ENABLE_THINKING_MODE === 'true';

const SKIP_VALIDATION =
    process.env.SKIP_VALIDATION === 'true';

// Optional Discord webhook.
const DISCORD_WEBHOOK_URL =
    process.env.DISCORD_WEBHOOK_URL;

// ============================================================
// LIMITS
// ============================================================

// Janitor AI can send large conversation histories.
// 50MB prevents the previous 413 PayloadTooLargeError.
const REQUEST_BODY_LIMIT = '50mb';

// Maximum max_tokens accepted from the client.
const MAX_TOKENS_LIMIT = 65536;

// NVIDIA request timeout.
// This is intentionally high because some models can take
// considerable time before the first streaming chunk.
const REQUEST_TIMEOUT_MS = 180000;

// Timeout for startup /models validation.
const VALIDATION_TIMEOUT_MS = 15000;

// Maximum internal SSE buffer.
const MAX_STREAM_BUFFER_SIZE = 1024 * 1024;

// ============================================================
// STARTUP CONFIG LOG
// ============================================================

console.log('==============================================');
console.log('SERVER VERSION: OPENAI-NIM-PROXY-FINAL');
console.log('==============================================');

console.log('NIM API Base:', NIM_API_BASE);
console.log('NIM API Key:', NIM_API_KEY
    ? `${NIM_API_KEY.substring(0, 10)}...`
    : 'NOT SET');

console.log(
    'Client authentication:',
    CLIENT_AUTH_KEY ? 'ENABLED' : 'DISABLED'
);

console.log(
    'Reasoning display:',
    SHOW_REASONING ? 'ENABLED' : 'DISABLED'
);

console.log(
    'Thinking mode:',
    ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'
);

console.log(
    'Request body limit:',
    REQUEST_BODY_LIMIT
);

console.log(
    'Request timeout:',
    `${REQUEST_TIMEOUT_MS}ms`
);

console.log('==============================================');

// ============================================================
// CONFIG VALIDATION
// ============================================================

function validateConfig() {
    if (!NIM_API_KEY) {
        console.error(
            '[FATAL] NIM_API_KEY is not set.'
        );

        console.error(
            '[FATAL] Set NIM_API_KEY in Render Environment Variables.'
        );

        process.exit(1);
    }

    if (!CLIENT_AUTH_KEY) {
        console.warn(
            '[AUTH] CLIENT_AUTH_KEY is not set.'
        );

        console.warn(
            '[AUTH] Client authentication is DISABLED.'
        );

        console.warn(
            '[AUTH] Anyone who knows this URL can send requests.'
        );
    } else {
        console.log(
            '[AUTH] CLIENT_AUTH_KEY is configured.'
        );
    }
}

validateConfig();

// ============================================================
// MODEL MAPPING
// ============================================================
//
// The LEFT side is what Janitor AI sends.
// The RIGHT side is what NVIDIA receives.
//
// Keep your manually verified NVIDIA model IDs here.
//

const MODEL_MAPPING = {

    // OpenAI-style aliases
    'gpt-3.5-turbo':
        'nvidia/nemotron-3-super-120b-a12b',

    'gpt-4':
        'nvidia/nemotron-3-ultra-550b-a55b',

    'gpt-3.5':
        'qwen/qwen3.5-397b-a17b',

    'gpt-4-turbo':
        'moonshotai/kimi-k2.6',

    // Manually verified by you.
    'gpt-4o':
        'deepseek-ai/deepseek-v4-flash-0731',

    'claude-3-opus':
        'openai/gpt-oss-120b',

    'claude-3-sonnet':
        'openai/gpt-oss-20b',

    'gemini-pro':
        'nvidia/llama-3.3-nemotron-super-49b-v1.5',

    'gemini-turbo':
        'meta/llama-3.3-70b-instruct',

    'gemini-turbo?':
        'abacusai/dracarys-llama-3.1-70b-instruct',

    'gpt-3.5o':
        'nvidia/nemotron-mini-4b-instruct',

    'gpt-4-flash':
        'deepseek-ai/deepseek-v4-flash',

    'glm-5.2':
        'z-ai/glm-5.2',

    'mistral':
        'mistralai/mistral-large-3-675b-instruct-2512',

    'mistral-turbo':
        'mistralai/mistral-medium-3.5-128b',

    'mistral-pro':
        'mistralai/mistral-small-4-119b-2603',

    'mistral-nemo':
        'mistralai/mistral-nemotron',

    'mistral-fast':
        'mistralai/ministral-14b-instruct-2512',

    'google-light':
        'google/gemma-4-31b-it',

    'google-lightest':
        'google/gemma-2-2b-it',

    'google-lighter':
        'google/gemma-3n-e4b-it',

    'm2.7':
        'minimaxai/minimax-m2.7',

    'm3':
        'minimaxai/minimax-m3',

    'step-3.5-flash':
        'stepfun-ai/step-3.5-flash',

    'step-3.7-flash':
        'stepfun-ai/step-3.7-flash'
};

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors({
    origin: '*',
    methods: [
        'GET',
        'POST',
        'OPTIONS'
    ],
    allowedHeaders: [
        'Content-Type',
        'Authorization'
    ]
}));

// IMPORTANT:
// 50MB instead of the previous 10MB.
app.use(express.json({
    limit: REQUEST_BODY_LIMIT
}));

app.use(express.urlencoded({
    extended: true,
    limit: REQUEST_BODY_LIMIT
}));

// ============================================================
// AUTHENTICATION
// ============================================================

function extractBearerToken(authHeader) {

    if (!authHeader) {
        return null;
    }

    if (typeof authHeader !== 'string') {
        return null;
    }

    const parts = authHeader
        .trim()
        .split(/\s+/);

    if (parts.length !== 2) {
        return null;
    }

    if (parts[0].toLowerCase() !== 'bearer') {
        return null;
    }

    return parts[1];
}


function safeTimingEqual(a, b) {

    if (!a || !b) {
        return false;
    }

    if (a.length !== b.length) {
        return false;
    }

    try {

        return timingSafeEqual(
            Buffer.from(a),
            Buffer.from(b)
        );

    } catch {

        return false;
    }
}


// Authentication is OPTIONAL.
//
// If CLIENT_AUTH_KEY is not configured:
//     requests are allowed.
//
// If CLIENT_AUTH_KEY is configured:
//     Authorization: Bearer CLIENT_AUTH_KEY
//     is required.
//
// /health and /v1/models remain public.

app.use((req, res, next) => {

    // Public endpoints
    if (
        req.path === '/health' ||
        req.path === '/v1/models'
    ) {
        return next();
    }

    // Authentication disabled.
    if (!CLIENT_AUTH_KEY) {
        return next();
    }

    const token =
        extractBearerToken(
            req.headers.authorization
        );

    if (!token) {

        return res.status(403).json({
            error: {
                message:
                    'Forbidden: Invalid or missing authentication',
                type: 'authentication_error',
                code: 403
            }
        });
    }

    if (
        !safeTimingEqual(
            token,
            CLIENT_AUTH_KEY
        )
    ) {

        return res.status(403).json({
            error: {
                message:
                    'Forbidden: Invalid authentication credentials',
                type: 'authentication_error',
                code: 403
            }
        });
    }

    next();
});

// ============================================================
// MODEL RESOLUTION
// ============================================================

function resolveModel(requestedModel) {

    if (!requestedModel) {
        return null;
    }

    // First:
    // Check OpenAI/JANITOR alias.
    if (
        Object.prototype.hasOwnProperty.call(
            MODEL_MAPPING,
            requestedModel
        )
    ) {
        return MODEL_MAPPING[requestedModel];
    }

    // Second:
    // If Janitor sends an actual NVIDIA model ID directly,
    // preserve it exactly.
    //
    // Example:
    //
    // model:
    // "meta/llama-3.1-8b-instruct"
    //
    // will be sent to NVIDIA unchanged.

    return requestedModel;
}

// ============================================================
// SAFE RESPONSE WRITING
// ============================================================

function safeWrite(res, data) {

    try {

        if (
            !res.writableEnded &&
            !res.destroyed &&
            res.writable
        ) {

            res.write(data);
            return true;
        }

    } catch (err) {

        console.warn(
            '[STREAM] res.write failed:',
            err.message
        );
    }

    return false;
}

// ============================================================
// NVIDIA ERROR EXTRACTION
// ============================================================

function getNvidiaErrorData(error) {

    if (!error) {
        return null;
    }

    const response = error.response;

    if (!response) {
        return null;
    }

    return response.data;
}


function getNvidiaErrorMessage(error) {

    const data =
        getNvidiaErrorData(error);

    if (!data) {
        return error.message || 'Unknown error';
    }

    if (
        typeof data === 'object' &&
        data.error
    ) {

        if (typeof data.error === 'string') {
            return data.error;
        }

        if (data.error.message) {
            return data.error.message;
        }
    }

    if (
        typeof data === 'object' &&
        data.message
    ) {
        return data.message;
    }

    if (typeof data === 'string') {
        return data;
    }

    try {
        return JSON.stringify(data);
    } catch {
        return error.message || 'Unknown error';
    }
}

// ============================================================
// NVIDIA REQUEST
// ============================================================

async function sendToNvidia({
    model,
    messages,
    temperature,
    max_tokens,
    stream,
    extra
}) {

    const url =
        `${NIM_API_BASE}/chat/completions`;

    const nimRequest = {
        model,
        messages,
        temperature:
            temperature ?? 0.7,
        max_tokens:
            Math.min(
                max_tokens ?? 2048,
                MAX_TOKENS_LIMIT
            ),
        stream:
            stream === true
    };

    // Preserve optional extra OpenAI-compatible fields.
    //
    // Some Janitor AI requests may contain:
    // tools
    // tool_choice
    // top_p
    // stop
    // frequency_penalty
    // presence_penalty
    //
    // Copy only when provided.

    if (extra) {

        const allowedExtraFields = [
            'top_p',
            'top_k',
            'min_p',
            'stop',
            'frequency_penalty',
            'presence_penalty',
            'seed',
            'response_format',
            'tools',
            'tool_choice'
        ];

        for (
            const key of allowedExtraFields
        ) {

            if (
                extra[key] !== undefined
            ) {

                nimRequest[key] =
                    extra[key];
            }
        }
    }

    // Thinking mode.
    if (ENABLE_THINKING_MODE) {

        nimRequest.extra_body = {
            chat_template_kwargs: {
                thinking: true
            }
        };
    }

    console.log('');
    console.log(
        '========== NVIDIA REQUEST =========='
    );

    console.log(
        'URL:',
        url
    );

    console.log(
        'Model:',
        model
    );

    console.log(
        'Stream:',
        nimRequest.stream
    );

    console.log(
        'Temperature:',
        nimRequest.temperature
    );

    console.log(
        'Max Tokens:',
        nimRequest.max_tokens
    );

    console.log(
        'Messages:',
        Array.isArray(messages)
            ? messages.length
            : 0
    );

    console.log(
        '===================================='
    );

    return axios.post(
        url,
        nimRequest,
        {
            headers: {
                Authorization:
                    `Bearer ${NIM_API_KEY}`,

                'Content-Type':
                    'application/json',

                Accept:
                    stream
                        ? 'text/event-stream'
                        : 'application/json'
            },

            responseType:
                stream
                    ? 'stream'
                    : 'json',

            timeout:
                REQUEST_TIMEOUT_MS,

            // Do not let Axios convert HTTP errors
            // into some other custom behavior.
            validateStatus:
                status =>
                    status >= 200 &&
                    status < 300
        }
    );
}

// ============================================================
// HEALTH
// ============================================================

app.get('/health', (req, res) => {

    res.json({

        status: 'ok',

        service:
            'OpenAI to NVIDIA NIM Proxy',

        version:
            'FINAL-2026',

        reasoning_display:
            SHOW_REASONING,

        thinking_mode:
            ENABLE_THINKING_MODE,

        client_auth:
            Boolean(CLIENT_AUTH_KEY),

        model_count:
            Object.keys(MODEL_MAPPING).length
    });
});

// ============================================================
// MODELS
// ============================================================
//
// Return Janitor/OpenAI aliases.
//
// Janitor can therefore select:
// gpt-4o
// gpt-4
// claude-3-opus
// etc.
//
// The proxy then translates them through MODEL_MAPPING.

app.get('/v1/models', (req, res) => {

    const models =
        Object.keys(MODEL_MAPPING)
            .map(model => ({

                id: model,

                object: 'model',

                created:
                    Math.floor(
                        Date.now() / 1000
                    ),

                owned_by:
                    'nvidia-nim-proxy'
            }));

    res.json({

        object: 'list',

        data: models
    });
});

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post(
    '/v1/chat/completions',
    async (req, res) => {

        let upstreamStream = null;
        let streamFinished = false;

        try {

            const {
                model,
                messages,
                temperature,
                max_tokens,
                stream,

                // Preserve additional request parameters.
                top_p,
                top_k,
                min_p,
                stop,
                frequency_penalty,
                presence_penalty,
                seed,
                response_format,
                tools,
                tool_choice
            } = req.body || {};

            // ------------------------------------------------
            // Basic validation
            // ------------------------------------------------

            if (!model) {

                return res.status(400).json({
                    error: {
                        message:
                            'Missing model',
                        type:
                            'invalid_request_error',
                        code:
                            400
                    }
                });
            }

            if (
                !Array.isArray(messages)
            ) {

                return res.status(400).json({
                    error: {
                        message:
                            'messages must be an array',
                        type:
                            'invalid_request_error',
                        code:
                            400
                    }
                });
            }

            // ------------------------------------------------
            // Resolve model
            // ------------------------------------------------

            const nimModel =
                resolveModel(model);

            if (!nimModel) {

                return res.status(400).json({
                    error: {
                        message:
                            'Unable to resolve model',
                        type:
                            'invalid_request_error',
                        code:
                            400
                    }
                });
            }

            // ------------------------------------------------
            // Request logging
            // ------------------------------------------------

            let requestSize = 0;

            try {

                requestSize =
                    Buffer.byteLength(
                        JSON.stringify(
                            req.body
                        ),
                        'utf8'
                    );

            } catch {
                requestSize = 0;
            }

            console.log('');
            console.log(
                '========== REQUEST =========='
            );

            console.log(
                'Requested model:',
                model
            );

            console.log(
                'NVIDIA model:',
                nimModel
            );

            console.log(
                'Stream:',
                stream === true
            );

            console.log(
                'Temperature:',
                temperature
            );

            console.log(
                'Max Tokens:',
                max_tokens
            );

            console.log(
                'Messages:',
                messages.length
            );

            console.log(
                'Request Size:',
                `${requestSize} bytes`
            );

            console.log(
                '============================='
            );

            // ------------------------------------------------
            // Extra fields
            // ------------------------------------------------

            const extra = {
                top_p,
                top_k,
                min_p,
                stop,
                frequency_penalty,
                presence_penalty,
                seed,
                response_format,
                tools,
                tool_choice
            };

            // ------------------------------------------------
            // NVIDIA request
            // ------------------------------------------------

            const response =
                await sendToNvidia({
                    model:
                        nimModel,

                    messages:
                        messages,

                    temperature:
                        temperature,

                    max_tokens:
                        max_tokens,

                    stream:
                        stream === true,

                    extra:
                        extra
                });

            console.log(
                '[PROXY] NVIDIA response:',
                response.status
            );

            // =================================================
            // STREAMING
            // =================================================

            if (stream === true) {

                upstreamStream =
                    response.data;

                res.statusCode = 200;

                res.setHeader(
                    'Content-Type',
                    'text/event-stream'
                );

                res.setHeader(
                    'Cache-Control',
                    'no-cache, no-transform'
                );

                res.setHeader(
                    'Connection',
                    'keep-alive'
                );

                res.setHeader(
                    'X-Accel-Buffering',
                    'no'
                );

                if (
                    typeof res.flushHeaders ===
                    'function'
                ) {
                    res.flushHeaders();
                }

                const decoder =
                    new StringDecoder('utf8');

                let buffer = '';

                let reasoningOpen =
                    false;

                let doneSent =
                    false;

                // ------------------------------------------------
                // Send [DONE]
                // ------------------------------------------------

                function sendDone() {

                    if (doneSent) {
                        return;
                    }

                    safeWrite(
                        res,
                        'data: [DONE]\n\n'
                    );

                    doneSent = true;
                    streamFinished = true;
                }

                // ------------------------------------------------
                // Process SSE line
                // ------------------------------------------------

                function processSSELine(line) {

                    line =
                        line.replace(
                            /\r$/,
                            ''
                        );

                    // Ignore comments / empty lines.
                    if (
                        !line ||
                        !line.startsWith(
                            'data:'
                        )
                    ) {
                        return;
                    }

                    const payload =
                        line
                            .slice(5)
                            .trim();

                    if (
                        payload === '[DONE]'
                    ) {

                        sendDone();
                        return;
                    }

                    let data;

                    try {

                        data =
                            JSON.parse(
                                payload
                            );

                    } catch (err) {

                        console.warn(
                            '[STREAM] Invalid JSON:',
                            payload.slice(
                                0,
                                300
                            )
                        );

                        return;
                    }

                    // --------------------------------------------
                    // Reasoning handling
                    // --------------------------------------------

                    const delta =
                        data
                            ?.choices?.[0]
                            ?.delta;

                    if (delta) {

                        const reasoning =
                            delta.reasoning_content;

                        const originalContent =
                            delta.content;

                        let content =
                            originalContent ||
                            '';

                        if (
                            SHOW_REASONING &&
                            reasoning
                        ) {

                            if (
                                !reasoningOpen
                            ) {

                                content =
                                    '<think>\n' +
                                    reasoning;

                                reasoningOpen =
                                    true;

                            } else {

                                content =
                                    reasoning;
                            }
                        }

                        if (
                            SHOW_REASONING &&
                            originalContent &&
                            reasoningOpen
                        ) {

                            content +=
                                '\n</think>\n\n' +
                                originalContent;

                            reasoningOpen =
                                false;
                        }

                        delta.content =
                            content;

                        delete delta.reasoning_content;
                    }

                    safeWrite(
                        res,
                        `data: ${JSON.stringify(data)}\n\n`
                    );
                }

                // ------------------------------------------------
                // NVIDIA stream data
                // ------------------------------------------------

                upstreamStream.on(
                    'data',
                    chunk => {

                        if (
                            res.writableEnded ||
                            res.destroyed
                        ) {
                            return;
                        }

                        buffer +=
                            decoder.write(
                                chunk
                            );

                        if (
                            Buffer.byteLength(
                                buffer,
                                'utf8'
                            ) >
                            MAX_STREAM_BUFFER_SIZE
                        ) {

                            console.error(
                                '[STREAM] Buffer exceeded limit.'
                            );

                            safeWrite(
                                res,
                                `data: ${JSON.stringify({
                                    error: {
                                        message:
                                            'Stream buffer exceeded limit',
                                        type:
                                            'stream_error',
                                        code:
                                            500
                                    }
                                })}\n\n`
                            );

                            sendDone();

                            res.end();

                            upstreamStream.destroy();

                            return;
                        }

                        const lines =
                            buffer.split(
                                '\n'
                            );

                        buffer =
                            lines.pop() || '';

                        for (
                            const line
                            of lines
                        ) {

                            processSSELine(
                                line
                            );
                        }
                    }
                );

                // ------------------------------------------------
                // NVIDIA stream end
                // ------------------------------------------------

                upstreamStream.on(
                    'end',
                    () => {

                        try {

                            buffer +=
                                decoder.end();

                            if (
                                buffer.trim()
                            ) {

                                for (
                                    const line
                                    of buffer.split(
                                        '\n'
                                    )
                                ) {

                                    processSSELine(
                                        line
                                    );
                                }
                            }

                        } catch (
                            err
                        ) {

                            console.error(
                                '[STREAM] Final buffer error:',
                                err.message
                            );
                        }

                        sendDone();

                        if (
                            !res.writableEnded
                        ) {
                            res.end();
                        }

                        streamFinished =
                            true;

                        console.log(
                            '[STREAM] Completed.'
                        );
                    }
                );

                // ------------------------------------------------
                // NVIDIA stream error
                // ------------------------------------------------

                upstreamStream.on(
                    'error',
                    err => {

                        console.error(
                            '[STREAM] NVIDIA stream error:',
                            err.message
                        );

                        if (
                            !res.writableEnded
                        ) {

                            safeWrite(
                                res,
                                `data: ${JSON.stringify({
                                    error: {
                                        message:
                                            err.message,
                                        type:
                                            'stream_error',
                                        code:
                                            502
                                    }
                                })}\n\n`
                            );

                            sendDone();

                            res.end();
                        }
                    }
                );

                // ------------------------------------------------
                // Client disconnect
                // ------------------------------------------------

                req.on(
                    'close',
                    () => {

                        if (
                            !streamFinished &&
                            upstreamStream &&
                            !upstreamStream.destroyed
                        ) {

                            console.log(
                                '[STREAM] Client disconnected.'
                            );

                            upstreamStream.destroy();
                        }
                    }
                );

                return;
            }

            // =================================================
            // NON-STREAMING
            // =================================================

            const data =
                response.data;

            const choices =
                Array.isArray(
                    data.choices
                )
                    ? data.choices
                    : [];

            const openaiResponse = {

                id:
                    data.id ||
                    `chatcmpl-${Date.now()}`,

                object:
                    'chat.completion',

                created:
                    data.created ||
                    Math.floor(
                        Date.now() / 1000
                    ),

                model:
                    model,

                choices:
                    choices.map(
                        (choice, index) => {

                            let content =
                                choice
                                    ?.message
                                    ?.content ||
                                '';

                            if (
                                SHOW_REASONING &&
                                choice
                                    ?.message
                                    ?.reasoning_content
                            ) {

                                content =
                                    '<think>\n' +
                                    choice.message.reasoning_content +
                                    '\n</think>\n\n' +
                                    content;
                            }

                            const message = {

                                role:
                                    choice
                                        ?.message
                                        ?.role ||
                                    'assistant',

                                content:
                                    content
                            };

                            if (
                                choice
                                    ?.message
                                    ?.tool_calls
                            ) {

                                message.tool_calls =
                                    choice
                                        .message
                                        .tool_calls;
                            }

                            return {

                                index:
                                    choice.index ??
                                    index,

                                message:

                                    message,

                                finish_reason:
                                    choice
                                        .finish_reason ||
                                    'stop'
                            };
                        }
                    ),

                usage:
                    data.usage ||
                    {
                        prompt_tokens:
                            0,

                        completion_tokens:
                            0,

                        total_tokens:
                            0
                    }
            };

            return res.json(
                openaiResponse
            );

        } catch (error) {

            // =================================================
            // ERROR LOGGING
            // =================================================

            const status =
                error
                    ?.response
                    ?.status ||
                500;

            const errorData =
                error
                    ?.response
                    ?.data;

            console.error('');
            console.error(
                '========== PROXY ERROR =========='
            );

            console.error(
                'Status:',
                status
            );

            console.error(
                'Message:',
                error.message
            );

            console.error(
                'NVIDIA response:',
                typeof errorData ===
                    'string'
                    ? errorData
                    : JSON.stringify(
                        errorData,
                        null,
                        2
                    )
            );

            console.error(
                'URL:',
                error
                    ?.config
                    ?.url
            );

            console.error(
                'Method:',
                error
                    ?.config
                    ?.method
            );

            console.error(
                '=================================='
            );

            // =================================================
            // Build useful OpenAI-style error
            // =================================================

            const message =
                getNvidiaErrorMessage(
                    error
                );

            if (
                !res.headersSent
            ) {

                return res
                    .status(status)
                    .json({

                        error: {

                            message:
                                message,

                            type:
                                status === 401 ||
                                status === 403
                                    ? 'authentication_error'
                                    : 'invalid_request_error',

                            code:
                                status
                        }
                    });
            }

            // Streaming error after headers
            if (
                !res.writableEnded
            ) {

                safeWrite(
                    res,
                    `data: ${JSON.stringify({
                        error: {
                            message:
                                message,
                            type:
                                'proxy_error',
                            code:
                                status
                        }
                    })}\n\n`
                );

                safeWrite(
                    res,
                    'data: [DONE]\n\n'
                );

                res.end();
            }

            if (
                upstreamStream &&
                !upstreamStream.destroyed
            ) {

                upstreamStream.destroy();
            }
        }
    }
);

// ============================================================
// 413 HANDLER
// ============================================================
//
// This catches requests larger than the body parser limit.

app.use(
    (err, req, res, next) => {

        if (
            err &&
            (
                err.type ===
                'entity.too.large' ||
                err.name ===
                'PayloadTooLargeError'
            )
        ) {

            console.error(
                '[HTTP 413] Request entity too large.'
            );

            return res
                .status(413)
                .json({

                    error: {

                        message:
                            'Request payload is too large. Maximum size is 50MB.',

                        type:
                            'invalid_request_error',

                        code:
                            413
                    }
                });
        }

        next(err);
    }
);

// ============================================================
// GENERIC ERROR HANDLER
// ============================================================

app.use(
    (err, req, res, next) => {

        console.error(
            '[EXPRESS ERROR]',
            err
        );

        if (
            res.headersSent
        ) {
            return next(err);
        }

        res.status(
            err.status ||
            500
        ).json({

            error: {

                message:
                    err.message ||
                    'Internal server error',

                type:
                    'server_error',

                code:
                    err.status ||
                    500
            }
        });
    }
);

// ============================================================
// 404
// ============================================================

app.use(
    (req, res) => {

        res.status(404).json({

            error: {

                message:
                    `Endpoint ${req.method} ${req.path} not found`,

                type:
                    'invalid_request_error',

                code:
                    404
            }
        });
    }
);

// ============================================================
// STARTUP MODEL VALIDATION
// ============================================================
//
// This ONLY calls /v1/models.
// It does NOT send test generations.
// Therefore it does not consume inference quota
// and does not cause the previous fallback problem.
//

async function validateModels() {

    if (SKIP_VALIDATION) {

        console.log(
            '[VALIDATION] Skipped.'
        );

        return;
    }

    console.log('');
    console.log(
        '========== MODEL VALIDATION =========='
    );

    try {

        const response =
            await axios.get(
                `${NIM_API_BASE}/models`,
                {
                    headers: {

                        Authorization:
                            `Bearer ${NIM_API_KEY}`,

                        'Content-Type':
                            'application/json'
                    },

                    timeout:
                        VALIDATION_TIMEOUT_MS
                }
            );

        const availableModels =
            new Set(
                (
                    response
                        ?.data
                        ?.data ||
                    []
                ).map(
                    model =>
                        model.id
                )
            );

        let validCount =
            0;

        let invalidCount =
            0;

        for (
            const [
                alias,
                nimModel
            ]
            of Object.entries(
                MODEL_MAPPING
            )
        ) {

            if (
                availableModels.has(
                    nimModel
                )
            ) {

                console.log(
                    `[VALIDATION] ✓ ${alias} → ${nimModel}`
                );

                validCount++;

            } else {

                console.warn(
                    `[VALIDATION] ✗ ${alias} → ${nimModel} (not in catalog)`
                );

                invalidCount++;
            }
        }

        console.log(
            `[VALIDATION] Valid: ${validCount}`
        );

        console.log(
            `[VALIDATION] Not in catalog: ${invalidCount}`
        );

        console.log(
            '======================================'
        );

    } catch (error) {

        console.warn(
            '[VALIDATION] Could not retrieve NVIDIA model list.'
        );

        console.warn(
            '[VALIDATION]',
            error.message
        );

        console.warn(
            '[VALIDATION] Server will continue normally.'
        );

        console.log(
            '======================================'
        );
    }
}

// ============================================================
// START SERVER
// ============================================================

app.listen(
    PORT,
    () => {

        console.log('');
        console.log(
            '=============================================='
        );

        console.log(
            'OpenAI to NVIDIA NIM Proxy running on port',
            PORT
        );

        console.log(
            `Health check: http://localhost:${PORT}/health`
        );

        console.log(
            'Reasoning display:',
            SHOW_REASONING
                ? 'ENABLED'
                : 'DISABLED'
        );

        console.log(
            'Thinking mode:',
            ENABLE_THINKING_MODE
                ? 'ENABLED'
                : 'DISABLED'
        );

        console.log(
            'Client authentication:',
            CLIENT_AUTH_KEY
                ? 'ENABLED'
                : 'DISABLED'
        );

        console.log(
            '=============================================='
        );

        // Non-blocking validation.
        validateModels()
            .catch(
                error => {

                    console.error(
                        '[VALIDATION] Unexpected error:',
                        error.message
                    );
                }
            );
    }
);