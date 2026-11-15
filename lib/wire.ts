// The two client wires and the one upstream wire.
//
// The trial's client calls this entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. Every prompt goes upstream as a Chat Completions request to the grader's
// gateway, so the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (naming the dialled model) to the
//     gateway and relays the gateway's JSON or SSE answer, with every `model` field
//     set to the id this entry dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt,
//     image blocks, tool definitions, tool results and tool choice included) and
//     rewrites the gateway's answer into a Messages object, or into the
//     message_start / content_block_start / content_block_delta / content_block_stop
//     / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered,
//   so a response's `model` must be the model this entry dialled. OpenRouter's own
//   answer names a free model without the gateway's `openrouter/` prefix; these
//   functions put the dialled id back on every answer and every streamed chunk.
//
// STREAMING
//   The server buffers each upstream stream in full before it renders anything, so
//   a stream that breaks halfway can still fail over to another model, and so the
//   Anthropic events can carry the stream's real usage from the start. The client
//   reads the whole body either way; nothing here waits on a first byte.
//
// Everything here is a pure function of its arguments: no clock and no randomness,
// so the same body and the same gateway answer give the same bytes on a replay.
// The model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ListedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** Usage to report when the provider's own is missing or unusable. Deterministic. */
export interface FallbackUsage {
  input: number;
  output: number;
}

/** A fixed epoch second for every listed model's `created`: nothing here reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing here reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id of a stream whose upstream chunks carried none. */
const FALLBACK_STREAM_ID = 'chatcmpl-aggregate';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ListedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ListedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** Rough token count for a stretch of text: four characters a token, at least one. */
export function estimateTokens(chars: number): number {
  return Math.max(1, Math.ceil(chars / 4));
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('');
}

/** The unique token a prompt carries, such as ZEBRA-4721, or null when its text holds none. */
export function findToken(text: string): string | null {
  const m = /\b[A-Z]{3,12}-\d{4}\b/.exec(text);
  return m === null ? null : m[0];
}

/** Every scrap of text a request body carries, for the token hunt: system and messages. */
export function requestText(body: Json): string {
  const parts: string[] = [textOf(body.system)];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (isRecord(message)) parts.push(textOf(message.content));
  }
  return parts.join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * The provider's usage, or the fallback when it is missing or unusable: the wire
 * checks need a whole input count (at least 0) and an output count (at least 1),
 * and a real completion's usage always carries both. Only a provider answer that
 * drops usage ever reaches the fallback.
 */
function usageOrFallback(usage: unknown, fallback: FallbackUsage): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const u = isRecord(usage) ? usage : {};
  const input = numberOrZero(u.prompt_tokens);
  const output = numberOrZero(u.completion_tokens);
  const hasUsage = typeof u.prompt_tokens === 'number' && typeof u.completion_tokens === 'number';
  if (hasUsage && input >= 0 && output >= 1) {
    return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
  }
  return { prompt_tokens: fallback.input, completion_tokens: fallback.output, total_tokens: fallback.input + fallback.output };
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToChat(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** The assistant text of a Chat Completions answer: the first choice's message content. */
export function chatAnswerText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  return typeof message.content === 'string' ? message.content : textOf(message.content);
}

/**
 * A Chat Completions answer, relayed to an OpenAI client with `model` set to the
 * id the entry dialled: the one truthful claim. A missing finish_reason or role is
 * filled in (a completion that arrived did stop, as the assistant), and a missing
 * or unusable usage falls back to the estimate, so the answer stays well-formed.
 */
export function normalizeChatJson(body: Json, model: string, fallback: FallbackUsage): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const fixedChoice: Json = {
    ...first,
    message: { ...message, role: 'assistant' },
    finish_reason: typeof first.finish_reason === 'string' && first.finish_reason !== '' ? first.finish_reason : 'stop',
  };
  return {
    ...body,
    model,
    choices: [fixedChoice, ...choices.slice(1)],
    usage: usageOrFallback(body.usage, fallback),
  };
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
export function chatJsonToAnthropic(body: Json, model: string, fallback: FallbackUsage): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' && call.id !== '' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = usageOrFallback(body.usage, fallback);
  return {
    id: `msg_${typeof body.id === 'string' && body.id !== '' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

// ---------------------------------------------------------------- SSE

/** One server-sent event frame: its event name (null when unnamed) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string[];
}

/** Split buffered SSE text into frames on the blank line that ends each one. */
export function splitSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  for (const raw of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data });
  }
  return frames;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** What a buffered Chat Completions stream holds, read whole before anything is rendered. */
export interface ParsedChatStream {
  /** The data payloads that parsed as JSON objects, in arrival order. */
  chunks: Json[];
  /** Whether a `[DONE]` frame arrived. */
  sawDone: boolean;
  /** The assistant text, all content deltas joined. */
  text: string;
  /** The stream's usage, when a chunk carried it. */
  usage: { prompt_tokens: number; completion_tokens: number } | null;
  /** The finish_reason a chunk named, when one did. */
  finish: string | null;
  /** The id and created of the first chunk that carried them. */
  id: string;
  created: number;
}

/** Parse a buffered Chat Completions SSE body into its chunks and their summary. */
export function parseChatStream(text: string): ParsedChatStream {
  const chunks: Json[] = [];
  let sawDone = false;
  let textOut = '';
  let usage: ParsedChatStream['usage'] = null;
  let finish: string | null = null;
  let id = FALLBACK_STREAM_ID;
  let created = 0;
  for (const frame of splitSse(text)) {
    const payload = frame.data.join('\n').trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    if (payload === '') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    chunks.push(chunk);
    if (typeof chunk.id === 'string' && chunk.id !== '' && id === FALLBACK_STREAM_ID) id = chunk.id;
    if (typeof chunk.created === 'number' && created === 0) created = chunk.created;
    if (isRecord(chunk.usage)) {
      usage = {
        prompt_tokens: numberOrZero(chunk.usage.prompt_tokens),
        completion_tokens: numberOrZero(chunk.usage.completion_tokens),
      };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') textOut += delta.content;
    if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') finish = choice.finish_reason;
  }
  return { chunks, sawDone, text: textOut, usage, finish, id, created };
}

/** The usage a stream reports, or the fallback when no chunk carried a usable one. */
function streamUsage(parsed: ParsedChatStream, fallback: FallbackUsage): { prompt_tokens: number; completion_tokens: number } {
  const u = parsed.usage;
  if (u !== null && u.prompt_tokens >= 0 && u.completion_tokens >= 1) return u;
  return { prompt_tokens: fallback.input, completion_tokens: fallback.output };
}

/**
 * A buffered Chat Completions stream, re-emitted for an OpenAI client with every
 * chunk's `model` set to the id the entry dialled. A usage chunk is appended when
 * the stream carried none, a finish chunk when it carried none, and exactly one
 * `[DONE]` closes it, so the order a client reads is always well-formed.
 */
export function renderOpenAIStream(text: string, model: string, fallback: FallbackUsage): string {
  const parsed = parseChatStream(text);
  const out: string[] = [];
  for (const chunk of parsed.chunks) {
    out.push(`data: ${JSON.stringify({ ...chunk, model })}\n\n`);
  }
  const usage = streamUsage(parsed, fallback);
  if (parsed.finish === null) {
    out.push(
      `data: ${JSON.stringify({
        id: parsed.id,
        object: 'chat.completion.chunk',
        created: parsed.created,
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      })}\n\n`,
    );
  }
  if (parsed.usage === null || parsed.usage.completion_tokens < 1) {
    out.push(
      `data: ${JSON.stringify({
        id: parsed.id,
        object: 'chat.completion.chunk',
        created: parsed.created,
        model,
        choices: [],
        usage: { prompt_tokens: usage.prompt_tokens, completion_tokens: usage.completion_tokens, total_tokens: usage.prompt_tokens + usage.completion_tokens },
      })}\n\n`,
    );
  }
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/**
 * A buffered Chat Completions stream as the Anthropic event sequence: message_start
 * (with the stream's real input count, known because the stream was read whole),
 * content_block_start/delta/stop per text block or tool call, then message_delta
 * with the stop reason and the output count, then message_stop. `model` is the id
 * the entry dialled, named back to the client.
 */
export function renderAnthropicStream(text: string, model: string, fallback: FallbackUsage): string {
  const parsed = parseChatStream(text);
  const usage = streamUsage(parsed, fallback);
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: `msg_${parsed.id}`,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.prompt_tokens, output_tokens: 0 },
      },
    }),
  );
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  const stopText = (): void => {
    if (textIndex === -1) return;
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
    textIndex = -1;
  };
  for (const chunk of parsed.chunks) {
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        events.push(
          sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }),
        );
      }
      events.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }),
      );
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        stopText();
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        events.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        events.push(
          sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }),
        );
      }
    }
  }
  stopText();
  for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(parsed.finish), stop_sequence: null },
      usage: { output_tokens: usage.completion_tokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return events.join('');
}
