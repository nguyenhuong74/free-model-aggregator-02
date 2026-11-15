// A live free-model aggregator: one server, two client wires, a real free model
// behind every answer.
//
// The client points at this server, and every completion it answers comes from a
// real free OpenRouter model dialled through the grader's gateway. The server:
//
//   1. listens on 0.0.0.0 at a free port it picks (never 8080: the env owns it);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <the catalog document>} and is told the trial's prompt count;
//   3. serves GET /v1/models (in the shape of the auth header it got: x-api-key
//      means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//      and POST /v1/messages, each in JSON or server-sent events;
//   4. answers each prompt by dialling a free model through the gateway on its
//      Chat Completions route (an Anthropic-shaped request is translated into a
//      Chat Completions body first, and the answer translated back);
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says
//      {"done": true}. A process still running when the wall ends scores zero.
//
// `node main.ts --list` prints the catalog document and exits 0 before any server
// starts; the register body's catalog is that same document, byte for byte.
//
// ROUTING (quality is 60% of the score)
//   For each request, the candidates are the brief's free models meeting the
//   prompt's requirements (JSON mode, tools, vision, context), ranked by the
//   house's published rating_milli. The best-rated model serves first; a model
//   that refused a call, or answered without the token the prompt asked for, is
//   demoted by 100 milli-points per failure for the rest of the trial, so a
//   rate-limited model stops taking later prompts down with it while a merely
//   unlucky one stays near the top. Every ranking input comes from the brief, the
//   requests and the gateway's answers: no wall clock and no randomness, so a
//   replay makes the same choices.
//
// FAILOVER (10% of the score)
//   A refused gateway call (a 4xx or 5xx, or a connection that never answered) is
//   not an answer: the prompt is retried on the next-ranked free model that meets
//   it, up to five attempts, and every refusal is reported on the response in the
//   `x-makefun-upstream-failures` header, the only way the failover column can see
//   a refusal the server recovered from. A 401 or a trial-level gateway refusal
//   (request_cap, budget_exhausted) stops the attempts at once: no other model
//   would answer those either. An answer that does not carry the token the prompt
//   asked for is likewise retried on the next model: the prompt asked for that
//   token back, and a model that did not follow the instruction has not served it.
//   When every attempt fails, the first completed answer is served if there is
//   one (it may still pass the wire checks); otherwise the client gets a 502 in
//   its own wire, carrying the failure trail.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field is the model it
//   actually dialled. The provider's own answer spells a free model without the
//   gateway's `openrouter/` prefix; this server puts the dialled id back on every
//   answer and every streamed chunk, so its claim is the call it made, never
//   another. A prompt whose reported model has no matching successful call in the
//   house's record earns nothing.
//
// Run with Node 22's standard library alone: nothing is installed, TypeScript
// runs directly, and a relative import needs its .ts extension.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToChat,
  chatAnswerText,
  chatJsonToAnthropic,
  estimateTokens,
  findToken,
  normalizeChatJson,
  openAIError,
  openAIModelList,
  parseChatStream,
  renderAnthropicStream,
  renderOpenAIStream,
  requestText,
  type FallbackUsage,
  type ListedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most bytes of one upstream answer this server buffers. */
const MAX_UPSTREAM_BYTES = 4 * 1024 * 1024;
/** The most gateway attempts one prompt gets, across models. */
const MAX_ATTEMPTS = 5;
/** The milli-points one failure costs a model in the ranking, for the trial. */
const FAILURE_PENALTY_MILLI = 100;
/** The most rows the failure trail header carries. */
const MAX_TRAIL_ROWS = 8;
/** How many times /register is retried while the env is not yet answering. */
const REGISTER_RETRIES = 60;
/** The pause between /register retries, in milliseconds. */
const REGISTER_RETRY_MS = 500;
/** A default output cap, used only when a request names none (the gateway refuses uncapped calls). */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's trial-level refusals: no other model would answer these either. */
const FINAL_CODES = new Set(['budget_exhausted', 'request_cap']);

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key; the gateway swaps it for the entrant's own OpenRouter key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: Catalog = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Failures each model has racked up this trial (refusals and token misses alike). */
const failuresByModel = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible
 * root (`…/v1` on the trial network); a base that does not already end in `/v1`
 * gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ListedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, displayName: model.displayName }));
}

/**
 * The free models that can serve a request, best first: the house's rating, less
 * 100 milli-points per failure the model has racked up this trial, ties broken by
 * the brief's order. Every input is the brief or a gateway answer, so a replay
 * ranks the same way.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const ranked = freeModels(brief)
    .map((model, index) => ({ model, index }))
    .filter(({ model }) => meetsRequirements(requires, model))
    .map(({ model, index }) => ({
      model,
      index,
      score: model.ratingMilli - FAILURE_PENALTY_MILLI * (failuresByModel.get(model.id) ?? 0),
    }));
  ranked.sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked.map((r) => r.model);
}

function noteFailure(modelId: string): void {
  failuresByModel.set(modelId, (failuresByModel.get(modelId) ?? 0) + 1);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, unioned with the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  return {
    ...(entry.requires.tools === true || fromBody.tools === true ? { tools: true } : {}),
    ...(entry.requires.vision === true || fromBody.vision === true ? { vision: true } : {}),
    ...(entry.requires.json === true || fromBody.json === true ? { json: true } : {}),
    ...(entry.requires.minContext !== undefined ? { minContext: entry.requires.minContext } : {}),
  };
}

/** Requirements read off a request body: declared tools, image parts, a JSON-mode response_format. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One row of the failure trail: a refused gateway call, in the order it refused. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

/** The failure trail as the header carries it: at most 8 rows, each code at most 64 characters. */
function trailHeader(failures: readonly FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code.slice(0, 64), model: f.model }));
  return JSON.stringify(rows);
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly FailureRow[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly FailureRow[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

function sendStream(res: ServerResponse, body: string, failures: readonly FailureRow[] = []): void {
  const headers: Record<string, string> = {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': String(Buffer.byteLength(body)),
  };
  const trail = trailHeader(failures);
  if (trail !== null) headers['x-makefun-upstream-failures'] = trail;
  res.writeHead(200, headers);
  res.end(body);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** The error code a refused gateway answer carries, from its error body. */
async function refusalCode(res: Response): Promise<string> {
  let text = '';
  try {
    text = await res.text();
  } catch {
    return '';
  }
  try {
    const value: unknown = JSON.parse(text);
    if (!isRecord(value) || !isRecord(value.error)) return '';
    const code = value.error.code;
    if (typeof code === 'string' && code !== '') return code;
    const type = value.error.type;
    return typeof type === 'string' ? type : '';
  } catch {
    return '';
  }
}

/** One completed upstream answer: the model that served it and its raw body. */
interface CompletedCall {
  model: string;
  text: string;
}

/** What serving one prompt came to: the answer to render, or nothing when every attempt failed. */
interface ServeOutcome {
  answer: CompletedCall | null;
  failures: FailureRow[];
}

/**
 * Serve one prompt: dial the best-ranked free model that meets it, and on a
 * refusal — or an answer that does not carry the token the prompt asked for —
 * retry on the next-ranked one, up to five attempts. The first completed answer
 * is kept as a fallback, so a prompt whose every model missed the token still
 * gets the best model's answer rather than nothing.
 */
async function serveOne(upstreamBody: (model: string) => Json, requires: Requirement, stream: boolean, token: string | null): Promise<ServeOutcome> {
  const failures: FailureRow[] = [];
  const tried = new Set<string>();
  let fallback: CompletedCall | null = null;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    const next = candidatesFor(requires).find((m) => !tried.has(m.id));
    if (next === undefined) break;
    tried.add(next.id);
    let upstream: Response;
    try {
      upstream = await dial(next.id, upstreamBody(next.id));
    } catch (err) {
      log(`dial ${next.id} failed: ${(err as Error).message}`);
      failures.push({ status: 502, code: 'upstream_unreachable', model: next.id });
      noteFailure(next.id);
      continue;
    }
    if (!upstream.ok) {
      const code = await refusalCode(upstream);
      log(`dial ${next.id} refused: HTTP ${upstream.status} ${code}`);
      failures.push({ status: upstream.status, code, model: next.id });
      noteFailure(next.id);
      if (upstream.status === 401 || FINAL_CODES.has(code)) break;
      continue;
    }
    let text: string;
    try {
      text = await readUpstream(upstream);
    } catch {
      failures.push({ status: 502, code: 'upstream_too_large', model: next.id });
      noteFailure(next.id);
      continue;
    }
    const answerText = stream ? parseChatStream(text).text : chatAnswerText(parseJsonOrEmpty(text));
    if (fallback === null) fallback = { model: next.id, text };
    if (token !== null && !answerText.includes(token)) {
      log(`dial ${next.id} answered without the token; trying the next model`);
      noteFailure(next.id);
      continue;
    }
    return { answer: { model: next.id, text }, failures };
  }
  return { answer: fallback, failures };
}

/** One upstream response body as text, or a throw when it passes the buffer cap. */
async function readUpstream(res: Response): Promise<string> {
  if (res.body === null) return '';
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > MAX_UPSTREAM_BYTES) throw new Error('upstream answer over the buffer cap');
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(chunks).toString('utf8');
}

function parseJsonOrEmpty(text: string): Json {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

/** The usage estimate for a prompt: from the request's length and the answer's text. */
function fallbackUsage(requestChars: number, answerText: string): FallbackUsage {
  return { input: estimateTokens(requestChars), output: estimateTokens(answerText.length) };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by a free model through the gateway. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (candidatesFor(requires).length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const stream = body.stream === true;
  const token = findToken(requestText(body));
  const requestChars = JSON.stringify(body).length;
  const outcome = await serveOne(
    (model) => ({
      ...body,
      model,
      ...(typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number' ? { max_tokens: DEFAULT_MAX_TOKENS } : {}),
      ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
    }),
    requires,
    stream,
    token,
  );
  if (outcome.answer === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it', outcome.failures);
  }
  const { model, text } = outcome.answer;
  if (stream) {
    const parsed = parseChatStream(text);
    return sendStream(res, renderOpenAIStream(text, model, fallbackUsage(requestChars, parsed.text)), outcome.failures);
  }
  const answer = parseJsonOrEmpty(text);
  sendJson(res, 200, normalizeChatJson(answer, model, fallbackUsage(requestChars, chatAnswerText(answer))), outcome.failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (candidatesFor(requires).length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const stream = body.stream === true;
  const token = findToken(requestText(body));
  const requestChars = JSON.stringify(body).length;
  const outcome = await serveOne(
    (model) => {
      const translated = anthropicToChat(body, model);
      if (typeof translated.max_tokens !== 'number') translated.max_tokens = DEFAULT_MAX_TOKENS;
      return translated;
    },
    requires,
    stream,
    token,
  );
  if (outcome.answer === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it', outcome.failures);
  }
  const { model, text } = outcome.answer;
  if (stream) {
    const parsed = parseChatStream(text);
    return sendStream(res, renderAnthropicStream(text, model, fallbackUsage(requestChars, parsed.text)), outcome.failures);
  }
  const answer = parseJsonOrEmpty(text);
  sendJson(res, 200, chatJsonToAnthropic(answer, model, fallbackUsage(requestChars, chatAnswerText(answer))), outcome.failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/**
 * POST /register with the port and the one catalog; the env answers with the
 * trial's prompt count. Retried while the env is not yet answering: the catalog
 * is the same on every attempt, so a retry changes nothing the trial records.
 */
async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  throw new Error(`could not register after ${REGISTER_RETRIES} attempts: ${(lastError as Error)?.message ?? String(lastError)}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts. A process whose stdout is
    // not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
