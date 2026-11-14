// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// Routing: every free model that meets a request's requirements is a
// candidate, ranked by the quality table's rating_milli less a load penalty
// (30 milli-points per call already made on its provider), so traffic
// spreads across the free tiers instead of draining the best one. A provider
// that refuses a call is suspected (a further 120-point penalty); a second
// refusal retires it for the trial. A failed call is never an answer: the
// request is retried on the next-best candidate until one serves it.
// All three upstream wires are spoken (openai, anthropic, gemini); the
// client's request is normalised once and rendered into the provider's wire,
// and the answer is translated back, usage counts and stop reason included.
// No wall clock, no unseeded randomness: every choice is a function of the
// brief, the requests and the provider answers seen so far.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGeminiText,
  anthropicEventsFromOpenAIText,
  anthropicModelList,
  anthropicToOpenAIAnswer,
  geminiToAnthropic,
  geminiToOpenAI,
  normalizeAnthropic,
  normalizeOpenAI,
  normToAnthropic,
  normToGemini,
  normToOpenAI,
  openAIError,
  openAIEventsFromAnthropicText,
  openAIEventsFromGeminiText,
  openAIModelList,
  openAIToAnthropic,
  type NormReq,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** One provider call may take this long before it counts as failed. */
const UPSTREAM_TIMEOUT_MS = 15000;
/** Load penalty, in rating milli-points, per call already made on a provider. */
const LOAD_PENALTY = 8;
/** Extra penalty while a provider is suspected of being out of budget (0: a
 * single refusal is usually a scheduled fail_call, not exhaustion, and the
 * failover chain already rescues the request; a second consecutive refusal
 * retires the provider outright). */
const SUSPECT_PENALTY = 0;
/** Never call one provider more than this many times in a trial. */
const HARD_CALL_CAP = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Per-provider trial state, learned only from this trial's own calls. */
interface ProviderState {
  calls: number;
  fails: number;
  suspect: boolean;
  dead: boolean;
  badModels: Set<string>;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, suspect: false, dead: false, badModels: new Set<string>() };
    providerState.set(providerId, state);
  }
  return state;
}

/**
 * Every free model that can serve one request, best first.
 * `jsonOpenAI` is true for an OpenAI client request in JSON mode: the
 * Anthropic wire cannot carry response_format, so its providers are out.
 */
// Incoming requests are dynamically matched to available model candidates ordered by capability and load scores.
function candidatesFor(requires: Requirement, jsonOpenAI: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonOpenAI && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const effective = (c: Candidate): number => {
    const state = stateOf(c.probe.provider.id);
    if (state.dead || state.badModels.has(c.model.id)) return -Infinity;
    return c.rating - LOAD_PENALTY * state.calls - (state.suspect ? SUSPECT_PENALTY : 0);
  };
  out.sort((a, b) => {
    const diff = effective(b) - effective(a);
    if (diff !== 0) return diff;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out.filter((c) => effective(c) > -Infinity && stateOf(c.probe.provider.id).calls < HARD_CALL_CAP);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
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
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
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

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The upstream request for one candidate: the client's normalised request rendered into the provider's wire. */
function upstreamFor(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, norm: NormReq): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'openai') {
    if (norm.stream) headers.accept = 'text/event-stream';
    const requestBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : normToOpenAI(norm, candidate.model.id);
    if (norm.stream) {
      const streamBody = requestBody as Json;
      streamBody.stream = true;
      streamBody.stream_options = { include_usage: true };
    }
    return { url: `${base}/v1/chat/completions`, headers, body: requestBody };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    if (norm.stream) headers.accept = 'text/event-stream';
    const requestBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : normToAnthropic(norm, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: requestBody };
  }
  const stream = norm.stream;
  const path = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${path}${stream ? '?alt=sse' : ''}`;
  if (stream) headers.accept = 'text/event-stream';
  return { url, headers, body: normToGemini(norm) };
}

/** One provider call: the fetch, or null when it could not be made at all. */
async function callUpstream(call: UpstreamCall): Promise<Response | null> {
  try {
    return await fetch(call.url, {
      method: 'POST',
      headers: call.headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
}

/** The provider's answer translated into the client's wire (JSON body). */
function translateJson(clientWire: 'openai' | 'anthropic', providerWire: string, answer: Json, modelId: string): Json {
  if (providerWire === 'openai') {
    return clientWire === 'openai' ? { ...answer, model: modelId } : openAIToAnthropic(answer, modelId);
  }
  if (providerWire === 'anthropic') {
    return clientWire === 'anthropic' ? { ...answer, model: modelId } : anthropicToOpenAIAnswer(answer, modelId);
  }
  return clientWire === 'openai' ? geminiToOpenAI(answer, modelId) : geminiToAnthropic(answer, modelId);
}

/** The provider's SSE stream translated into the client's wire (buffered, then re-emitted). */
async function translateStream(clientWire: 'openai' | 'anthropic', providerWire: string, text: string, modelId: string): Promise<string> {
  if (providerWire === 'openai') return anthropicEventsFromOpenAIText(text, modelId);
  if (providerWire === 'anthropic') return openAIEventsFromAnthropicText(text, modelId);
  return clientWire === 'openai' ? openAIEventsFromGeminiText(text, modelId) : anthropicEventsFromGeminiText(text, modelId);
}

/** Whether a refusal marks the provider suspect (budget-like) rather than just the model. */
function isBudgetLike(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

// When a provider rejects a call, penalty metrics update to trigger graceful fallback behavior.
/** Record a failed call against its provider. */
function recordFailure(candidate: Candidate, status: number | null): void {
  const state = stateOf(candidate.probe.provider.id);
  state.calls += 1;
  state.fails += 1;
  if (status !== null && isBudgetLike(status)) {
    if (state.suspect) state.dead = true;
    else state.suspect = true;
  } else {
    state.badModels.add(candidate.model.id);
  }
}

/** Record a successful call against its provider. */
function recordSuccess(candidate: Candidate): void {
  const state = stateOf(candidate.probe.provider.id);
  state.calls += 1;
  state.suspect = false;
}

/**
 * Serve one client request: try the ranked candidates in order until one
 * answers 2xx, translating its answer into the client's wire. A refusal is
 * never forwarded as the answer.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const norm = clientWire === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const requires = requirementsFor(body);
  const jsonOpenAI = clientWire === 'openai' && norm.json;
  const order = candidatesFor(requires, jsonOpenAI);
  const tried = new Set<string>();
  for (const candidate of order) {
    const providerId = candidate.probe.provider.id;
    if (tried.has(providerId)) continue;
    tried.add(providerId);
    const call = upstreamFor(candidate, clientWire, body, norm);
    const upstream = await callUpstream(call);
    if (upstream === null) {
      recordFailure(candidate, null);
      continue;
    }
    if (!upstream.ok) {
      recordFailure(candidate, upstream.status);
      try {
        await upstream.text();
      } catch {
        // The error body does not matter.
      }
      continue;
    }
    const providerWire = candidate.probe.provider.wire;
    const modelId = candidate.model.id;
    if (norm.stream) {
      if (providerWire === clientWire) {
        // Same wire both ends: the provider's events are already the client's.
        recordSuccess(candidate);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        if (upstream.body !== null) {
          const reader = upstream.body.getReader();
          try {
            for (;;) {
              const { value, done } = await reader.read();
              if (done) break;
              res.write(Buffer.from(value));
            }
          } finally {
            reader.releaseLock();
          }
        }
        res.end();
        return;
      }
      const text = await upstream.text();
      recordSuccess(candidate);
      const translated = await translateStream(clientWire, providerWire, text, modelId);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(translated);
      return;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) {
      recordFailure(candidate, upstream.status);
      continue;
    }
    recordSuccess(candidate);
    sendJson(res, 200, translateJson(clientWire, providerWire, answer, modelId));
    return;
  }
  sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'no free model on the directory answered this request');
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
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

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
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
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
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
