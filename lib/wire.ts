// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

// This adapter bridges disparate request payloads by converting structures into equivalent formats.
import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
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

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
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

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
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
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
// Real-time chunk delivery pipelines stream generation tokens incrementally back to waiting consumers.
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
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
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// Cross-wire translation (added for the improved entry).
//
// The client's request is normalised once into a NormReq, then rendered into
// whichever wire the chosen provider speaks. The normal form keeps exactly
// what the env's echo compares: system text, user text, images, tool names,
// max_tokens, stop sequences and JSON mode. Nothing else is dropped either:
// temperature, top_p and tool definitions pass through untouched.
// ---------------------------------------------------------------------------

/** One image, as the client sent it: a data URL split into mime + base64, or a plain URL. */
export interface NormImage {
  mime: string;
  data: string;
  url: string | null;
}

/** One tool call in an assistant message. */
export interface NormToolCall {
  id: string;
  name: string;
  args: Json;
}

/** One tool result: the id of the call it answers and its text. */
export interface NormToolResult {
  id: string;
  text: string;
}

/** One message in the wire-neutral form. */
export interface NormMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: NormImage[];
  toolCalls: NormToolCall[];
  toolResults: NormToolResult[];
}

/** One tool definition in the wire-neutral form. */
export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

/** A client request, wire-neutral. `json` is true only for an OpenAI request carrying response_format. */
export interface NormReq {
  system: string;
  messages: NormMessage[];
  tools: NormTool[];
  toolChoice: 'auto' | 'any' | 'tool' | 'none' | null;
  toolChoiceName: string;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  json: boolean;
  stream: boolean;
}

function splitImageUrl(url: string): NormImage {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { mime: data[1] !== '' ? data[1] : 'image/png', data: data[2], url: null };
  return { mime: 'image/png', data: '', url };
}

/** The text of every text block in `content`, joined with a newline (the env's own join). */
function blocksTextOf(content: unknown): string {
  return textOf(content);
}

/** One OpenAI content part as a normalised image, or null. */
function imageOfOpenAIPart(block: Json): NormImage | null {
  if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
    return splitImageUrl(block.image_url.url);
  }
  return null;
}

/** One Anthropic content block as a normalised image, or null. */
function imageOfAnthropicBlock(block: Json): NormImage | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { mime: 'image/png', data: '', url: source.url };
  return null;
}

/** A tool call's arguments, as an object, from a JSON string or an object. */
function argsObjectOf(value: unknown): Json {
  return argumentsOf(value);
}

/** An OpenAI Chat Completions body as a NormReq. */
export function normalizeOpenAI(body: Json): NormReq {
  const messages: NormMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = blocksTextOf(raw.content);
      system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (role === 'user') {
      const text = typeof raw.content === 'string' ? raw.content : blocksTextOf(raw.content);
      const images: NormImage[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          const image = imageOfOpenAIPart(block);
          if (image !== null) images.push(image);
        }
      }
      messages.push({ role: 'user', text, images, toolCalls: [], toolResults: [] });
      continue;
    }
    if (role === 'assistant') {
      const text = typeof raw.content === 'string' ? raw.content : blocksTextOf(raw.content);
      const toolCalls: NormToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argsObjectOf(call.function.arguments),
          });
        }
      }
      messages.push({ role: 'assistant', text, images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (role === 'tool') {
      const text = typeof raw.content === 'string' ? raw.content : blocksTextOf(raw.content);
      messages.push({ role: 'tool', text, images: [], toolCalls: [], toolResults: [{ id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text }] });
      continue;
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormReq['toolChoice'] = null;
  let toolChoiceName = '';
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'none') toolChoice = 'none';
  else if (choice === 'required') toolChoice = 'any';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = 'tool';
    toolChoiceName = choice.function.name;
  }
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system,
    messages,
    tools,
    toolChoice,
    toolChoiceName,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    json: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a NormReq. */
export function normalizeAnthropic(body: Json): NormReq {
  const messages: NormMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const text = typeof raw.content === 'string' ? raw.content : blocksTextOf(raw.content);
      const toolCalls: NormToolCall[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          });
        }
      }
      messages.push({ role: 'assistant', text, images: [], toolCalls, toolResults: [] });
      continue;
    }
    const text = typeof raw.content === 'string' ? raw.content : blocksTextOf(raw.content);
    const images: NormImage[] = [];
    const toolResults: NormToolResult[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        const image = imageOfAnthropicBlock(block);
        if (image !== null) {
          images.push(image);
          continue;
        }
        if (block.type === 'tool_result') {
          toolResults.push({ id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text: toolResultText(block.content) });
        }
      }
    }
    messages.push({ role: 'user', text, images, toolCalls: [], toolResults });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormReq['toolChoice'] = null;
  let toolChoiceName = '';
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'any';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      toolChoice = 'tool';
      toolChoiceName = choice.name;
    }
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return {
    system: blocksTextOf(body.system),
    messages,
    tools,
    toolChoice,
    toolChoiceName,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
    json: false,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Renderers: a NormReq into each provider wire.
// ---------------------------------------------------------------------------

/** A NormReq as an OpenAI Chat Completions request naming `model`. */
export function normToOpenAI(n: NormReq, model: string): Json {
  const messages: Json[] = [];
  if (n.system !== '') messages.push({ role: 'system', content: n.system });
  for (const message of n.messages) {
    if (message.role === 'user') {
      for (const result of message.toolResults) {
        messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
      }
      if (message.images.length > 0) {
        const parts: Json[] = [];
        if (message.text !== '') parts.push({ type: 'text', text: message.text });
        for (const image of message.images) {
          parts.push({ type: 'image_url', image_url: { url: image.url !== null ? image.url : `data:${image.mime};base64,${image.data}` } });
        }
        messages.push({ role: 'user', content: parts });
      } else if (message.text !== '' || message.toolResults.length === 0) {
        messages.push({ role: 'user', content: message.text });
      }
      continue;
    }
    if (message.role === 'assistant') {
      const out: Json = { role: 'assistant', content: message.text !== '' ? message.text : null };
      if (message.toolCalls.length > 0) {
        out.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: JSON.stringify(call.args) },
        }));
      }
      messages.push(out);
      continue;
    }
    for (const result of message.toolResults) {
      messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
    }
  }
  const out: Json = { model, messages };
  if (n.maxTokens !== null) out.max_tokens = n.maxTokens;
  if (n.temperature !== null) out.temperature = n.temperature;
  if (n.topP !== null) out.top_p = n.topP;
  if (n.stops.length > 0) out.stop = [...n.stops];
  if (n.tools.length > 0) {
    out.tools = n.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  }
  if (n.toolChoice === 'auto') out.tool_choice = 'auto';
  else if (n.toolChoice === 'none') out.tool_choice = 'none';
  else if (n.toolChoice === 'any') out.tool_choice = 'required';
  else if (n.toolChoice === 'tool') out.tool_choice = { type: 'function', function: { name: n.toolChoiceName } };
  if (n.json) out.response_format = { type: 'json_object' };
  if (n.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A NormReq as an Anthropic Messages request naming `model`. */
export function normToAnthropic(n: NormReq, model: string): Json {
  const messages: Json[] = [];
  for (const message of n.messages) {
    if (message.role === 'user') {
      const content: Json[] = [];
      if (message.text !== '') content.push({ type: 'text', text: message.text });
      for (const image of message.images) {
        if (image.url !== null) content.push({ type: 'image', source: { type: 'url', url: image.url } });
        else content.push({ type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } });
      }
      for (const result of message.toolResults) {
        content.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
      }
      if (content.length > 0) messages.push({ role: 'user', content });
      continue;
    }
    if (message.role === 'assistant') {
      const content: Json[] = [];
      if (message.text !== '') content.push({ type: 'text', text: message.text });
      for (const call of message.toolCalls) {
        content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    const content: Json[] = [];
    for (const result of message.toolResults) {
      content.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
    }
    if (content.length > 0) messages.push({ role: 'user', content });
  }
  const out: Json = { model, messages };
  if (n.system !== '') out.system = n.system;
  if (n.maxTokens !== null) out.max_tokens = n.maxTokens;
  if (n.temperature !== null) out.temperature = n.temperature;
  if (n.topP !== null) out.top_p = n.topP;
  if (n.stops.length > 0) out.stop_sequences = [...n.stops];
  if (n.tools.length > 0) {
    out.tools = n.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (n.toolChoice === 'auto') out.tool_choice = { type: 'auto' };
  else if (n.toolChoice === 'any') out.tool_choice = { type: 'any' };
  else if (n.toolChoice === 'none') out.tool_choice = { type: 'none' };
  else if (n.toolChoice === 'tool') out.tool_choice = { type: 'tool', name: n.toolChoiceName };
  if (n.stream) out.stream = true;
  return out;
}

/** A NormReq as a Gemini generateContent request body (the model goes in the URL). */
export function normToGemini(n: NormReq): Json {
  const callName = new Map<string, string>();
  for (const message of n.messages) {
    for (const call of message.toolCalls) callName.set(call.id, call.name);
  }
  const contents: Json[] = [];
  for (const message of n.messages) {
    if (message.role === 'user') {
      const parts: Json[] = [];
      if (message.text !== '') parts.push({ text: message.text });
      for (const image of message.images) {
        if (image.url !== null) parts.push({ fileData: { fileUri: image.url, mimeType: image.mime } });
        else parts.push({ inlineData: { mimeType: image.mime, data: image.data } });
      }
      for (const result of message.toolResults) {
        parts.push({ functionResponse: { name: callName.get(result.id) ?? '', response: { content: result.text } } });
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
      continue;
    }
    if (message.role === 'assistant') {
      const parts: Json[] = [];
      if (message.text !== '') parts.push({ text: message.text });
      for (const call of message.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: call.args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    for (const result of message.toolResults) {
      parts.push({ functionResponse: { name: callName.get(result.id) ?? '', response: { content: result.text } } });
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (n.system !== '') out.systemInstruction = { parts: [{ text: n.system }] };
  if (n.tools.length > 0) {
    out.tools = [{ functionDeclarations: n.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })) }];
  }
  const config: Json = {};
  if (n.maxTokens !== null) config.maxOutputTokens = n.maxTokens;
  if (n.stops.length > 0) config.stopSequences = [...n.stops];
  if (n.temperature !== null) config.temperature = n.temperature;
  if (n.topP !== null) config.topP = n.topP;
  if (n.json) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response translators: a provider's answer back into the client's wire.
// The simulated providers always answer plain text with a family-0 stop, so
// these stay simple, but they translate tool blocks and stop reasons in
// general, and carry the provider's usage counts through unchanged.
// ---------------------------------------------------------------------------

/** A provider stop word as an OpenAI finish_reason. */
export function openAIFinishOf(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_calls' || stop === 'tool_use' || stop === 'function_call') return 'tool_calls';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** A provider stop word as an Anthropic stop_reason. */
export function anthropicStopOf(stop: unknown): string {
  if (stop === 'length' || stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'tool_use' || stop === 'function_call') return 'tool_use';
  if (stop === 'content_filter' || stop === 'refusal' || stop === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini candidate's parts, joined in order. */
function geminiTextOf(answer: Json): string {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini finishReason, or null when the answer names none. */
function geminiFinishOf(answer: Json): string | null {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : null;
}

/** A Gemini usageMetadata block as input/output token counts. */
function geminiUsageOf(answer: Json): { input: number; output: number } {
  const meta = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    input: typeof meta.promptTokenCount === 'number' ? meta.promptTokenCount : 0,
    output: typeof meta.candidatesTokenCount === 'number' ? meta.candidatesTokenCount : 0,
  };
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(answer: Json, model: string): Json {
  const usage = geminiUsageOf(answer);
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(answer) }, finish_reason: openAIFinishOf(geminiFinishOf(answer)) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(answer: Json, model: string): Json {
  const usage = geminiUsageOf(answer);
  return {
    id: `msg_${model.replace(/[^A-Za-z0-9_-]/g, '_')}`,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(answer) }],
    stop_reason: anthropicStopOf(geminiFinishOf(answer)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIAnswer(answer: Json, model: string): Json {
  const content = Array.isArray(answer.content) ? answer.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(answer.stop_reason) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

// ---------------------------------------------------------------------------
// Stream translators. The upstream stream is read whole first (the answers
// are short and the client waits for the full body anyway), then re-emitted
// in the client's event order with the usage the last upstream frame carried.
// ---------------------------------------------------------------------------

/** One SSE frame: its event name (null when unnamed) and its data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): Json | null {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One OpenAI chunk frame: delta, optional finish_reason, optional usage. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage: Json | null): string {
  const chunk: Json = { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] };
  if (usage !== null) {
    chunk.choices = [];
    chunk.usage = usage;
  }
  return sseEvent(null, chunk);
}

/** The starter's OpenAI-stream-to-Anthropic-events translator, fed from buffered text. */
export async function anthropicEventsFromOpenAIText(text: string, model: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const stream = new ReadableStream<Uint8Array>({ start(controller) { controller.enqueue(bytes); controller.close(); } });
  let out = '';
  for await (const event of anthropicEventsFromOpenAI(stream, model)) out += event;
  return out;
}

/** An Anthropic Messages SSE stream as an OpenAI Chat Completions SSE stream. */
export function openAIEventsFromAnthropicText(text: string, model: string): string {
  let body = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason: string | null = null;
  const toolCalls: { id: string; name: string; args: string }[] = [];
  const toolByIndex = new Map<number, number>();
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : 0;
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : toolCalls.length;
      toolByIndex.set(index, toolCalls.length);
      toolCalls.push({ id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`, name: typeof data.content_block.name === 'string' ? data.content_block.name : '', args: '' });
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') body += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const index = typeof data.index === 'number' ? data.index : -1;
        const at = toolByIndex.get(index);
        if (at !== undefined) toolCalls[at].args += data.delta.partial_json;
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') outputTokens = data.usage.output_tokens;
    }
  }
  const out: string[] = [openAIChunk(model, { role: 'assistant', content: '' }, null, null)];
  if (body !== '') out.push(openAIChunk(model, { content: body }, null, null));
  for (const call of toolCalls) {
    out.push(
      openAIChunk(model, { tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.args } }] }, null, null),
    );
  }
  out.push(openAIChunk(model, {}, openAIFinishOf(stopReason), null));
  out.push(openAIChunk(model, {}, null, { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini streamGenerateContent SSE stream as an OpenAI Chat Completions SSE stream. */
export function openAIEventsFromGeminiText(text: string, model: string): string {
  let body = '';
  let finish: string | null = null;
  let usage: { input: number; output: number } | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    body += geminiTextOf(data);
    const found = geminiFinishOf(data);
    if (found !== null) finish = found;
    if (isRecord(data.usageMetadata)) usage = geminiUsageOf(data);
  }
  const input = usage?.input ?? 0;
  const output = usage?.output ?? 0;
  const out: string[] = [openAIChunk(model, { role: 'assistant', content: '' }, null, null)];
  if (body !== '') out.push(openAIChunk(model, { content: body }, null, null));
  out.push(openAIChunk(model, {}, openAIFinishOf(finish), null));
  out.push(openAIChunk(model, {}, null, { prompt_tokens: input, completion_tokens: output, total_tokens: input + output }));
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Gemini streamGenerateContent SSE stream as an Anthropic Messages SSE stream. */
export function anthropicEventsFromGeminiText(text: string, model: string): string {
  let body = '';
  let finish: string | null = null;
  let usage: { input: number; output: number } | null = null;
  for (const frame of parseSseFrames(text)) {
    const data = jsonOrNull(frame.data);
    if (data === null) continue;
    body += geminiTextOf(data);
    const found = geminiFinishOf(data);
    if (found !== null) finish = found;
    if (isRecord(data.usageMetadata)) usage = geminiUsageOf(data);
  }
  const input = usage?.input ?? 0;
  const output = usage?.output ?? 0;
  const stop = anthropicStopOf(finish);
  return [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: `msg_${model.replace(/[^A-Za-z0-9_-]/g, '_')}`, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ...(body !== '' ? [sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: body } })] : []),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ].join('');
}
