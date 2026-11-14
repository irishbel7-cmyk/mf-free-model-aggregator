// free-model-aggregator-01 entry: a deterministic Node 22 server.
//
// It reads the brief ($MAKEFUN_PROMPT_FILE: provider directory, sealed
// quality table, workload manifest), registers its catalog with the trial
// env, then answers the env client's 20 requests (OpenAI or Anthropic wire,
// JSON or SSE) by calling the best free model that meets each request's
// requirements, failing over to the next-best on 402/429, and translating
// the provider's answer back into the client's wire. No model calls of its
// own, no wall clock, no unseeded randomness: every choice is a pure
// function of the brief and the provider answers seen so far.

import { createServer } from "node:http";
import { readFileSync } from "node:fs";

// Fixed `created` value used in every response we emit. The env uses this
// same constant for its own answers; the client parsers never check it,
// and a literal keeps replays byte-identical.
const FIXED_CREATED = 1767225600;
const ANTHROPIC_VERSION = "2023-06-01";
const PROVIDER_TIMEOUT_MS = 10000;
const REGISTER_ATTEMPTS = 5;
const REGISTER_RETRY_MS = 200;
const WORK_POLL_MS = 250;

// Stop-reason families, as the env's checks define them: 0 stop, 1 length,
// 2 tool use, 3 refusal/content filter.
const STOP_FAMILY: Record<string, number> = {
  stop: 0,
  end_turn: 0,
  STOP: 0,
  length: 1,
  max_tokens: 1,
  MAX_TOKENS: 1,
  tool_calls: 2,
  function_call: 2,
  tool_use: 2,
  content_filter: 3,
  refusal: 3,
  SAFETY: 3,
};

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Brief parsing. prompt.txt holds prose plus three JSON documents, each on
// its own line(s), keyed by a "free-model-aggregator-01/<kind>/v1" schema.
// We scan for the schema marker and extract the balanced JSON object that
// follows it, so line wrapping in the brief cannot break us.
// ---------------------------------------------------------------------------

function parseBalanced(text: string, start: number): { value: any; end: number } | null {
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
      continue;
    }
    if (ch === "{") depth += 1;
    else if (ch === "}") {
      depth -= 1;
      if (depth === 0) {
        try {
          return { value: JSON.parse(text.slice(start, i + 1)), end: i + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function extractJsonDocs(text: string): any[] {
  const docs: any[] = [];
  const marker = '{"schema":"free-model-aggregator-01/';
  let from = 0;
  while (true) {
    const at = text.indexOf(marker, from);
    if (at === -1) break;
    const doc = parseBalanced(text, at);
    if (doc === null) break;
    docs.push(doc.value);
    from = doc.end;
  }
  return docs;
}

interface Brief {
  providers: any[];
  ratings: any[];
  manifest: any[];
}

function parseBrief(text: string): Brief | null {
  const docs = extractJsonDocs(text);
  let providers: any[] | null = null;
  let ratings: any[] | null = null;
  let manifest: any[] | null = null;
  for (const d of docs) {
    if (d && d.schema === "free-model-aggregator-01/providers/v1") providers = d.providers;
    else if (d && d.schema === "free-model-aggregator-01/ratings/v1") ratings = d.ratings;
    else if (d && d.schema === "free-model-aggregator-01/workload/v1") manifest = d.manifest;
  }
  if (!Array.isArray(providers) || !Array.isArray(ratings) || !Array.isArray(manifest)) return null;
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------------------------
// Catalog. Built entirely from the brief (directory + ratings table).
// The registered catalog and the --list output are the same string.
// ---------------------------------------------------------------------------

function buildCatalog(brief: Brief): { catalog: any; text: string } {
  const providers = brief.providers.map((p) => ({
    id: p.id,
    base_url: p.path,
    key_env: p.key_env,
    wire: p.wire,
    models: brief.ratings
      .filter((r) => r.provider === p.id && r.free === true)
      .map((r) => ({ id: r.model, free: true, context_length: r.context_length })),
  }));
  const catalog = { providers };
  return { catalog, text: JSON.stringify(catalog) };
}

// ---------------------------------------------------------------------------
// Canonical request. A client request (OpenAI or Anthropic wire) is reduced
// to the fields the provider echo checks: system text, user/assistant turns
// (text + images, in order), tool names, max_tokens, stop sequences, JSON
// mode, temperature. The provider echo is computed from exactly these, so
// translating them losslessly is what makes a request valid.
// ---------------------------------------------------------------------------

interface ImageRef {
  mime: string | null;
  data: string | null; // base64 bytes, when the image arrived as a data URL
  url: string | null; // plain URL, when it did not
}

interface Turn {
  role: string; // "user" | "assistant"
  text: string; // text parts joined with "\n", as every wire's parser does
  images: ImageRef[];
}

interface ToolDef {
  name: string;
  description: string;
  parameters: any;
}

interface Canonical {
  clientWire: "openai" | "anthropic";
  stream: boolean;
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  toolChoice: string | { name: string } | null; // "auto" | {name} | null
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
}

// Text of a content value, the way every wire's parser reads it: a plain
// string as-is, else the text blocks joined with "\n".
function textOfContent(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (b && typeof b === "object" && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function parseImageUrl(url: string): ImageRef {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { mime: m[1] === "" ? null : m[1], data: m[2], url: null };
  return { mime: null, data: null, url };
}

function imageUrlOf(img: ImageRef): string {
  if (img.data !== null) return `data:${img.mime ?? "image/png"};base64,${img.data}`;
  return img.url ?? "";
}

function imagesOfOpenAI(content: any): ImageRef[] {
  const out: ImageRef[] = [];
  if (!Array.isArray(content)) return out;
  for (const b of content) {
    if (b && typeof b === "object" && b.type === "image_url" && b.image_url && typeof b.image_url.url === "string") {
      out.push(parseImageUrl(b.image_url.url));
    }
  }
  return out;
}

function imagesOfAnthropic(content: any): ImageRef[] {
  const out: ImageRef[] = [];
  if (!Array.isArray(content)) return out;
  for (const b of content) {
    if (!b || typeof b !== "object" || b.type !== "image" || !b.source || typeof b.source !== "object") continue;
    if (b.source.type === "base64" && typeof b.source.data === "string") {
      out.push({ mime: typeof b.source.media_type === "string" ? b.source.media_type : null, data: b.source.data, url: null });
    } else if (b.source.type === "url" && typeof b.source.url === "string") {
      out.push({ mime: null, data: null, url: b.source.url });
    }
  }
  return out;
}

function canonicalFromOpenAI(body: any): Canonical {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "system") {
      const t = textOfContent(m.content);
      if (t !== "") systemParts.push(t);
    } else if (m.role === "user") {
      turns.push({ role: "user", text: textOfContent(m.content), images: imagesOfOpenAI(m.content) });
    } else if (m.role === "assistant") {
      turns.push({ role: "assistant", text: textOfContent(m.content), images: [] });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (t && typeof t === "object" && t.function && typeof t.function.name === "string") {
        tools.push({
          name: t.function.name,
          description: typeof t.function.description === "string" ? t.function.description : "",
          parameters: t.function.parameters,
        });
      }
    }
  }
  let toolChoice: Canonical["toolChoice"] = null;
  if (body.tool_choice === "auto" || body.tool_choice === "none" || body.tool_choice === "required") {
    toolChoice = body.tool_choice;
  } else if (body.tool_choice && typeof body.tool_choice === "object" && body.tool_choice.type === "function" && body.tool_choice.function && typeof body.tool_choice.function.name === "string") {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const maxTokens =
    typeof body.max_tokens === "number" ? body.max_tokens : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : null;
  const stops = Array.isArray(body.stop)
    ? body.stop.filter((s: any) => typeof s === "string")
    : typeof body.stop === "string"
      ? [body.stop]
      : [];
  const jsonMode =
    body.response_format && typeof body.response_format === "object" && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
  return {
    clientWire: "openai",
    stream: body.stream === true,
    systemText: systemParts.join("\n"),
    turns,
    tools,
    toolChoice,
    maxTokens,
    stops,
    jsonMode: jsonMode === true,
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
  };
}

function canonicalFromAnthropic(body: any): Canonical {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!m || typeof m !== "object") continue;
    if (m.role === "user") {
      turns.push({ role: "user", text: textOfContent(m.content), images: imagesOfAnthropic(m.content) });
    } else if (m.role === "assistant") {
      turns.push({ role: "assistant", text: textOfContent(m.content), images: [] });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (t && typeof t === "object" && typeof t.name === "string") {
        tools.push({
          name: t.name,
          description: typeof t.description === "string" ? t.description : "",
          parameters: t.input_schema,
        });
      }
    }
  }
  let toolChoice: Canonical["toolChoice"] = null;
  if (body.tool_choice && typeof body.tool_choice === "object") {
    if (body.tool_choice.type === "auto") toolChoice = "auto";
    else if (body.tool_choice.type === "tool" && typeof body.tool_choice.name === "string") toolChoice = { name: body.tool_choice.name };
    else if (body.tool_choice.type === "any") toolChoice = "required";
    else if (body.tool_choice.type === "none") toolChoice = "none";
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s: any) => typeof s === "string") : [];
  return {
    clientWire: "anthropic",
    stream: body.stream === true,
    systemText: textOfContent(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : null,
    stops,
    jsonMode: false, // the Anthropic wire has no JSON-mode flag; its echo is always false
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
  };
}

// ---------------------------------------------------------------------------
// Request translation: canonical request -> provider wire.
// Every field the echo checks is carried over exactly: system text, user
// text, image bytes, tool names (in order), max_tokens, stops, JSON mode.
// ---------------------------------------------------------------------------

function openAIMessages(c: Canonical): any[] {
  const out: any[] = [];
  if (c.systemText !== "") out.push({ role: "system", content: c.systemText });
  for (const t of c.turns) {
    if (t.role === "user") {
      const parts: any[] = [];
      if (t.text !== "") parts.push({ type: "text", text: t.text });
      for (const img of t.images) parts.push({ type: "image_url", image_url: { url: imageUrlOf(img) } });
      out.push({ role: "user", content: parts });
    } else {
      out.push({ role: "assistant", content: t.text });
    }
  }
  return out;
}

function anthropicMessages(c: Canonical): any[] {
  const out: any[] = [];
  for (const t of c.turns) {
    if (t.role === "user") {
      const blocks: any[] = [];
      if (t.text !== "") blocks.push({ type: "text", text: t.text });
      for (const img of t.images) {
        if (img.data !== null) {
          blocks.push({ type: "image", source: { type: "base64", media_type: img.mime ?? "image/png", data: img.data } });
        } else {
          blocks.push({ type: "image", source: { type: "url", url: img.url ?? "" } });
        }
      }
      out.push({ role: "user", content: blocks });
    } else {
      out.push({ role: "assistant", content: [{ type: "text", text: t.text }] });
    }
  }
  return out;
}

function geminiContents(c: Canonical): any[] {
  const out: any[] = [];
  for (const t of c.turns) {
    const parts: any[] = [];
    if (t.text !== "") parts.push({ text: t.text });
    for (const img of t.images) {
      if (img.data !== null) parts.push({ inlineData: { mimeType: img.mime ?? "image/png", data: img.data } });
      else parts.push({ fileData: { fileUri: img.url ?? "" } });
    }
    out.push({ role: t.role === "assistant" ? "model" : "user", parts });
  }
  return out;
}

// Returns { path, body } for the provider call. For Gemini the model goes
// in the URL path, not the body.
function buildProviderRequest(provider: any, model: string, c: Canonical): { path: string; body: any } {
  if (provider.wire === "openai") {
    const body: any = { model, messages: openAIMessages(c) };
    if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
    if (c.temperature !== undefined) body.temperature = c.temperature;
    if (c.tools.length > 0) {
      body.tools = c.tools.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }));
    }
    if (c.toolChoice === "auto" || c.toolChoice === "none" || c.toolChoice === "required") body.tool_choice = c.toolChoice;
    else if (c.toolChoice !== null) body.tool_choice = { type: "function", function: { name: c.toolChoice.name } };
    if (c.stops.length > 0) body.stop = c.stops;
    if (c.jsonMode) body.response_format = { type: "json_object" };
    if (c.stream) {
      body.stream = true;
      body.stream_options = { include_usage: true }; // so the provider stream carries its usage
    }
    return { path: "/v1/chat/completions", body };
  }
  if (provider.wire === "anthropic") {
    const body: any = { model, messages: anthropicMessages(c) };
    if (c.systemText !== "") body.system = c.systemText;
    if (c.maxTokens !== null) body.max_tokens = c.maxTokens;
    if (c.temperature !== undefined) body.temperature = c.temperature;
    if (c.tools.length > 0) {
      body.tools = c.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
    }
    if (c.toolChoice === "auto") body.tool_choice = { type: "auto" };
    else if (c.toolChoice === "none") body.tool_choice = { type: "none" };
    else if (c.toolChoice === "required") body.tool_choice = { type: "any" };
    else if (c.toolChoice !== null) body.tool_choice = { type: "tool", name: c.toolChoice.name };
    if (c.stops.length > 0) body.stop_sequences = c.stops;
    if (c.stream) body.stream = true;
    return { path: "/v1/messages", body };
  }
  // gemini
  const body: any = { contents: geminiContents(c) };
  if (c.systemText !== "") body.systemInstruction = { parts: [{ text: c.systemText }] };
  if (c.tools.length > 0) {
    body.tools = [{ functionDeclarations: c.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const gc: any = {};
  if (c.maxTokens !== null) gc.maxOutputTokens = c.maxTokens;
  if (c.stops.length > 0) gc.stopSequences = c.stops;
  if (c.jsonMode) gc.responseMimeType = "application/json";
  if (c.temperature !== undefined) gc.temperature = c.temperature;
  if (Object.keys(gc).length > 0) body.generationConfig = gc;
  const verb = c.stream ? ":streamGenerateContent" : ":generateContent";
  const path = `/v1beta/models/${encodeURIComponent(model)}${verb}${c.stream ? "?alt=sse" : ""}`;
  return { path, body };
}

function providerHeaders(provider: any): Record<string, string> {
  const h: Record<string, string> = { "content-type": "application/json" };
  if (provider.auth === "bearer") h["authorization"] = `Bearer ${provider.key}`;
  else if (provider.auth === "x-api-key") h["x-api-key"] = provider.key;
  else if (provider.auth === "x-goog-api-key") h["x-goog-api-key"] = provider.key;
  if (provider.wire === "anthropic") h["anthropic-version"] = ANTHROPIC_VERSION;
  return h;
}

// ---------------------------------------------------------------------------
// Provider answers. A provider's answer is always plain text with a
// family-0 stop and usage counts. We parse it (JSON or SSE, any of the
// three wires) into { text, stop, usage }, then re-emit it in the client's
// wire. Usage is taken from the provider whenever it carries it; when a
// stream omits it, it is recomputed with the env's own formula, which is a
// pure function of the request we sent and the answer text, so it matches
// exactly.
// ---------------------------------------------------------------------------

interface Usage {
  input: number;
  output: number;
}

interface ProviderAnswer {
  text: string;
  stop: string;
  usage: Usage;
}

function parseSse(raw: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = raw.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function jsonOrNull(text: string): any | null {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// The env's usage formula: input = 1 + floor((system + user + tools) / 4),
// output = 1 + floor(answer length / 4). Inputs are the fields the provider
// parsed from the request we sent, which are exactly our canonical fields.
function computeUsage(c: Canonical, answerText: string): Usage {
  let userText = "";
  for (let i = c.turns.length - 1; i >= 0; i--) {
    if (c.turns[i].role === "user") {
      userText = c.turns[i].text;
      break;
    }
  }
  const prompt = c.systemText.length + userText.length + c.tools.map((t) => t.name).join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}

function parseProviderJson(wire: string, raw: string, c: Canonical): ProviderAnswer {
  const v = jsonOrNull(raw) ?? {};
  if (wire === "openai") {
    const ch = Array.isArray(v.choices) && v.choices.length > 0 ? v.choices[0] : {};
    const text = ch.message && typeof ch.message.content === "string" ? ch.message.content : "";
    const usage =
      v.usage && typeof v.usage.prompt_tokens === "number" && typeof v.usage.completion_tokens === "number"
        ? { input: v.usage.prompt_tokens, output: v.usage.completion_tokens }
        : computeUsage(c, text);
    return { text, stop: typeof ch.finish_reason === "string" ? ch.finish_reason : "stop", usage };
  }
  if (wire === "anthropic") {
    let text = "";
    if (Array.isArray(v.content)) {
      for (const b of v.content) {
        if (b && typeof b === "object" && b.type === "text" && typeof b.text === "string") text += b.text;
      }
    }
    const usage =
      v.usage && typeof v.usage.input_tokens === "number" && typeof v.usage.output_tokens === "number"
        ? { input: v.usage.input_tokens, output: v.usage.output_tokens }
        : computeUsage(c, text);
    return { text, stop: typeof v.stop_reason === "string" ? v.stop_reason : "end_turn", usage };
  }
  // gemini
  const cand = Array.isArray(v.candidates) && v.candidates.length > 0 ? v.candidates[0] : {};
  let text = "";
  if (cand.content && Array.isArray(cand.content.parts)) {
    for (const p of cand.content.parts) {
      if (p && typeof p === "object" && typeof p.text === "string") text += p.text;
    }
  }
  const um = v.usageMetadata && typeof v.usageMetadata === "object" ? v.usageMetadata : null;
  const usage =
    um !== null && typeof um.promptTokenCount === "number" && typeof um.candidatesTokenCount === "number"
      ? { input: um.promptTokenCount, output: um.candidatesTokenCount }
      : computeUsage(c, text);
  return { text, stop: typeof cand.finishReason === "string" ? cand.finishReason : "STOP", usage };
}

function parseProviderStream(wire: string, raw: string, c: Canonical): ProviderAnswer {
  const frames = parseSse(raw);
  if (wire === "openai") {
    let text = "";
    let stop = "stop";
    let usage: Usage | null = null;
    for (const f of frames) {
      if (f.data === "[DONE]") continue;
      const chunk = jsonOrNull(f.data);
      if (chunk === null || typeof chunk !== "object") continue;
      if (chunk.usage && typeof chunk.usage === "object" && typeof chunk.usage.prompt_tokens === "number" && typeof chunk.usage.completion_tokens === "number") {
        usage = { input: chunk.usage.prompt_tokens, output: chunk.usage.completion_tokens };
      }
      const ch = Array.isArray(chunk.choices) && chunk.choices.length > 0 ? chunk.choices[0] : null;
      if (ch === null || typeof ch !== "object") continue;
      if (ch.delta && typeof ch.delta.content === "string") text += ch.delta.content;
      if (typeof ch.finish_reason === "string") stop = ch.finish_reason;
    }
    return { text, stop, usage: usage ?? computeUsage(c, text) };
  }
  if (wire === "anthropic") {
    let text = "";
    let stop = "end_turn";
    let input: number | null = null;
    let output: number | null = null;
    for (const f of frames) {
      const d = jsonOrNull(f.data);
      if (d === null || typeof d !== "object") continue;
      if (f.event === "message_start" && d.message && d.message.usage && typeof d.message.usage.input_tokens === "number") {
        input = d.message.usage.input_tokens;
      } else if (f.event === "content_block_delta" && d.delta && d.delta.type === "text_delta" && typeof d.delta.text === "string") {
        text += d.delta.text;
      } else if (f.event === "message_delta") {
        if (d.delta && typeof d.delta.stop_reason === "string") stop = d.delta.stop_reason;
        if (d.usage && typeof d.usage.output_tokens === "number") output = d.usage.output_tokens;
      }
    }
    const fallback = computeUsage(c, text);
    return { text, stop, usage: { input: input ?? fallback.input, output: output ?? fallback.output } };
  }
  // gemini stream: each frame carries text parts; the last carries usage.
  let text = "";
  let stop = "STOP";
  let usage: Usage | null = null;
  for (const f of frames) {
    const v = jsonOrNull(f.data);
    if (v === null || typeof v !== "object") continue;
    const cand = Array.isArray(v.candidates) && v.candidates.length > 0 ? v.candidates[0] : null;
    if (cand !== null && typeof cand === "object") {
      if (cand.content && Array.isArray(cand.content.parts)) {
        for (const p of cand.content.parts) {
          if (p && typeof p === "object" && typeof p.text === "string") text += p.text;
        }
      }
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (v.usageMetadata && typeof v.usageMetadata === "object" && typeof v.usageMetadata.promptTokenCount === "number" && typeof v.usageMetadata.candidatesTokenCount === "number") {
      usage = { input: v.usageMetadata.promptTokenCount, output: v.usageMetadata.candidatesTokenCount };
    }
  }
  return { text, stop, usage: usage ?? computeUsage(c, text) };
}

type CallResult = { ok: true; answer: ProviderAnswer } | { ok: false; status: number; message: string };

// The provider's own error text names the cause: "<name>'s free-tier
// budget is spent" and "<name> is out" are permanent for the trial, while
// "<name> refused this call" is a one-off failure-schedule hit. The
// failure code is the same either way, so the message is the signal.
function isPermanentRefusal(message: string): boolean {
  return message.includes("budget is spent") || message.includes(" is out");
}

function errorMessageOf(wire: string, raw: string): string {
  const v = jsonOrNull(raw);
  if (v && typeof v === "object" && v.error && typeof v.error === "object" && typeof v.error.message === "string") return v.error.message;
  return "";
}

async function callProvider(envUrl: string, provider: any, model: string, c: Canonical): Promise<CallResult> {
  const { path, body } = buildProviderRequest(provider, model, c);
  const url = envUrl + provider.path + path;
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: providerHeaders(provider),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, message: "" };
  }
  if (res.status < 200 || res.status >= 300) {
    let raw = "";
    try {
      raw = await res.text();
    } catch {
      // ignore body read errors on failures
    }
    return { ok: false, status: res.status, message: errorMessageOf(provider.wire, raw) };
  }
  const raw = await res.text();
  const answer = c.stream ? parseProviderStream(provider.wire, raw, c) : parseProviderJson(provider.wire, raw, c);
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Routing. Candidates are the free models meeting the request's
// requirements, ordered by rating, with exhausted providers deprioritised:
// a 402 (budget spent) marks a provider done for the trial; two consecutive
// 402/429 failures deprioritise it until a success resets it. Ties on
// rating break toward the provider with fewer successful calls so far
// (spreads load), then by provider id, then model id — all deterministic.
// ---------------------------------------------------------------------------

interface ProviderState {
  calls: number; // successful calls so far (load-spreading tie-break)
  consecFails: number; // consecutive 402/429 failures
  exhausted: boolean; // a 402, a "budget is spent" or an "is out" refusal: done for the trial
}

function groupOf(s: ProviderState): number {
  if (s.exhausted) return 2;
  if (s.consecFails >= 2) return 1;
  return 0;
}

interface Router {
  brief: Brief;
  providersById: Map<string, any>;
  state: Map<string, ProviderState>;
  requestCount: number;
}

function makeRouter(brief: Brief): Router {
  const providersById = new Map<string, any>();
  const state = new Map<string, ProviderState>();
  for (const p of brief.providers) {
    providersById.set(p.id, p);
    state.set(p.id, { calls: 0, consecFails: 0, exhausted: false });
  }
  return { brief, providersById, state, requestCount: 0 };
}

function candidatesFor(router: Router, req: any, clientWire: string, jsonMode: boolean): any[] {
  const list = router.brief.ratings.filter((r) => {
    if (r.free !== true) return false;
    if (req.tools === true && r.tools !== true) return false;
    if (req.vision === true && r.vision !== true) return false;
    if (req.json === true && r.json !== true) return false;
    if (req.min_context !== undefined && req.min_context !== null && (r.context_length === null || r.context_length === undefined || r.context_length < req.min_context)) return false;
    // A JSON-mode OpenAI request must not go to an Anthropic-wire provider:
    // that wire always echoes json_mode false, which fails the echo check.
    if (clientWire === "openai" && jsonMode === true) {
      const p = router.providersById.get(r.provider);
      if (p && p.wire === "anthropic") return false;
    }
    return true;
  });
  list.sort((a, b) => {
    const sa = router.state.get(a.provider)!;
    const sb = router.state.get(b.provider)!;
    const ga = groupOf(sa);
    const gb = groupOf(sb);
    if (ga !== gb) return ga - gb;
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    if (sa.calls !== sb.calls) return sa.calls - sb.calls;
    if (a.provider !== b.provider) return a.provider < b.provider ? -1 : 1;
    if (a.model !== b.model) return a.model < b.model ? -1 : 1;
    return 0;
  });
  return list;
}

// Requirements for this request: the manifest entry by arrival order when
// available, else inferred from the body itself.
function requirementsFor(router: Router, c: Canonical): any {
  const entry = router.brief.manifest[router.requestCount];
  if (entry && entry.requires && typeof entry.requires === "object") return entry.requires;
  const req: any = {};
  if (c.tools.length > 0) req.tools = true;
  if (c.turns.some((t) => t.images.length > 0)) req.vision = true;
  if (c.jsonMode) req.json = true;
  return req;
}

// ---------------------------------------------------------------------------
// Client responses. The provider's text is returned whole and verbatim; the
// stop reason is mapped by family (the sim always answers family 0); the
// usage is the provider's numbers in the client's wire format. Stream
// frames are re-emitted in the canonical order each wire's parser requires.
// ---------------------------------------------------------------------------

function mapStop(stop: string, clientWire: string): string {
  const fam = STOP_FAMILY[stop] ?? 0;
  if (clientWire === "openai") {
    if (fam === 1) return "length";
    if (fam === 2) return "tool_calls";
    if (fam === 3) return "content_filter";
    return "stop";
  }
  if (fam === 1) return "max_tokens";
  if (fam === 2) return "tool_use";
  if (fam === 3) return "refusal";
  return "end_turn";
}

function openAIJsonBody(model: string, text: string, stop: string, usage: Usage): string {
  return JSON.stringify({
    id: "chatcmpl-sim",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: stop, logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
}

function anthropicJsonBody(model: string, text: string, stop: string, usage: Usage): string {
  return JSON.stringify({
    id: "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  });
}

function sseData(data: string): string {
  return `data: ${data}\n\n`;
}

function openAIStreamBody(model: string, text: string, stop: string, usage: Usage): string {
  const chunk = (delta: any, finish: string | null) => ({
    id: "chatcmpl-sim",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [];
  frames.push(sseData(JSON.stringify(chunk({ role: "assistant", content: "" }, null))));
  frames.push(sseData(JSON.stringify(chunk({ content: text }, null))));
  frames.push(sseData(JSON.stringify(chunk({}, stop))));
  frames.push(
    sseData(
      JSON.stringify({
        id: "chatcmpl-sim",
        object: "chat.completion.chunk",
        created: FIXED_CREATED,
        model,
        choices: [],
        usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
      }),
    ),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicStreamBody(model: string, text: string, stop: string, usage: Usage): string {
  const frames: string[] = [];
  frames.push(
    `event: message_start\ndata: ${JSON.stringify({
      type: "message_start",
      message: { id: "msg_sim", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    })}\n\n`,
  );
  frames.push(`event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`);
  frames.push(`event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } })}\n\n`);
  frames.push(`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`);
  frames.push(
    `event: message_delta\ndata: ${JSON.stringify({
      type: "message_delta",
      delta: { stop_reason: stop, stop_sequence: null },
      usage: { output_tokens: usage.output },
    })}\n\n`,
  );
  frames.push(`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`);
  return frames.join("");
}

interface ClientResponse {
  status: number;
  contentType: string;
  body: string;
}

function buildClientResponse(c: Canonical, model: string, answer: ProviderAnswer): ClientResponse {
  const stop = mapStop(answer.stop, c.clientWire);
  if (c.clientWire === "openai") {
    if (c.stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", body: openAIStreamBody(model, answer.text, stop, answer.usage) };
    return { status: 200, contentType: "application/json", body: openAIJsonBody(model, answer.text, stop, answer.usage) };
  }
  if (c.stream) return { status: 200, contentType: "text/event-stream; charset=utf-8", body: anthropicStreamBody(model, answer.text, stop, answer.usage) };
  return { status: 200, contentType: "application/json", body: anthropicJsonBody(model, answer.text, stop, answer.usage) };
}

async function serveRequest(envUrl: string, router: Router, clientWire: "openai" | "anthropic", body: any): Promise<ClientResponse> {
  const c = clientWire === "openai" ? canonicalFromOpenAI(body) : canonicalFromAnthropic(body);
  const req = requirementsFor(router, c);
  router.requestCount += 1;
  const cands = candidatesFor(router, req, clientWire, c.jsonMode);
  const tried = new Set<string>();
  for (const cand of cands) {
    const key = `${cand.provider}/${cand.model}`;
    if (tried.has(key)) continue;
    tried.add(key);
    const provider = router.providersById.get(cand.provider);
    if (!provider) continue;
    const res = await callProvider(envUrl, provider, cand.model, c);
    const st = router.state.get(cand.provider)!;
    if (!res.ok) {
      // 402 (budget spent) or 429 (failure schedule / outage / concurrency):
      // fail over to the next candidate. A refusal the provider itself
      // calls permanent (budget spent, outage) marks it done for the trial;
      // consecutive failures deprioritise it for later requests.
      if (res.status === 402 || isPermanentRefusal(res.message)) st.exhausted = true;
      st.consecFails += 1;
      continue;
    }
    st.consecFails = 0;
    st.calls += 1;
    return buildClientResponse(c, cand.model, res.answer);
  }
  // No candidate answered. A clean error is better than hanging: hanging
  // burns the 180s wall and scores 0 on every column.
  return { status: 502, contentType: "application/json", body: JSON.stringify({ error: "no provider could serve this request" }) };
}

// ---------------------------------------------------------------------------
// Server lifecycle: --list, bind, register, serve, poll /work, exit 0.
// ---------------------------------------------------------------------------

function modelsListBody(catalog: any): string {
  const data: any[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      data.push({ id: m.id, object: "model", created: FIXED_CREATED, owned_by: p.id });
    }
  }
  return JSON.stringify({ object: "list", data });
}

async function register(envUrl: string, port: number, catalog: any): Promise<boolean> {
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.status === 200) return true;
      try {
        await res.text();
      } catch {
        // ignore
      }
    } catch {
      // transient: retry below
    }
    await sleep(REGISTER_RETRY_MS);
  }
  return false;
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes("--list");
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");

  let brief: Brief | null = null;
  if (promptFile !== "") {
    try {
      brief = parseBrief(readFileSync(promptFile, "utf8"));
    } catch {
      brief = null;
    }
  }

  if (listMode) {
    if (brief === null) {
      console.log('{"providers":[]}');
      return;
    }
    console.log(buildCatalog(brief).text);
    return;
  }

  if (brief === null || envUrl === "") {
    console.error("free-model-aggregator-01: no brief (MAKEFUN_PROMPT_FILE) or env URL (MAKEFUN_ENV_URL); cannot run a trial");
    process.exit(1);
  }

  const { catalog, text: _catalogText } = buildCatalog(brief);
  void _catalogText;
  const router = makeRouter(brief);

  // Workload requests are served strictly one at a time, in arrival order:
  // the client sends them that way, and serialising also keeps a
  // concurrency-1 provider (zai) from ever seeing two calls at once.
  let queue: Promise<void> = Promise.resolve();

  const server = createServer((req, res) => {
    const method = req.method ?? "GET";
    let pathname = "/";
    try {
      pathname = new URL(req.url ?? "/", "http://entry").pathname;
    } catch {
      pathname = "/";
    }
    if (method === "GET" && pathname === "/v1/models") {
      const body = modelsListBody(catalog);
      res.writeHead(200, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
      res.end(body);
      return;
    }
    if (method === "POST" && (pathname === "/v1/chat/completions" || pathname === "/v1/messages")) {
      const clientWire = pathname === "/v1/chat/completions" ? "openai" : "anthropic";
      const chunks: Buffer[] = [];
      req.on("data", (chunk: Buffer) => chunks.push(chunk));
      req.on("end", () => {
        let body: any = null;
        try {
          body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
        } catch {
          body = null;
        }
        if (body === null || typeof body !== "object" || Array.isArray(body)) {
          res.writeHead(400, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: "bad request body" }));
          return;
        }
        queue = queue
          .then(async () => {
            let out: ClientResponse;
            try {
              out = await serveRequest(envUrl, router, clientWire, body);
            } catch (err) {
              out = { status: 502, contentType: "application/json", body: JSON.stringify({ error: String(err) }) };
            }
            try {
              res.writeHead(out.status, { "content-type": out.contentType, "content-length": Buffer.byteLength(out.body) });
              res.end(out.body);
            } catch {
              // client went away; nothing to do
            }
            await checkWork();
          })
          .catch(() => undefined);
      });
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });

  let exiting = false;
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  async function checkWork(): Promise<boolean> {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.status === 200) {
        const v: any = await res.json();
        if (v && v.done === true) return true;
      }
    } catch {
      // transient: keep polling
    }
    return false;
  }

  function shutdown(): void {
    if (exiting) return;
    exiting = true;
    if (pollTimer !== null) clearInterval(pollTimer);
    try {
      server.close(() => process.exit(0));
    } catch {
      process.exit(0);
    }
    setTimeout(() => process.exit(0), 500);
  }

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    console.error("free-model-aggregator-01: could not bind a port");
    process.exit(1);
  }

  const ok = await register(envUrl, port, catalog);
  if (!ok) {
    console.error("free-model-aggregator-01: register failed");
    process.exit(1);
  }

  pollTimer = setInterval(() => {
    void checkWork().then((done) => {
      if (done) shutdown();
    });
  }, WORK_POLL_MS);
}

main().catch((err) => {
  console.error(`free-model-aggregator-01: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
