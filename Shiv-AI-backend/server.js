/**
 * Shiv AI backend — free Node, no paid LLM
 * Modes: chat / compare / comprehensive / search / code / emotion / word-union
 * Think step + stronger scoring + dictionary (EN dictionaryapi.dev, ES free)
 *
 * Run:  node server.js
 * Env:  PORT=8787  ADMIN_PIN=yeahimtheownerhereicreateKEYS
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const PORT = Number(process.env.PORT || 8787);
const ADMIN_PIN = process.env.ADMIN_PIN || "yeahimtheownerhereicreateKEYS";
const DATA_DIR = path.join(__dirname, "data");
const LEARNED_PATH = path.join(DATA_DIR, "learned_samples.json");
const PUBLIC_DIR = path.join(__dirname, "..");

const SEED_FILES = [
  "basic_responses.json",
  "code_basics.json",
  "emotions_basics.json",
  "comprehensive_responses.json",
  "comparison_responses.json",
  "search_responses.json",
];

/** @type {Array<{id:string,q:string,a:string,tag?:string,at?:number,source?:string}>} */
let samples = [];
/** @type {Set<string>} */
let wordSet = new Set();

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2), "utf8");
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

function loadAll() {
  samples = [];
  for (const name of SEED_FILES) {
    const arr = readJson(path.join(DATA_DIR, name), []);
    if (!Array.isArray(arr)) continue;
    for (const s of arr) {
      if (!s || !s.q || !s.a) continue;
      samples.push({
        id: s.id || uid(),
        q: String(s.q),
        a: String(s.a),
        tag: s.tag ? String(s.tag) : "",
        at: s.at || Date.now(),
        source: name,
      });
    }
  }
  const learned = readJson(LEARNED_PATH, []);
  if (Array.isArray(learned)) {
    for (const s of learned) {
      if (!s || !s.q || !s.a) continue;
      samples.push({
        id: s.id || uid(),
        q: String(s.q),
        a: String(s.a),
        tag: s.tag ? String(s.tag) : "learned",
        at: s.at || Date.now(),
        source: "learned",
      });
    }
  }
  const wordsDoc = readJson(path.join(DATA_DIR, "basic_words.json"), { words: [] });
  wordSet = new Set((wordsDoc.words || []).map((w) => String(w).toLowerCase()));
  console.log(`[shiv-ai] loaded ${samples.length} samples, ${wordSet.size} words`);
}

function tokenize(s) {
  return String(s)
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9_+\-#.\u00f1]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function isGreeting(tokens, prompt) {
  const greets = new Set([
    "hi", "hello", "hey", "yo", "hola", "buenas", "morning", "evening", "night", "sup", "saludos",
  ]);
  const p = String(prompt).toLowerCase().trim();
  if (greets.has(p)) return true;
  if (tokens.length <= 3 && tokens.some((t) => greets.has(t))) return true;
  return false;
}

/** Detect intent mode from prompt */
function detectMode(prompt, tokens) {
  const p = String(prompt).toLowerCase();
  const has = (arr) => arr.some((k) => p.includes(k) || tokens.includes(k));

  if (
    has([
      "compare", "comparison", "vs", "versus", "difference", "differ", "contrast",
      "comparar", "comparacion", "diferencia", "diferencias", "contra",
    ])
  ) {
    return "compare";
  }
  if (
    has([
      "search", "buscar", "lookup", "look up", "define", "definition", "definicion",
      "qué es", "que es", "qué significa", "que significa", "meaning of", "what is a ", "what is an ",
    ])
  ) {
    return "search";
  }
  if (
    has([
      "explain", "how does", "how do", "why does", "comprehensive", "in detail", "overview",
      "explica", "explicar", "como funciona", "cómo funciona", "por que", "por qué", "detalle",
    ])
  ) {
    return "comprehensive";
  }
  if (
    has([
      "code", "snippet", "function", "script", "lua", "luau", "python", "javascript", "c++", "cpp",
      "codigo", "código", "funcion", "función",
    ])
  ) {
    return "code";
  }
  if (
    has([
      "synonym", "synonyms", "antonym", "antonyms", "word union", "related words", "similar words",
      "sinonimo", "sinónimo", "sinonimos", "antonimo", "antónimo", "palabras relacionadas",
    ])
  ) {
    return "wordunion";
  }
  if (
    has([
      "feel", "feeling", "emotion", "mood", "sad", "angry", "lonely", "anxious", "tired", "bored",
      "siento", "emocion", "emoción", "triste", "enojado", "ansioso", "cansado", "aburrido",
    ])
  ) {
    return "emotion";
  }
  // bare "happy" without feel → still emotion if only mood words
  if (tokens.length <= 3 && has(["happy", "feliz", "sad", "triste"])) {
    return "emotion";
  }
  return "chat";
}

function scoreSample(promptTokens, sample, mode) {
  const qTokens = tokenize(sample.q + " " + (sample.tag || "") + " " + (sample.a || "").slice(0, 120));
  const set = new Set(qTokens);
  let score = 0;
  let overlap = 0;
  // content tokens matter more than stopwords
  const stop = new Set([
    "what", "is", "a", "an", "the", "how", "do", "does", "to", "of", "in", "on", "for", "and", "or",
    "que", "es", "de", "la", "el", "un", "una", "como", "por", "con", "show", "me", "please",
  ]);
  const content = promptTokens.filter((t) => !stop.has(t));
  const useTokens = content.length ? content : promptTokens;

  for (const t of useTokens) {
    if (set.has(t)) {
      score += 5;
      overlap += 1;
    } else if (qTokens.some((q) => q.length > 2 && (q.includes(t) || t.includes(q)))) {
      score += 1.5;
      overlap += 0.5;
    }
  }

  // exact-ish phrase boost: share of content tokens covered
  if (useTokens.length) {
    const covered = useTokens.filter((t) => set.has(t)).length;
    score += (covered / useTokens.length) * 8;
  }

  // strong boost when sample.q tokens ~= prompt content tokens
  const qOnly = tokenize(sample.q);
  if (useTokens.length && qOnly.length) {
    const qSet = new Set(qOnly);
    const inter = useTokens.filter((t) => qSet.has(t)).length;
    if (inter === useTokens.length && inter === qOnly.length) score += 20;
    else if (inter === useTokens.length) score += 10;
  }

  // weak overlap penalty
  if (useTokens.length >= 1 && overlap < 1) score *= 0.12;
  if (useTokens.length >= 2 && overlap < 1.5) score *= 0.55;

  const tag = (sample.tag || "").toLowerCase();
  if (tag && promptTokens.includes(tag)) score += 5;

  // mode bias
  const modeTagBoost = {
    compare: ["compare", "comparison"],
    comprehensive: ["comprehensive", "basic"],
    search: ["search"],
    code: ["code", "lua", "python", "js", "cpp"],
    emotion: ["emotion"],
    chat: ["basic", "learned"],
    wordunion: ["basic", "search"],
  };
  const boostTags = modeTagBoost[mode] || [];
  if (boostTags.includes(tag)) score += 5;
  if (mode === "code" && (sample.source || "").includes("code")) score += 5;
  if (mode === "emotion" && (sample.source || "").includes("emotion")) score += 5;
  if (mode === "compare" && (sample.source || "").includes("comparison")) score += 10;
  if (mode === "comprehensive" && (sample.source || "").includes("comprehensive")) score += 10;
  if (mode === "search" && (sample.source || "").includes("search")) score += 10;

  // keep code from winning weak general prompts
  if ((tag === "code" || (sample.source || "").includes("code")) && mode !== "code") {
    if (overlap < 2.5) score *= 0.2;
  }

  // generic "computers work" style samples shouldn't win specific topics
  if ((sample.q || "").toLowerCase().includes("computers work") && !useTokens.includes("computer")) {
    score *= 0.2;
  }

  score += Math.min(0.6, (sample.a || "").length / 4000);
  if (sample.source === "learned" && overlap >= 2) score += 1.5;

  return { score, overlap };
}

function buildThink(mode, tokens, prompt) {
  const top = tokens.slice(0, 8).join(", ") || "(none)";
  const modeLabel = {
    compare: "comparison",
    comprehensive: "comprehensive explanation",
    search: "search / definition",
    code: "code",
    emotion: "emotion / tone",
    wordunion: "word union",
    chat: "general chat",
  }[mode] || mode;
  return `think: mode=${modeLabel} · keys=[${top}] · prompt_len=${String(prompt).length}`;
}

function httpGetJson(urlStr, timeoutMs = 4500) {
  return new Promise((resolve, reject) => {
    const u = new URL(urlStr);
    const lib = u.protocol === "https:" ? https : http;
    const req = lib.get(
      {
        hostname: u.hostname,
        path: u.pathname + u.search,
        protocol: u.protocol,
        headers: { Accept: "application/json", "User-Agent": "shiv-ai/1.0" },
        timeout: timeoutMs,
      },
      (res) => {
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const raw = Buffer.concat(chunks).toString("utf8");
          if (res.statusCode && res.statusCode >= 400) {
            return reject(new Error("http " + res.statusCode));
          }
          try {
            resolve(JSON.parse(raw));
          } catch (e) {
            reject(e);
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("timeout"));
    });
  });
}

function looksSpanish(prompt, tokens) {
  const p = String(prompt).toLowerCase();
  const hits = ["qué", "que", "cómo", "como", "por qué", "porque", "hola", "gracias", "buscar", "significa", "definición", "definicion", "español", "hola"];
  if (hits.some((h) => p.includes(h))) return true;
  // crude: presence of ñ or accented already stripped; use common tokens
  const esTokens = new Set(["hola", "que", "como", "por", "para", "con", "una", "uno", "esto", "esta", "buscar", "significa", "gracias", "buenas"]);
  let n = 0;
  for (const t of tokens) if (esTokens.has(t)) n++;
  return n >= 2;
}

async function dictLookup(word, preferEs) {
  const w = String(word || "").trim().toLowerCase();
  if (!w || w.length < 2) return null;

  // English free dictionary
  if (!preferEs) {
    try {
      const data = await httpGetJson(
        "https://api.dictionaryapi.dev/api/v2/entries/en/" + encodeURIComponent(w)
      );
      if (Array.isArray(data) && data[0]) {
        const e = data[0];
        const meanings = (e.meanings || []).slice(0, 2);
        const parts = [];
        parts.push(`**${e.word}**` + (e.phonetic ? ` · ${e.phonetic}` : ""));
        for (const m of meanings) {
          const defs = (m.definitions || []).slice(0, 2).map((d) => d.definition);
          parts.push(`_${m.partOfSpeech || "—"}:_ ${defs.join(" · ")}`);
          if (m.synonyms && m.synonyms.length) {
            parts.push(`synonyms: ${m.synonyms.slice(0, 6).join(", ")}`);
          }
          if (m.antonyms && m.antonyms.length) {
            parts.push(`antonyms: ${m.antonyms.slice(0, 4).join(", ")}`);
          }
        }
        return { lang: "en", text: parts.join("\n"), word: e.word };
      }
    } catch {
      /* fall through */
    }
  }

  // Spanish: free dictionaryapi often weak; try EN API still + local note
  // Also try official-ish public mirror patterns — keep resilient
  try {
    const data = await httpGetJson(
      "https://api.dictionaryapi.dev/api/v2/entries/es/" + encodeURIComponent(w)
    );
    if (Array.isArray(data) && data[0]) {
      const e = data[0];
      const parts = [`**${e.word}**`];
      for (const m of (e.meanings || []).slice(0, 2)) {
        const defs = (m.definitions || []).slice(0, 2).map((d) => d.definition);
        parts.push(`_${m.partOfSpeech || "—"}_: ${defs.join(" · ")}`);
      }
      return { lang: "es", text: parts.join("\n"), word: e.word };
    }
  } catch {
    /* ignore */
  }

  // local word presence
  if (wordSet.has(w)) {
    return {
      lang: preferEs ? "es" : "en",
      text: preferEs
        ? `**${w}** está en el vocabulario local (sin definición RAE en vivo). Entrena un Q→A o usa más contexto.`
        : `**${w}** is in the local word list (no live definition). Add a Train AI sample for a richer answer.`,
      word: w,
      local: true,
    };
  }
  return null;
}

function extractLookupWord(prompt, tokens) {
  const p = String(prompt);
  const m =
    p.match(/(?:define|definition|definici[oó]n|meaning of|qu[eé]\s+es(?:\s+un|\s+una)?|qu[eé]\s+significa|search|buscar|synonym(?:s)?\s+(?:for\s+)?|antonym(?:s)?\s+(?:for\s+)?)\s*["']?([a-zA-ZñÑáéíóúÁÉÍÓÚ0-9_+#.-]{2,})/i) ||
    p.match(/["']([a-zA-ZñÑ]{2,})["']/) ||
    null;
  if (m) return m[1].toLowerCase();
  const stop = new Set([
    "what", "is", "a", "an", "the", "de", "la", "el", "un", "una", "que", "qué", "como", "cómo",
    "search", "buscar", "define", "please", "por", "favor", "me", "you", "for", "synonym", "synonyms",
    "antonym", "antonyms", "related", "words", "word", "union", "similar",
  ]);
  for (let i = tokens.length - 1; i >= 0; i--) {
    if (!stop.has(tokens[i]) && tokens[i].length > 2) return tokens[i];
  }
  return tokens[0] || "";
}

function wordUnionLocal(word) {
  const w = String(word).toLowerCase();
  // tiny built-in unions for common stems
  const map = {
    happy: ["glad", "joyful", "content", "cheerful"],
    sad: ["unhappy", "down", "blue", "miserable"],
    fast: ["quick", "rapid", "swift", "speedy"],
    slow: ["sluggish", "leisurely", "unhurried"],
    code: ["program", "script", "source", "implementation"],
    bug: ["defect", "error", "fault", "issue"],
    aim: ["target", "point", "focus", "lock"],
    memory: ["ram", "storage", "address space"],
    strong: ["powerful", "sturdy", "robust"],
    weak: ["fragile", "frail", "feeble"],
  };
  if (map[w]) {
    return `**${w}** word union\nsynonyms / related: ${map[w].join(", ")}\n(local union — expand via Train AI)`;
  }
  // related by shared prefix in wordSet
  const related = [];
  for (const x of wordSet) {
    if (x === w) continue;
    if (x.startsWith(w.slice(0, 3)) && Math.abs(x.length - w.length) <= 4) related.push(x);
    if (related.length >= 8) break;
  }
  if (related.length) {
    return `**${w}** word union (local)\nrelated: ${related.join(", ")}`;
  }
  return null;
}

async function generateReply(prompt, fileContext) {
  const tokens = tokenize(prompt);
  const pLow = String(prompt).toLowerCase().trim();
  const mode = detectMode(prompt, tokens);
  const think = buildThink(mode, tokens, prompt);
  const es = looksSpanish(prompt, tokens);

  const greetMap = {
    hi: "hey. what's up?",
    hello: "hey. what's up?",
    hey: "hey. ready when you are.",
    yo: "yo. what are we building?",
    hola: "hola. ¿qué armamos?",
    buenas: "buenas. dime.",
    "good morning": "morning. what do you need?",
    "good night": "night. catch you later.",
  };
  if (greetMap[pLow]) {
    return {
      ok: true,
      text: `${think}\n\n${greetMap[pLow]}`,
      score: 99,
      sampleId: "greet",
      mode: "chat",
    };
  }
  if (isGreeting(tokens, prompt)) {
    return {
      ok: true,
      text: `${think}\n\n${es ? "hola. ¿en qué te ayudo?" : "hey. what's up?"}`,
      score: 99,
      sampleId: "greet",
      mode: "chat",
    };
  }

  // search / define / word-union → dictionary first
  if (mode === "search" || mode === "wordunion") {
    const word = extractLookupWord(prompt, tokens);
    let chunks = [think];
    if (mode === "wordunion") {
      const localU = wordUnionLocal(word);
      if (localU) chunks.push(localU);
    }
    const dict = await dictLookup(word, es);
    if (dict) {
      chunks.push(dict.text);
      if (mode === "wordunion" && dict.text.includes("synonyms")) {
        /* already rich */
      }
    } else if (mode === "search") {
      chunks.push(
        es
          ? `No encontré definición en vivo para **${word}**. Prueba Train AI o más contexto.`
          : `No live definition for **${word}**. Train a sample or add context.`
      );
    }

    // also blend best search-tagged sample
    let best = null;
    let bestScore = 0;
    for (const s of samples) {
      const { score } = scoreSample(tokens, s, mode);
      if (score > bestScore) {
        bestScore = score;
        best = s;
      }
    }
    if (best && bestScore >= 3) {
      chunks.push("\n— matched sample —\n" + best.a);
    }
    if (fileContext) chunks.push("\n(from your project)\n" + fileContext);
    chunks.push(`\n_matched · score ${bestScore.toFixed(1)} · mode ${mode}_`);
    return {
      ok: true,
      text: chunks.join("\n\n"),
      score: Math.max(bestScore, dict ? 8 : 0),
      sampleId: best && best.id,
      mode,
    };
  }

  // standard scored path with mode bias
  let best = null;
  let bestScore = 0;
  let bestOverlap = 0;
  for (const s of samples) {
    const { score, overlap } = scoreSample(tokens, s, mode);
    if (score > bestScore) {
      bestScore = score;
      bestOverlap = overlap;
      best = s;
    }
  }

  const threshold = mode === "code" ? 3.2 : mode === "compare" || mode === "comprehensive" ? 2.8 : 2.5;

  if (best && bestScore >= threshold) {
    let out = `${think}\n\n${best.a}`;
    // light word-union appendix if prompt asks related words casually
    if (tokens.includes("synonym") || tokens.includes("related")) {
      const w = extractLookupWord(prompt, tokens);
      const u = wordUnionLocal(w);
      if (u) out += "\n\n" + u;
    }
    if (fileContext) out += "\n\n(from your project)\n" + fileContext;
    out += `\n\n_matched · score ${bestScore.toFixed(1)} · overlap ${bestOverlap.toFixed(1)} · tag ${best.tag || "—"} · mode ${mode} · source ${best.source || "seed"}_`;
    return { ok: true, text: out, score: bestScore, sampleId: best.id, mode };
  }

  // fallback: try dictionary on unknown content words
  const contentWords = tokens.filter(
    (t) =>
      !["what", "is", "the", "a", "an", "how", "do", "does", "can", "you", "me", "please", "por", "que", "qué"].includes(t)
  );
  if (contentWords.length) {
    const dict = await dictLookup(contentWords[0], es);
    if (dict) {
      return {
        ok: true,
        text: `${think}\n\nNo strong training match.\n\n${dict.text}\n\nAdd a Q→A under Train AI for a direct answer.`,
        score: 1,
        sampleId: null,
        mode,
      };
    }
  }

  let out =
    `${think}\n\nNo strong training match yet.\n\n` +
    (es
      ? "Agrega un Q→A en Train AI (admin) para este prompt, o usa más palabras clave del tema."
      : "Add a Q→A under Train AI (admin) for this prompt, or ask with more keywords from your domain.");
  if (fileContext) out += "\n\n" + fileContext;
  return { ok: true, text: out, score: bestScore, sampleId: null, mode };
}

function send(res, status, body, extraHeaders) {
  const data = typeof body === "string" ? body : JSON.stringify(body);
  const headers = {
    "Content-Type": typeof body === "string" ? "text/plain; charset=utf-8" : "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Admin-Pin",
    ...extraHeaders,
  };
  res.writeHead(status, headers);
  res.end(data);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) return resolve({});
      try {
        resolve(JSON.parse(raw));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

function isAdmin(req, body) {
  const h = req.headers["x-admin-pin"];
  if (h && h === ADMIN_PIN) return true;
  if (body && body.pin === ADMIN_PIN) return true;
  return false;
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const map = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
  };
  return map[ext] || "application/octet-stream";
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const pathname = url.pathname;

  if (req.method === "OPTIONS") {
    return send(res, 204, "");
  }

  try {
    if (pathname === "/api/health" && req.method === "GET") {
      return send(res, 200, {
        ok: true,
        samples: samples.length,
        words: wordSet.size,
        name: "shiv-ai-backend",
        free: true,
        modes: ["chat", "compare", "comprehensive", "search", "code", "emotion", "wordunion"],
      });
    }

    if (pathname === "/api/stats" && req.method === "GET") {
      const byTag = {};
      const bySource = {};
      for (const s of samples) {
        byTag[s.tag || "none"] = (byTag[s.tag || "none"] || 0) + 1;
        bySource[s.source || "unknown"] = (bySource[s.source || "unknown"] || 0) + 1;
      }
      return send(res, 200, {
        samples: samples.length,
        words: wordSet.size,
        byTag,
        bySource,
      });
    }

    if (pathname === "/api/chat" && req.method === "POST") {
      const body = await readBody(req);
      const prompt = String(body.prompt || body.message || "").trim();
      if (!prompt) return send(res, 400, { ok: false, error: "prompt required" });
      const fileContext = body.fileContext ? String(body.fileContext).slice(0, 4000) : "";
      const result = await generateReply(prompt, fileContext);
      return send(res, 200, result);
    }

    if (pathname === "/api/dict" && req.method === "GET") {
      const word = String(url.searchParams.get("word") || "").trim();
      const lang = String(url.searchParams.get("lang") || "").toLowerCase();
      if (!word) return send(res, 400, { ok: false, error: "word required" });
      const hit = await dictLookup(word, lang === "es" || lang === "es");
      if (!hit) return send(res, 404, { ok: false, error: "not found", word });
      return send(res, 200, { ok: true, ...hit });
    }

    if (pathname === "/api/samples" && req.method === "GET") {
      if (!isAdmin(req, {})) {
        return send(res, 200, { ok: true, count: samples.length, samples: [] });
      }
      return send(res, 200, { ok: true, count: samples.length, samples });
    }

    if (pathname === "/api/train" && req.method === "POST") {
      const body = await readBody(req);
      if (!isAdmin(req, body)) return send(res, 401, { ok: false, error: "admin pin required" });
      const q = String(body.q || "").trim();
      const a = String(body.a || "").trim();
      const tag = String(body.tag || "learned").trim();
      if (!q || !a) return send(res, 400, { ok: false, error: "q and a required" });
      const sample = { id: uid(), q, a, tag, at: Date.now(), source: "learned" };
      samples.push(sample);
      const learned = samples.filter((s) => s.source === "learned");
      writeJson(LEARNED_PATH, learned);
      return send(res, 200, { ok: true, sample, total: samples.length });
    }

    if (pathname === "/api/train/import" && req.method === "POST") {
      const body = await readBody(req);
      if (!isAdmin(req, body)) return send(res, 401, { ok: false, error: "admin pin required" });
      const arr = Array.isArray(body.samples) ? body.samples : [];
      let added = 0;
      for (const s of arr) {
        if (!s || !s.q || !s.a) continue;
        samples.push({
          id: s.id || uid(),
          q: String(s.q),
          a: String(s.a),
          tag: String(s.tag || "learned"),
          at: s.at || Date.now(),
          source: "learned",
        });
        added++;
      }
      const learned = samples.filter((x) => x.source === "learned");
      writeJson(LEARNED_PATH, learned);
      return send(res, 200, { ok: true, added, total: samples.length });
    }

    if (pathname === "/api/train/reload" && req.method === "POST") {
      const body = await readBody(req);
      if (!isAdmin(req, body)) return send(res, 401, { ok: false, error: "admin pin required" });
      loadAll();
      return send(res, 200, { ok: true, samples: samples.length, words: wordSet.size });
    }

    if (pathname === "/api/words" && req.method === "GET") {
      return send(res, 200, { ok: true, count: wordSet.size });
    }

    // static
    let rel = pathname === "/" ? "/shiv-ai.html" : pathname;
    const candidate = path.normalize(path.join(PUBLIC_DIR, rel));
    if (candidate.startsWith(PUBLIC_DIR) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
      const buf = fs.readFileSync(candidate);
      res.writeHead(200, {
        "Content-Type": contentType(candidate),
        "Access-Control-Allow-Origin": "*",
      });
      return res.end(buf);
    }

    send(res, 404, { ok: false, error: "not found" });
  } catch (e) {
    console.error(e);
    send(res, 500, { ok: false, error: String(e.message || e) });
  }
});

loadAll();
server.listen(PORT, "0.0.0.0", () => {
  console.log(`[shiv-ai] http://0.0.0.0:${PORT}`);
  console.log(`[shiv-ai] health  GET  /api/health`);
  console.log(`[shiv-ai] chat    POST /api/chat`);
  console.log(`[shiv-ai] dict    GET  /api/dict?word=`);
  console.log(`[shiv-ai] train   POST /api/train  (header X-Admin-Pin)`);
});
