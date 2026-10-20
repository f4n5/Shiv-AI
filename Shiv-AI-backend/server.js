/**
 * Shiv AI backend — free Node, no paid LLM
 * Modes: chat / compare / comprehensive / search / code / emotion / word-union
 * Self-learn: high-confidence answers + feedback persist to learned_samples.json
 * Words: GET/POST /api/words  (add vocab that survives restarts)
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
const WORDS_PATH = path.join(DATA_DIR, "basic_words.json");
const PUBLIC_DIR = path.join(__dirname, "..");

const SEED_FILES = [
  "basic_responses.json",
  "code_basics.json",
  "emotions_basics.json",
  "comprehensive_responses.json",
  "comparison_responses.json",
  "search_responses.json",
  "conversational_responses.json",
];

/** @type {Array<{id:string,q:string,a:string,tag?:string,at?:number,source?:string,hits?:number}>} */
let samples = [];
/** @type {Set<string>} */
let wordSet = new Set();
/** @type {Map<string, number>} recent auto-learn keys to avoid spam */
const recentLearn = new Map();

/** Markov brain — learns sentence flow from all answers so it can invent lines */
/** @type {Map<string, Map<string, number>>} */
let markov = new Map();
/** @type {string[]} sentence starters */
let starters = [];
/** @type {Map<string, string[]>} short phrase banks by mode/tag */
let phraseBank = new Map();

function markovSize() {
  let n = 0;
  for (const m of markov.values()) n += m.size;
  return n;
}

function pushMarkov(a, b) {
  if (!a || !b) return;
  if (!markov.has(a)) markov.set(a, new Map());
  const row = markov.get(a);
  row.set(b, (row.get(b) || 0) + 1);
}

function wordsForMarkov(text) {
  return String(text)
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[_*`#]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((w) => w.trim())
    .filter((w) => w.length > 0 && w.length < 40);
}

function rebuildBrain() {
  markov = new Map();
  starters = [];
  phraseBank = new Map();
  for (const s of samples) {
    const a = String(s.a || "");
    if (a.length < 4) continue;
    // phrase bank by tag/mode
    const key = (s.tag || s.source || "general").toLowerCase();
    if (!phraseBank.has(key)) phraseBank.set(key, []);
    const bank = phraseBank.get(key);
    // split into sentence-ish chunks
    for (const part of a.split(/(?<=[.!?])\s+/)) {
      const cleaned = part.replace(/\n+/g, " ").trim();
      if (cleaned.length >= 8 && cleaned.length <= 220 && !cleaned.includes("```")) {
        bank.push(cleaned);
      }
    }
    const words = wordsForMarkov(a);
    if (words.length < 2) continue;
    starters.push(words[0]);
    for (let i = 0; i < words.length - 1; i++) {
      pushMarkov(words[i], words[i + 1]);
      if (i < words.length - 2) {
        // bigram key → next
        pushMarkov(words[i] + " " + words[i + 1], words[i + 2]);
      }
    }
  }
  // keep banks bounded
  for (const [k, arr] of phraseBank) {
    if (arr.length > 400) phraseBank.set(k, arr.slice(-400));
  }
  if (starters.length > 2000) starters = starters.slice(-2000);
}

function pickWeighted(row) {
  if (!row || !row.size) return null;
  let total = 0;
  for (const c of row.values()) total += c;
  let r = Math.random() * total;
  for (const [w, c] of row) {
    r -= c;
    if (r <= 0) return w;
  }
  return [...row.keys()][0];
}

/** Generate a novel sentence via markov, biased by prompt tokens */
function markovSpeak(promptTokens, maxWords = 28) {
  if (!markov.size) return null;
  const stop = new Set(["the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on", "for"]);
  const seeds = promptTokens.filter((t) => !stop.has(t) && markov.has(t));
  let cur = seeds.length ? seeds[Math.floor(Math.random() * seeds.length)] : null;
  if (!cur) {
    cur = starters.length ? starters[Math.floor(Math.random() * starters.length)] : null;
  }
  if (!cur || !markov.has(cur)) {
    // random start key
    const keys = [...markov.keys()].filter((k) => !k.includes(" "));
    if (!keys.length) return null;
    cur = keys[Math.floor(Math.random() * keys.length)];
  }
  const out = [cur];
  let prev = cur;
  for (let i = 0; i < maxWords; i++) {
    // prefer bigram continuity
    let next = null;
    if (out.length >= 2) {
      const big = out[out.length - 2] + " " + out[out.length - 1];
      if (markov.has(big)) next = pickWeighted(markov.get(big));
    }
    if (!next && markov.has(prev)) next = pickWeighted(markov.get(prev));
    if (!next) break;
    out.push(next);
    prev = next;
    if (/[.!?]$/.test(next) && out.length > 6) break;
  }
  let text = out.join(" ").replace(/\s+([,.!?])/g, "$1");
  // capitalize first letter
  text = text.charAt(0).toUpperCase() + text.slice(1);
  if (!/[.!?]$/.test(text)) text += ".";
  return text;
}

function pickPhrase(tags) {
  for (const tag of tags) {
    const bank = phraseBank.get(tag);
    if (bank && bank.length) return bank[Math.floor(Math.random() * bank.length)];
  }
  // any bank
  for (const bank of phraseBank.values()) {
    if (bank && bank.length) return bank[Math.floor(Math.random() * bank.length)];
  }
  return null;
}

/** Intent-aware free talk — invents a line, does not paste a full JSON answer */
function inventReply(prompt, tokens, mode, lane) {
  const p = String(prompt);
  const content = tokens.filter(
    (t) =>
      !["the", "and", "for", "that", "this", "with", "from", "what", "how", "are", "you", "is", "a", "an", "do", "does", "can", "me", "my", "to", "of", "in", "on"].includes(t)
  );
  const topic = content.slice(0, 5).join(" ") || "that";
  const es = looksSpanish(prompt, tokens);
  const mark = markovSpeak(tokens, mode === "code" ? 20 : 32);
  const emojiPool = ["🔥", "💪", "👀", "✨", "👍", ""];
  const emoji = emojiPool[Math.floor(Math.random() * emojiPool.length)];

  // situational templates that fill with topic + optional markov clause
  const clause = mark ? mark.replace(/\.$/, "") : null;

  if (lane === "free" || mode === "chat" || mode === "emotion") {
    if (isGreeting(tokens, prompt) || /^(hey|hi|hello|yo)\b/i.test(p)) {
      const opts = es
        ? [`hola. acá estoy ${emoji} ¿qué armamos?`, `buenas. decime y lo encaramos ${emoji}`, `hola — listo cuando vos.`]
        : [
            `hey. solid over here ${emoji} what's the move?`,
            `yo. i'm here — drop whatever's on your mind ${emoji}`,
            `hey. free to talk or build, your call.`,
          ];
      return opts[Math.floor(Math.random() * opts.length)].replace(/\s+/g, " ").trim();
    }
    if (/how are you|how r u|como estas|cómo estás/i.test(p)) {
      return es
        ? `bien, enfocado. vos? ${emoji}`
        : `good. locked in. you? ${emoji}`.replace(/\s+/g, " ").trim();
    }
    if (/thank|gracias|thx/i.test(p)) {
      return es ? `de nada. cuando quieras seguimos ${emoji}` : `anytime. that's what i'm for ${emoji}`.trim();
    }
    if (/bye|chau|goodbye|see you|gn\b/i.test(p)) {
      return es ? `chau. acá quedo.` : `later. i don't leave — just quiet until you're back.`;
    }
    if (/bored|aburrido/i.test(p)) {
      return `bored is free real estate. want a small build, a weird idea, or just talk ${emoji}?`;
    }
    if (/stuck|help|problem|issue|broken|error|fail/i.test(p)) {
      const bit = clause || pickPhrase(["chat", "basic", "learned"]) || "split it into one next step";
      return `i hear you on **${topic}**. try this angle: ${bit}. if it's wrong, rewrite me and i'll learn.`;
    }
    if (/what do you think|opinion|should i/i.test(p)) {
      return `on **${topic}**: ${clause || "i'd keep it simple and ship a small version first"}. correct me if that misses your goal.`;
    }
    if (/who are you|what are you|are you ai/i.test(p)) {
      return `shiv. his ai. i talk free, learn when you correct me, and stay on the work.`;
    }
    // general free invention
    if (clause && content.length) {
      const frames = [
        `on **${topic}** — ${clause}. ${emoji}`,
        `my take: ${clause}. topic stays **${topic}**. say if that's off.`,
        `${clause} — that's where i land on **${topic}** ${emoji}`,
        `thinking out loud on **${topic}**: ${clause}. train me if you want a different line.`,
      ];
      return frames[Math.floor(Math.random() * frames.length)].replace(/\s+/g, " ").trim();
    }
    if (clause) {
      return `${clause} ${emoji}`.trim();
    }
    const soft = pickPhrase(["chat", "basic", "emotion", "learned"]);
    if (soft) return soft + (emoji ? " " + emoji : "");
    return es
      ? `sobre **${topic}** todavía estoy armando la frase. corregime y aprendo.`
      : `on **${topic}** i don't have a clean line yet — correct me and i lock it.`;
  }

  if (mode === "comprehensive" || mode === "compare") {
    const bit = clause || pickPhrase(["comprehensive", "compare", "basic"]) || "";
    return bit
      ? `here's a free-form pass on **${topic}**: ${bit}. expand or correct via teach.`
      : `i can outline **${topic}** once you train a fuller sample — or correct a draft.`;
  }

  if (mode === "code" || lane === "code") {
    return `no locked snippet for **${topic}** yet. paste the right code under teach / Train AI and i'll reuse the pattern.`;
  }

  return clause || `still forming a line for **${topic}**. correct me.`;
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

function writeJson(file, data) {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    fs.writeFileSync(file, JSON.stringify(data, null, 2), "utf8");
  } catch (e) {
    console.error("[shiv-ai] write fail", file, e.message);
  }
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

function persistLearned() {
  const learned = samples.filter((s) => s.source === "learned");
  // keep last 2000 learned max
  const trimmed = learned.slice(-2000);
  writeJson(LEARNED_PATH, trimmed);
}

function persistWords() {
  writeJson(WORDS_PATH, {
    version: 2,
    count: wordSet.size,
    words: Array.from(wordSet).sort(),
  });
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
        hits: 0,
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
        hits: s.hits || 0,
      });
    }
  }
  const wordsDoc = readJson(WORDS_PATH, { words: [] });
  wordSet = new Set((wordsDoc.words || []).map((w) => String(w).toLowerCase()));
  rebuildBrain();
  console.log(`[shiv-ai] loaded ${samples.length} samples, ${wordSet.size} words, markov ${markovSize()}`);
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
    "gm", "gn", "hru", "wyd",
  ]);
  const p = String(prompt).toLowerCase().trim()
    .replace(/[🔥👋☀️💪😂👀✨❤️💜🔥]+/g, "")
    .replace(/[!?.,]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (greets.has(p)) return true;
  if (/^(hey|hi|hello|yo|hola)\b/.test(p) && tokens.length <= 6) return true;
  if (tokens.length <= 4 && tokens.some((t) => greets.has(t))) return true;
  return false;
}

function isCasualTalk(prompt, tokens) {
  const p = String(prompt).toLowerCase();
  const patterns = [
    /how are you/, /how r u/, /how's it going/, /how is it going/,
    /what'?s up/, /what are you doing/, /you there/, /are you there/,
    /thank/, /thanks/, /thx/, /bye/, /goodbye/, /see you/, /good night/, /good morning/,
    /i'?m bored/, /i'?m tired/, /i'?m back/, /miss you/, /love you/,
    /como estas/, /cómo estás/, /que tal/, /qué tal/, /gracias/, /chau/,
    /let'?s go/, /lets build/, /you'?re the best/, /you'?re cool/, /you'?re funny/,
    /talk to me/, /be honest/, /don'?t lie/, /make me laugh/, /i did it/, /it works/,
  ];
  if (patterns.some((re) => re.test(p))) return true;
  if (tokens.length <= 5 && tokens.some((t) => ["hru", "wyd", "lol", "lmao", "brb", "gm", "gn"].includes(t))) return true;
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
      "synonym", "synonyms", "antonym", "antonyms", "word union", "related words", "similar words",
      "sinonimo", "sinónimo", "sinonimos", "antonimo", "antónimo", "palabras relacionadas",
    ])
  ) {
    return "wordunion";
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
      "codigo", "código", "funcion", "función", "```",
    ])
  ) {
    return "code";
  }
  if (
    has([
      "feel", "feeling", "emotion", "mood", "sad", "angry", "lonely", "anxious", "tired", "bored",
      "siento", "emocion", "emoción", "triste", "enojado", "ansioso", "cansado", "aburrido",
    ])
  ) {
    return "emotion";
  }
  if (tokens.length <= 3 && has(["happy", "feliz", "sad", "triste"])) {
    return "emotion";
  }
  if (isCasualTalk(prompt, tokens) || isGreeting(tokens, prompt)) {
    return "chat";
  }
  return "chat";
}

function scoreSample(promptTokens, sample, mode) {
  const qTokens = tokenize(sample.q + " " + (sample.tag || "") + " " + (sample.a || "").slice(0, 120));
  const set = new Set(qTokens);
  let score = 0;
  let overlap = 0;
  // keep pronouns for chat/emotion so "how are you" still matches
  const stopHard = new Set([
    "what", "is", "a", "an", "the", "to", "of", "in", "on", "for", "and", "or",
    "que", "es", "de", "la", "el", "un", "una", "por", "con", "show", "please", "can",
    "this", "that", "with", "from",
  ]);
  const stopSoft = new Set(["how", "do", "does", "como", "me", "my", "i"]);
  const stop = mode === "chat" || mode === "emotion"
    ? stopHard
    : new Set([...stopHard, ...stopSoft, "you"]);
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

  if (useTokens.length) {
    const covered = useTokens.filter((t) => set.has(t)).length;
    score += (covered / useTokens.length) * 8;
  }

  const qOnly = tokenize(sample.q);
  if (useTokens.length && qOnly.length) {
    const qSet = new Set(qOnly);
    const inter = useTokens.filter((t) => qSet.has(t)).length;
    if (inter === useTokens.length && inter === qOnly.length) score += 22;
    else if (inter === useTokens.length) score += 12;
    else if (inter >= Math.ceil(useTokens.length * 0.75)) score += 6;
  }

  // weak overlap kill — main anti-nonsense
  if (useTokens.length >= 1 && overlap < 1) score *= 0.05;
  if (useTokens.length >= 2 && overlap < 1.5) score *= 0.25;
  if (useTokens.length >= 3 && overlap < 2) score *= 0.45;
  // pure noise prompts (no real token hits) die
  if (overlap < 1 && useTokens.length >= 2) score = 0;

  const tag = (sample.tag || "").toLowerCase();
  if (tag && promptTokens.includes(tag)) score += 5;

  const modeTagBoost = {
    compare: ["compare", "comparison"],
    comprehensive: ["comprehensive", "basic"],
    search: ["search"],
    code: ["code", "lua", "python", "js", "cpp"],
    emotion: ["emotion"],
    chat: ["basic", "learned", "chat"],
    wordunion: ["basic", "search"],
  };
  const boostTags = modeTagBoost[mode] || [];
  if (boostTags.includes(tag)) score += 5;
  if (mode === "code" && (sample.source || "").includes("code")) score += 5;
  if (mode === "emotion" && (sample.source || "").includes("emotion")) score += 5;
  if (mode === "compare" && (sample.source || "").includes("comparison")) score += 10;
  if (mode === "comprehensive" && (sample.source || "").includes("comprehensive")) score += 10;
  if (mode === "search" && (sample.source || "").includes("search")) score += 10;
  if (mode === "chat" && (sample.source || "").includes("conversational")) score += 12;
  if (mode === "chat" && tag === "chat") score += 6;

  if ((tag === "code" || (sample.source || "").includes("code")) && mode !== "code") {
    if (overlap < 2.5) score *= 0.15;
  }

  if ((sample.q || "").toLowerCase().includes("computers work") && !useTokens.includes("computer")) {
    score *= 0.15;
  }

  // learned samples that actually matched before get a small boost
  if (sample.source === "learned" && overlap >= 2) score += 2 + Math.min(3, (sample.hits || 0) * 0.3);

  score += Math.min(0.5, (sample.a || "").length / 5000);
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
  const hits = [
    "qué", "que", "cómo", "como", "por qué", "porque", "hola", "gracias", "buscar",
    "significa", "definición", "definicion", "español", "hola", "sí", "si",
  ];
  if (hits.some((h) => p.includes(h))) return true;
  const esTokens = new Set([
    "hola", "que", "como", "por", "para", "con", "una", "uno", "esto", "esta",
    "buscar", "significa", "gracias", "buenas",
  ]);
  let n = 0;
  for (const t of tokens) if (esTokens.has(t)) n++;
  return n >= 2;
}

async function dictLookup(word, preferEs) {
  const w = String(word || "").trim().toLowerCase();
  if (!w || w.length < 2) return null;

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
    p.match(
      /(?:define|definition|definici[oó]n|meaning of|qu[eé]\s+es(?:\s+un|\s+una)?|qu[eé]\s+significa|search|buscar|synonym(?:s)?\s+(?:for\s+)?|antonym(?:s)?\s+(?:for\s+)?)\s*["']?([a-zA-ZñÑáéíóúÁÉÍÓÚ0-9_+#.-]{2,})/i
    ) ||
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

/** Auto-learn high-confidence answers so next time is stronger */
function maybeAutoLearn(prompt, answer, mode, score, sampleId) {
  if (!prompt || !answer) return false;
  if (score < 12) return false; // only strong matches
  const key = tokenize(prompt).slice(0, 6).join(" ");
  if (!key || key.length < 4) return false;
  const now = Date.now();
  if (recentLearn.has(key) && now - recentLearn.get(key) < 60_000) return false;
  recentLearn.set(key, now);

  // don't duplicate near-identical q
  const exists = samples.some((s) => {
    const a = tokenize(s.q).join(" ");
    return a === key || s.q.toLowerCase().trim() === prompt.toLowerCase().trim();
  });
  if (exists) {
    // bump hits on existing
    const hit = samples.find(
      (s) => s.id === sampleId || tokenize(s.q).join(" ") === key
    );
    if (hit) {
      hit.hits = (hit.hits || 0) + 1;
      if (hit.source === "learned") persistLearned();
    }
    return false;
  }

  // store a cleaned answer without think/meta footer
  let clean = String(answer)
    .replace(/^think:.*$/im, "")
    .replace(/\n*_matched ·[\s\S]*$/i, "")
    .trim();
  if (clean.length < 8 || clean.length > 4000) return false;

  const sample = {
    id: uid(),
    q: prompt.slice(0, 240),
    a: clean,
    tag: mode === "chat" ? "learned" : mode,
    at: now,
    source: "learned",
    hits: 1,
  };
  samples.push(sample);
  persistLearned();
  rebuildBrain();
  console.log(`[shiv-ai] auto-learned q="${sample.q.slice(0, 40)}…" score=${score.toFixed(1)}`);
  return true;
}

/** Harvest content words from a successful exchange into vocab */
function harvestWords(prompt, answer) {
  const stop = new Set([
    "the", "and", "for", "that", "this", "with", "from", "your", "have", "are", "was", "were",
    "will", "can", "not", "but", "you", "all", "any", "how", "what", "when", "where", "who",
    "why", "which", "into", "than", "then", "them", "they", "their", "there", "here", "about",
    "just", "like", "some", "more", "also", "only", "over", "such", "other", "into", "code",
  ]);
  const tokens = tokenize(prompt + " " + String(answer).slice(0, 400));
  let added = 0;
  for (const t of tokens) {
    if (t.length < 3 || t.length > 24) continue;
    if (stop.has(t)) continue;
    if (/^\d+$/.test(t)) continue;
    if (!wordSet.has(t)) {
      wordSet.add(t);
      added++;
    }
  }
  if (added) persistWords();
  return added;
}

/** Mix fragments + invent free-form reply (self-speak) */
function composeFromParts(prompt, tokens, mode, ranked, lane) {
  // primary: invent original sentence from markov + intent
  const invented = inventReply(prompt, tokens, mode, lane);
  // secondary candidates from nearby samples (for picker), not as main voice
  const bits = [];
  for (const r of ranked.slice(0, 5)) {
    const cut = String(r.a || "").replace(/```[\s\S]*?```/g, "").replace(/\n+/g, " ").trim();
    if (cut.length >= 12 && cut.length <= 160) bits.push(cut);
  }
  return {
    text: invented,
    composed: true,
    needsReview: true,
    parts: { bits: bits.slice(0, 3), content: tokens.slice(0, 8) },
  };
}

async function generateReply(prompt, fileContext, chatMode) {
  const tokens = tokenize(prompt);
  const pLow = String(prompt)
    .toLowerCase()
    .trim()
    .replace(/[🔥👋☀️💪😂👀✨❤️💜]+/g, "")
    .replace(/[!?.,]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  // chatMode: "free" | "code" — user-selected lane
  const lane = chatMode === "code" ? "code" : "free";
  let mode = detectMode(prompt, tokens);
  if (lane === "code") {
    // code chat: force code thinking; only fall to search/wordunion if explicit
    if (mode !== "search" && mode !== "wordunion") mode = "code";
  } else {
    // free chat: prefer chat / emotion / comprehensive situational talk
    if (mode === "code") mode = "chat";
  }
  const thinkBase = buildThink(mode, tokens, prompt);
  const think = `think: lane=${lane} · ` + thinkBase.replace(/^think:\s*/i, "");
  const es = looksSpanish(prompt, tokens);

  const greetMap = {
    hi: "hey! what's up? 🔥",
    hello: "Hey! how are you doing? solid over here 🔥",
    hey: "hey! ready when you are 🔥",
    yo: "yo. what are we building?",
    hola: "hola! ¿qué armamos? 🔥",
    buenas: "buenas. dime.",
    sup: "sup. go.",
    morning: "morning ☀️ what's the task?",
    gm: "gm ☀️ drop it.",
    gn: "gn. later 👋",
    hru: "good. you?",
    wyd: "waiting on you. go.",
  };

  // rank samples once — filtered by lane
  function rankAll() {
    const ranked = [];
    for (const s of samples) {
      const src = (s.source || "").toLowerCase();
      const tag = (s.tag || "").toLowerCase();
      if (lane === "code") {
        // code chat: only learned + code seeds (no pure smalltalk)
        const ok =
          src.includes("code") ||
          src === "learned" ||
          ["code", "lua", "python", "js", "cpp", "corrected", "composed-ok", "confirmed", "learned"].includes(tag) ||
          /```/.test(s.a || "");
        if (!ok) continue;
      } else {
        // free chat: skip pure code dumps unless prompt asks
        if ((src.includes("code") || tag === "code" || tag === "lua") && mode === "chat") {
          // allow weak keep only if strong overlap later via score
        }
      }
      let { score, overlap } = scoreSample(tokens, s, mode);
      if (lane === "code") {
        if (src.includes("code") || /```/.test(s.a || "")) score += 8;
        if (src === "learned") score += 4;
      } else {
        if (src.includes("conversational") || tag === "chat" || tag === "emotion") score += 8;
        if (src.includes("comprehensive") || tag === "comprehensive") score += 3;
        // penalize code blocks in free lane
        if (/```/.test(s.a || "") && mode === "chat") score *= 0.15;
      }
      if (score <= 0) continue;
      ranked.push({
        id: s.id,
        q: s.q,
        a: s.a,
        tag: s.tag || "",
        source: s.source || "",
        score,
        overlap,
      });
    }
    ranked.sort((a, b) => b.score - a.score);
    return ranked;
  }

  function packCandidates(ranked, limit = 5) {
    return ranked.slice(0, limit).map((c, i) => ({
      rank: i + 1,
      id: c.id,
      text: c.a,
      score: Number(c.score.toFixed(2)),
      tag: c.tag,
      source: c.source,
      overlap: Number((c.overlap || 0).toFixed(2)),
    }));
  }

  // hard greetings — free lane only (code lane stays technical)
  if (lane === "free" && (greetMap[pLow] || isGreeting(tokens, prompt))) {
    const text = greetMap[pLow] || "hey! what's up? 🔥";
    const ranked = rankAll();
    const cands = [
      { rank: 1, id: "greet", text, score: 99, tag: "chat", source: "greet", overlap: 1 },
      ...packCandidates(ranked, 4).map((c, i) => ({ ...c, rank: i + 2 })),
    ];
    return {
      ok: true,
      text: `${think}\n\n${text}`,
      score: 99,
      sampleId: "greet",
      mode: "chat",
      chatMode: lane,
      learned: false,
      composed: false,
      needsReview: true,
      candidates: cands.slice(0, 5),
      pickHint: "choose a candidate below to teach / lock that reply",
    };
  }

  // search / define / word-union
  if (mode === "search" || mode === "wordunion") {
    const word = extractLookupWord(prompt, tokens);
    const chunks = [think];
    let dict = null;
    if (mode === "wordunion") {
      const u = wordUnionLocal(word);
      if (u) chunks.push(u);
    }
    dict = await dictLookup(word, es);
    if (dict) {
      chunks.push(dict.text);
    } else if (mode === "wordunion") {
      const u = wordUnionLocal(word);
      if (u) chunks.push(u);
      else
        chunks.push(
          es
            ? `No encontré unión para **${word}**. Entrena un sample o agrega la palabra.`
            : `No union for **${word}**. Train a sample or add the word.`
        );
    } else {
      chunks.push(
        es
          ? `No encontré definición en vivo para **${word}**. Prueba Train AI o más contexto.`
          : `No live definition for **${word}**. Train a sample or add context.`
      );
    }

    const ranked = rankAll();
    if (ranked[0] && ranked[0].score >= 8) {
      chunks.push("— matched sample —\n" + ranked[0].a);
    }
    if (fileContext) chunks.push("(from your project)\n" + fileContext);
    const topScore = Math.max(ranked[0] ? ranked[0].score : 0, dict ? 8 : 0);
    chunks.push(`_matched · score ${topScore.toFixed(1)} · mode ${mode}_`);
    const text = chunks.join("\n\n");
    const learned = maybeAutoLearn(prompt, text, mode, topScore, ranked[0] && ranked[0].id);
    harvestWords(prompt, text);
    return {
      ok: true,
      text,
      score: topScore,
      sampleId: ranked[0] && ranked[0].id,
      mode,
      learned,
      candidates: packCandidates(ranked, 5),
      pickHint: "pick the best answer to lock it as training",
    };
  }

  // standard ranked path
  const ranked = rankAll();
  const best = ranked[0] || null;
  const bestScore = best ? best.score : 0;
  const bestOverlap = best ? best.overlap : 0;
  const candidates = packCandidates(ranked, 5);

  let threshold =
    mode === "code" ? 6.0 : mode === "compare" || mode === "comprehensive" ? 5.5 : mode === "emotion" ? 5.0 : mode === "chat" ? 5.5 : 8.0;
  if (lane === "free") {
    // free chat: almost always invent; only lock near-exact trained lines
    threshold = mode === "emotion" ? 14.0 : 28.0;
  }
  if (lane === "code") {
    // code chat: prefer learned/code hits; lower bar for code samples
    threshold = 5.0;
  }

  // free lane: only lock near-exact trained Q (not loose keyword hits)
  let allowLock = best && bestScore >= threshold && bestOverlap >= (mode === "chat" || mode === "emotion" ? 0.9 : 1.2);
  if (allowLock && lane === "free" && best) {
    const qTok = tokenize(best.q);
    const content = tokens.filter((x) => !["the","a","an","to","of","and","or","in","on","for","is","are","do","does"].includes(x));
    const inter = content.filter((x) => qTok.includes(x)).length;
    const exactish = content.length && inter >= Math.ceil(content.length * 0.85) && qTok.length <= content.length + 2;
    // greetings handled earlier; for normal free talk require exactish or huge score
    // free talk prefers invention; only lock rock-solid trained lines
    if (!exactish || bestScore < 70) allowLock = false;
    // short casual exact qs (thanks, how are you) may lock if score high
    if (exactish && tokens.length <= 5 && bestScore >= 40) allowLock = true;
  }

  if (allowLock) {
    let out = `${think}\n\n${best.a}`;
    if (tokens.includes("synonym") || tokens.includes("related")) {
      const w = extractLookupWord(prompt, tokens);
      const u = wordUnionLocal(w);
      if (u) out += "\n\n" + u;
    }
    if (fileContext) out += "\n\n(from your project)\n" + fileContext;
    out += `\n\n_matched · score ${bestScore.toFixed(1)} · overlap ${bestOverlap.toFixed(1)} · tag ${best.tag || "—"} · mode ${mode} · source ${best.source || "seed"}_`;
    // bump hits
    const live = samples.find((s) => s.id === best.id);
    if (live) live.hits = (live.hits || 0) + 1;
    const learned = maybeAutoLearn(prompt, out, mode, bestScore, best.id);
    harvestWords(prompt, best.a);
    const needsReview = mode === "chat" || bestScore < 18;
    return {
      ok: true,
      text: out,
      score: bestScore,
      sampleId: best.id,
      mode,
      learned,
      composed: false,
      needsReview,
      candidates,
      pickHint: candidates.length > 1 ? "not perfect? pick another candidate to teach" : (needsReview ? "mark correct or rewrite if off" : undefined),
    };
  }

  // free / weak match → invent original sentence (markov + intent), user corrects
  const composed = composeFromParts(prompt, tokens, mode, ranked, lane);
  let out = `${think}\n\n${composed.text}`;
  if (fileContext) out += "\n\n(from your project)\n" + fileContext;
  out += `\n\n_self-spoke · invented · mode ${mode}_`;
  harvestWords(prompt, composed.text);
  const invCand = {
    rank: 1,
    id: "invented",
    text: composed.text,
    score: 3,
    tag: "invented",
    source: "markov",
    overlap: 0,
  };
  const merged = [invCand, ...candidates.filter((c) => c.text !== composed.text)].slice(0, 5);
  return {
    ok: true,
    text: out,
    score: Math.max(bestScore, 2),
    sampleId: null,
    mode,
    learned: false,
    composed: true,
    needsReview: true,
    candidates: merged,
    pickHint: "invented line — teach this if good, or rewrite via Train / teach",
  };
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
        selfLearn: true,
        generative: true,
        markov: markovSize(),
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
        learned: (bySource.learned || 0),
      });
    }

    if (pathname === "/api/chat" && req.method === "POST") {
      const body = await readBody(req);
      const prompt = String(body.prompt || body.message || "").trim();
      if (!prompt) return send(res, 400, { ok: false, error: "prompt required" });
      const fileContext = body.fileContext ? String(body.fileContext).slice(0, 4000) : "";
      const chatMode = String(body.chatMode || body.mode || "free").toLowerCase() === "code" ? "code" : "free";
      const result = await generateReply(prompt, fileContext, chatMode);
      result.chatMode = chatMode;
      return send(res, 200, result);
    }

    // explicit feedback: teach this Q→A (anyone can reinforce; admin optional)
    if (pathname === "/api/learn" && req.method === "POST") {
      const body = await readBody(req);
      const q = String(body.q || body.prompt || "").trim();
      const a = String(body.a || body.answer || "").trim();
      const tag = String(body.tag || "learned").trim().slice(0, 40);
      if (!q || !a) return send(res, 400, { ok: false, error: "q and a required" });
      if (a.length > 6000) return send(res, 400, { ok: false, error: "answer too long" });

      // replace near-duplicate
      const qKey = tokenize(q).join(" ");
      samples = samples.filter(
        (s) => !(s.source === "learned" && tokenize(s.q).join(" ") === qKey)
      );
      const sample = {
        id: uid(),
        q: q.slice(0, 240),
        a,
        tag,
        at: Date.now(),
        source: "learned",
        hits: 1,
      };
      samples.push(sample);
      persistLearned();
      harvestWords(q, a);
      rebuildBrain();
      return send(res, 200, { ok: true, sample, total: samples.length, brain: markovSize() });
    }

    if (pathname === "/api/dict" && req.method === "GET") {
      const word = url.searchParams.get("word") || "";
      const lang = (url.searchParams.get("lang") || "").toLowerCase();
      const hit = await dictLookup(word, lang === "es");
      if (!hit) return send(res, 404, { ok: false, error: "not found" });
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
      const sample = { id: uid(), q, a, tag, at: Date.now(), source: "learned", hits: 0 };
      samples.push(sample);
      persistLearned();
      harvestWords(q, a);
      rebuildBrain();
      return send(res, 200, { ok: true, sample, total: samples.length, brain: markovSize() });
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
          hits: 0,
        });
        added++;
      }
      persistLearned();
      return send(res, 200, { ok: true, added, total: samples.length });
    }

    if (pathname === "/api/train/reload" && req.method === "POST") {
      const body = await readBody(req);
      if (!isAdmin(req, body)) return send(res, 401, { ok: false, error: "admin pin required" });
      loadAll();
      return send(res, 200, { ok: true, samples: samples.length, words: wordSet.size });
    }

    // words: list / add
    if (pathname === "/api/words" && req.method === "GET") {
      const q = (url.searchParams.get("q") || "").toLowerCase();
      let list = Array.from(wordSet);
      if (q) list = list.filter((w) => w.includes(q)).slice(0, 100);
      else list = list.slice(0, 50);
      return send(res, 200, { ok: true, count: wordSet.size, words: list });
    }

    if (pathname === "/api/words" && req.method === "POST") {
      const body = await readBody(req);
      // open add for single words; bulk needs admin
      const single = body.word ? [String(body.word)] : [];
      const bulk = Array.isArray(body.words) ? body.words.map(String) : [];
      const list = [...single, ...bulk].map((w) => w.toLowerCase().trim()).filter((w) => w.length >= 2 && w.length <= 32);
      if (!list.length) return send(res, 400, { ok: false, error: "word or words[] required" });
      if (list.length > 5 && !isAdmin(req, body)) {
        return send(res, 401, { ok: false, error: "admin pin required for bulk add" });
      }
      let added = 0;
      for (const w of list) {
        if (!wordSet.has(w)) {
          wordSet.add(w);
          added++;
        }
      }
      if (added) persistWords();
      return send(res, 200, { ok: true, added, total: wordSet.size });
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
  console.log(`[shiv-ai] learn   POST /api/learn`);
  console.log(`[shiv-ai] words   GET/POST /api/words`);
  console.log(`[shiv-ai] dict    GET  /api/dict?word=`);
  console.log(`[shiv-ai] train   POST /api/train  (header X-Admin-Pin)`);
});
