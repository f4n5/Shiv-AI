/**
 * Shiv AI backend — 100% free, local Node (no paid APIs)
 * Loads seed JSON training sets + accepts new samples from admin
 * Serves chat completions via keyword/tag scoring (+ optional word boost)
 *
 * Run:  node server.js
 * Env:  PORT=8787  ADMIN_PIN=yeahimtheownerhereicreateKEYS
 */

const http = require("http");
const fs = require("fs");
const path = require("path");
const { URL } = require("url");

const PORT = Number(process.env.PORT || 8787);
const ADMIN_PIN = process.env.ADMIN_PIN || "yeahimtheownerhereicreateKEYS";
const DATA_DIR = path.join(__dirname, "data");
const LEARNED_PATH = path.join(DATA_DIR, "learned_samples.json");
const PUBLIC_DIR = path.join(__dirname, ".."); // docs/ (for static shiv-ai.html)

const SEED_FILES = [
  "basic_responses.json",
  "code_basics.json",
  "emotions_basics.json",
];

/** @type {Array<{id:string,q:string,a:string,tag?:string,at?:number,source?:string}>} */
let samples = [];
/** @type {Set<string>} */
let wordSet = new Set();

function readJson(file, fallback) {
  try {
    const raw = fs.readFileSync(file, "utf8");
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2), "utf8");
}

function loadAll() {
  samples = [];
  for (const name of SEED_FILES) {
    const arr = readJson(path.join(DATA_DIR, name), []);
    if (Array.isArray(arr)) {
      for (const s of arr) {
        if (s && s.q && s.a) {
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
    }
  }
  const learned = readJson(LEARNED_PATH, []);
  if (Array.isArray(learned)) {
    for (const s of learned) {
      if (s && s.q && s.a) {
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
  }
  const wordsDoc = readJson(path.join(DATA_DIR, "basic_words.json"), { words: [] });
  wordSet = new Set((wordsDoc.words || []).map((w) => String(w).toLowerCase()));
  console.log(`[shiv-ai] loaded ${samples.length} samples, ${wordSet.size} words`);
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

function tokenize(s) {
  return String(s)
    .toLowerCase()
    .replace(/[^a-z0-9_+\-#.]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function scoreSample(promptTokens, sample) {
  const qTokens = tokenize(sample.q + " " + (sample.tag || ""));
  const set = new Set(qTokens);
  let score = 0;
  for (const t of promptTokens) {
    if (set.has(t)) score += 2;
    else if (qTokens.some((q) => q.includes(t) || t.includes(q))) score += 1;
    if (wordSet.has(t)) score += 0.15; // mild vocabulary presence boost
  }
  if (sample.tag) {
    const tag = sample.tag.toLowerCase();
    if (promptTokens.includes(tag)) score += 3;
  }
  // prefer longer answers slightly when tied-ish
  score += Math.min(1, (sample.a || "").length / 2000);
  return score;
}

function generateReply(prompt, fileContext) {
  const tokens = tokenize(prompt);
  let best = null;
  let bestScore = 0;
  for (const s of samples) {
    const sc = scoreSample(tokens, s);
    if (sc > bestScore) {
      bestScore = sc;
      best = s;
    }
  }
  if (best && bestScore >= 1.5) {
    let out = best.a;
    if (fileContext) out += "\n\n(from your project)\n" + fileContext;
    out += `\n\n_matched · score ${bestScore.toFixed(1)} · tag ${best.tag || "—"} · source ${best.source || "seed"}_`;
    return { ok: true, text: out, score: bestScore, sampleId: best.id };
  }
  let out =
    "No strong training match yet.\n\n" +
    "Add a Q→A under Train AI (admin) for this prompt, or ask with more keywords from your code domain.";
  if (fileContext) out += "\n\n" + fileContext;
  return { ok: true, text: out, score: bestScore, sampleId: null };
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
    // —— API ——
    if (pathname === "/api/health" && req.method === "GET") {
      return send(res, 200, {
        ok: true,
        samples: samples.length,
        words: wordSet.size,
        name: "shiv-ai-backend",
        free: true,
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
      const result = generateReply(prompt, fileContext);
      return send(res, 200, result);
    }

    if (pathname === "/api/samples" && req.method === "GET") {
      // public can see count; full list only with admin
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

    // —— static (optional): serve docs/ ——
    let rel = pathname === "/" ? "/shiv-ai.html" : pathname;
    // also allow /shiv-ai-data/*
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
server.listen(PORT, () => {
  console.log(`[shiv-ai] http://127.0.0.1:${PORT}`);
  console.log(`[shiv-ai] health  GET  /api/health`);
  console.log(`[shiv-ai] chat    POST /api/chat`);
  console.log(`[shiv-ai] train   POST /api/train  (header X-Admin-Pin)`);
});
