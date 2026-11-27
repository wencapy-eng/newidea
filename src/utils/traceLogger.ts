export type LogLevel = 'call' | 'stage' | 'metric' | 'info' | 'warn' | 'error';

export interface LogEntry {
  id: string;
  seq: number;
  sessionId: string;
  timestamp: string;
  relMs: number;
  level: LogLevel;
  module: string;
  fn: string;
  message: string;
  parentId?: string;
  depth: number;
  durationMs?: number;
  input?: unknown;
  output?: unknown;
  error?: string;
}

export interface LogSessionSummary {
  algoVersion?: string;
  robotId?: string;
  mode?: string;
  initMode?: string;
  finalTransMm?: number;
  finalRotDeg?: number;
  success?: boolean;
  elapsedMs?: number;
  ablationRowCount?: number;
  [key: string]: unknown;
}

export interface LogSession {
  id: string;
  title: string;
  kind: 'lifecycle' | 'single' | 'ablation' | 'imported';
  startedAt: string;
  endedAt?: string;
  algoVersion?: string;
  robotId?: string;
  configSnapshot?: unknown;
  summary?: LogSessionSummary;
  entries: LogEntry[];
  sourceLabel?: string;
}

export interface FunctionProfileRow {
  key: string;
  module: string;
  fn: string;
  calls: number;
  errors: number;
  totalMs: number;
  avgMs: number;
  maxMs: number;
}

export interface LogFilterOptions {
  level?: LogLevel | 'all';
  module?: string | 'all';
  query?: string;
}

const SENSITIVE_KEY_RE = /(token|secret|password|passwd|authorization|apikey|api_key|cookie|credential)/i;
const MAX_ENTRIES_PER_SESSION = 2500;
const MAX_STORED_SESSIONS = 8;
const STORAGE_KEY = 'hand_eye_trace_sessions_v1';

type TypedArray =
  | Float32Array
  | Float64Array
  | Uint8Array
  | Uint8ClampedArray
  | Int32Array
  | Uint16Array
  | Int16Array;

function isTypedArray(val: unknown): val is TypedArray {
  return ArrayBuffer.isView(val) && !(val instanceof DataView);
}

function roundNum(n: number): number {
  if (!Number.isFinite(n)) return n;
  const abs = Math.abs(n);
  if (abs === 0) return 0;
  if (abs >= 1000) return Number(n.toFixed(3));
  if (abs >= 1) return Number(n.toFixed(5));
  return Number(n.toPrecision(5));
}

function summarizeTypedArray(arr: TypedArray): Record<string, unknown> {
  const len = arr.length;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let finiteCount = 0;
  let nonZero = 0;
  const stride = len > 4096 ? Math.floor(len / 4096) : 1;
  for (let i = 0; i < len; i += stride) {
    const v = arr[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      finiteCount++;
      if (v !== 0) nonZero++;
    }
  }
  const sampleCount = Math.min(6, len);
  const sample: number[] = [];
  for (let i = 0; i < sampleCount; i++) {
    sample.push(roundNum(arr[i]));
  }
  return {
    __typedArray: arr.constructor.name,
    length: len,
    sampledStride: stride,
    nonZeroSampled: nonZero,
    min: finiteCount > 0 ? roundNum(min) : null,
    max: finiteCount > 0 ? roundNum(max) : null,
    mean: finiteCount > 0 ? roundNum(sum / finiteCount) : null,
    preview: sample,
  };
}

/**
 * Deterministic, bounded serializer for arbitrary runtime inputs & outputs.
 * Preserves exact small vectors/matrices/configs while summarizing large typed arrays
 * and redacting any credential-like keys.
 */
export function summarizeForLog(
  value: unknown,
  depth = 0,
  seen: WeakSet<object> = new WeakSet()
): unknown {
  if (value === null || value === undefined) return value;
  if (typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return 'NaN';
    if (!Number.isFinite(value)) return value > 0 ? 'Infinity' : '-Infinity';
    return roundNum(value);
  }
  if (typeof value === 'string') {
    return value.length > 600 ? `${value.slice(0, 600)}… (${value.length} chars)` : value;
  }
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'function') {
    const fnName = (value as { name?: string }).name || 'anonymous';
    return `[Function ${fnName}]`;
  }
  if (typeof value === 'symbol') return value.toString();

  if (value instanceof Date) return value.toISOString();
  if (value instanceof Error) {
    return {
      name: value.name,
      message: value.message,
      stack: value.stack ? value.stack.split('\n').slice(0, 4).join('\n') : undefined,
    };
  }
  if (isTypedArray(value)) {
    return summarizeTypedArray(value);
  }

  if (typeof value === 'object') {
    if (seen.has(value)) return '[Circular]';
    if (depth >= 5) {
      if (Array.isArray(value)) return `[Array(${value.length})]`;
      return `[Object ${value.constructor?.name || 'Object'}]`;
    }
    seen.add(value);

    if (value instanceof Set) {
      const items = Array.from(value).slice(0, 16).map((v) => summarizeForLog(v, depth + 1, seen));
      return { __set: value.size, items };
    }
    if (value instanceof Map) {
      const entries: Record<string, unknown> = {};
      let count = 0;
      for (const [k, v] of value.entries()) {
        if (count >= 16) break;
        const keyStr = String(k);
        entries[keyStr] = SENSITIVE_KEY_RE.test(keyStr)
          ? '[REDACTED]'
          : summarizeForLog(v, depth + 1, seen);
        count++;
      }
      return { __map: value.size, entries };
    }

    if (Array.isArray(value)) {
      const len = value.length;
      if (len <= 20) {
        return value.map((item) => summarizeForLog(item, depth + 1, seen));
      }
      const head = value.slice(0, 8).map((item) => summarizeForLog(item, depth + 1, seen));
      const tail = value.slice(len - 3).map((item) => summarizeForLog(item, depth + 1, seen));
      return {
        __array: len,
        head,
        tail,
      };
    }

    const out: Record<string, unknown> = {};
    const keys = Object.keys(value as Record<string, unknown>);
    const limit = Math.min(keys.length, 32);
    for (let i = 0; i < limit; i++) {
      const k = keys[i];
      if (SENSITIVE_KEY_RE.test(k)) {
        out[k] = '[REDACTED]';
      } else {
        const propVal = (value as Record<string, unknown>)[k];
        if (typeof propVal === 'function') continue;
        out[k] = summarizeForLog(propVal, depth + 1, seen);
      }
    }
    if (keys.length > limit) {
      out.__truncatedKeys = keys.length - limit;
    }
    return out;
  }

  return String(value);
}

/** Filter log entries by level, module, and case-insensitive substring query */
export function filterLogEntries(
  entries: readonly LogEntry[],
  opts: LogFilterOptions
): LogEntry[] {
  const level = opts.level ?? 'all';
  const mod = opts.module ?? 'all';
  const q = (opts.query ?? '').trim().toLowerCase();

  return entries.filter((entry) => {
    if (level !== 'all' && entry.level !== level) return false;
    if (mod !== 'all' && entry.module !== mod) return false;
    if (!q) return true;

    if (entry.fn.toLowerCase().includes(q)) return true;
    if (entry.module.toLowerCase().includes(q)) return true;
    if (entry.message.toLowerCase().includes(q)) return true;
    if (entry.error && entry.error.toLowerCase().includes(q)) return true;
    if (entry.input !== undefined) {
      try {
        if (JSON.stringify(entry.input).toLowerCase().includes(q)) return true;
      } catch {
        // ignore serialization error
      }
    }
    if (entry.output !== undefined) {
      try {
        if (JSON.stringify(entry.output).toLowerCase().includes(q)) return true;
      } catch {
        // ignore serialization error
      }
    }
    return false;
  });
}

/** Aggregate per-function invocation count, total/average/max latency, and errors */
export function computeFunctionProfile(entries: readonly LogEntry[]): FunctionProfileRow[] {
  const map = new Map<string, FunctionProfileRow>();
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    if (e.level !== 'call' && e.durationMs === undefined) continue;
    const key = `${e.module}.${e.fn}`;
    let row = map.get(key);
    if (!row) {
      row = {
        key,
        module: e.module,
        fn: e.fn,
        calls: 0,
        errors: 0,
        totalMs: 0,
        avgMs: 0,
        maxMs: 0,
      };
      map.set(key, row);
    }
    const dur = typeof e.durationMs === 'number' && Number.isFinite(e.durationMs) ? e.durationMs : 0;
    row.calls += 1;
    if (e.error || e.level === 'error') row.errors += 1;
    row.totalMs += dur;
    if (dur > row.maxMs) row.maxMs = dur;
  }

  const rows = Array.from(map.values());
  for (const r of rows) {
    r.avgMs = r.calls > 0 ? r.totalMs / r.calls : 0;
  }
  rows.sort((a, b) => b.totalMs - a.totalMs);
  return rows;
}

/** Export a full LogSession as structured pretty JSON suitable for human & programmatic analysis */
export function exportSessionJson(session: LogSession): string {
  const profile = computeFunctionProfile(session.entries);
  return JSON.stringify(
    {
      schema: 'hand-eye-calib-trace/v1',
      exportedAt: new Date().toISOString(),
      session: {
        id: session.id,
        title: session.title,
        kind: session.kind,
        startedAt: session.startedAt,
        endedAt: session.endedAt,
        algoVersion: session.algoVersion,
        robotId: session.robotId,
        configSnapshot: session.configSnapshot,
        summary: session.summary,
        entryCount: session.entries.length,
      },
      functionProfile: profile,
      entries: session.entries,
    },
    null,
    2
  );
}

/** Export a LogSession as NDJSON (.jsonl): first line is session header, remaining lines are LogEntry records */
export function exportSessionJsonl(session: LogSession): string {
  const header = JSON.stringify({
    __recordType: 'session_header',
    schema: 'hand-eye-calib-trace/v1',
    id: session.id,
    title: session.title,
    kind: session.kind,
    startedAt: session.startedAt,
    endedAt: session.endedAt,
    algoVersion: session.algoVersion,
    robotId: session.robotId,
    configSnapshot: session.configSnapshot,
    summary: session.summary,
  });
  const lines = [header];
  for (let i = 0; i < session.entries.length; i++) {
    lines.push(JSON.stringify(session.entries[i]));
  }
  return lines.join('\n');
}

function normalizeImportedEntry(raw: unknown, sessionId: string, fallbackSeq: number): LogEntry | null {
  if (!raw || typeof raw !== 'object') return null;
  const obj = raw as Record<string, unknown>;
  const validLevels: ReadonlySet<string> = new Set(['call', 'stage', 'metric', 'info', 'warn', 'error']);
  const level: LogLevel =
    typeof obj.level === 'string' && validLevels.has(obj.level) ? (obj.level as LogLevel) : 'info';
  return {
    id: typeof obj.id === 'string' ? obj.id : `${sessionId}_e${fallbackSeq}`,
    seq: typeof obj.seq === 'number' ? obj.seq : fallbackSeq,
    sessionId,
    timestamp: typeof obj.timestamp === 'string' ? obj.timestamp : new Date().toISOString(),
    relMs: typeof obj.relMs === 'number' && Number.isFinite(obj.relMs) ? obj.relMs : 0,
    level,
    module: typeof obj.module === 'string' ? obj.module : 'unknown',
    fn: typeof obj.fn === 'string' ? obj.fn : 'event',
    message: typeof obj.message === 'string' ? obj.message : '',
    parentId: typeof obj.parentId === 'string' ? obj.parentId : undefined,
    depth: typeof obj.depth === 'number' && Number.isFinite(obj.depth) ? Math.max(0, Math.floor(obj.depth)) : 0,
    durationMs: typeof obj.durationMs === 'number' && Number.isFinite(obj.durationMs) ? obj.durationMs : undefined,
    input: obj.input,
    output: obj.output,
    error: typeof obj.error === 'string' ? obj.error : undefined,
  };
}

/**
 * Parse an imported `.json` or `.jsonl` log file (from local disk or fetched from GitHub)
 * into a clean `LogSession` ready for inspection inside the app.
 */
export function parseImportedLogContent(rawText: string, sourceLabel: string): LogSession {
  const trimmed = rawText.trim();
  if (!trimmed) {
    throw new Error('日志文件内容为空');
  }

  const importId = `imp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;

  // 1. Try single JSON document first
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (Array.isArray(parsed)) {
        const entries = parsed
          .map((item, i) => normalizeImportedEntry(item, importId, i + 1))
          .filter((e): e is LogEntry => e !== null);
        return {
          id: importId,
          title: `导入日志: ${sourceLabel}`,
          kind: 'imported',
          startedAt: entries[0]?.timestamp ?? new Date().toISOString(),
          endedAt: entries[entries.length - 1]?.timestamp,
          entries,
          sourceLabel,
        };
      }
      if (parsed && typeof parsed === 'object') {
        const doc = parsed as Record<string, unknown>;
        const meta =
          doc.session && typeof doc.session === 'object'
            ? (doc.session as Record<string, unknown>)
            : doc;
        const rawEntries = Array.isArray(doc.entries)
          ? doc.entries
          : Array.isArray(meta.entries)
            ? meta.entries
            : null;
        if (rawEntries) {
          const entries = rawEntries
            .map((item, i) => normalizeImportedEntry(item, importId, i + 1))
            .filter((e): e is LogEntry => e !== null);
          return {
            id: importId,
            title:
              typeof meta.title === 'string' && meta.title.trim()
                ? `${meta.title} (${sourceLabel})`
                : `导入日志: ${sourceLabel}`,
            kind: 'imported',
            startedAt:
              typeof meta.startedAt === 'string'
                ? meta.startedAt
                : entries[0]?.timestamp ?? new Date().toISOString(),
            endedAt: typeof meta.endedAt === 'string' ? meta.endedAt : undefined,
            algoVersion: typeof meta.algoVersion === 'string' ? meta.algoVersion : undefined,
            robotId: typeof meta.robotId === 'string' ? meta.robotId : undefined,
            configSnapshot: meta.configSnapshot,
            summary:
              meta.summary && typeof meta.summary === 'object'
                ? (meta.summary as LogSessionSummary)
                : undefined,
            entries,
            sourceLabel,
          };
        }
      }
    } catch {
      // Fall through to JSONL parser if multi-line NDJSON
    }
  }

  // 2. Parse as JSONL (NDJSON)
  const lines = trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error('未在日志文件中解析到有效记录');
  }

  let headerMeta: Record<string, unknown> | null = null;
  const entries: LogEntry[] = [];
  for (let i = 0; i < lines.length; i++) {
    let parsedLine: unknown;
    try {
      parsedLine = JSON.parse(lines[i]);
    } catch {
      throw new Error(`第 ${i + 1} 行 JSON 解析失败，请确认文件为有效的 .json 或 .jsonl 格式`);
    }
    if (
      parsedLine &&
      typeof parsedLine === 'object' &&
      (parsedLine as Record<string, unknown>).__recordType === 'session_header'
    ) {
      headerMeta = parsedLine as Record<string, unknown>;
      continue;
    }
    const entry = normalizeImportedEntry(parsedLine, importId, entries.length + 1);
    if (entry) entries.push(entry);
  }

  return {
    id: importId,
    title:
      headerMeta && typeof headerMeta.title === 'string'
        ? `${headerMeta.title} (${sourceLabel})`
        : `导入 JSONL: ${sourceLabel}`,
    kind: 'imported',
    startedAt:
      headerMeta && typeof headerMeta.startedAt === 'string'
        ? headerMeta.startedAt
        : entries[0]?.timestamp ?? new Date().toISOString(),
    endedAt: headerMeta && typeof headerMeta.endedAt === 'string' ? headerMeta.endedAt : undefined,
    algoVersion:
      headerMeta && typeof headerMeta.algoVersion === 'string' ? headerMeta.algoVersion : undefined,
    robotId: headerMeta && typeof headerMeta.robotId === 'string' ? headerMeta.robotId : undefined,
    configSnapshot: headerMeta?.configSnapshot,
    summary:
      headerMeta?.summary && typeof headerMeta.summary === 'object'
        ? (headerMeta.summary as LogSessionSummary)
        : undefined,
    entries,
    sourceLabel,
  };
}

export class TraceLogger {
  private sessions: LogSession[] = [];
  private activeSessionId: string | null = null;
  private activeSessionStartPerf = 0;
  private callStack: string[] = [];
  private seqCounter = 0;
  private listeners = new Set<() => void>();
  private notifyTimer: ReturnType<typeof setTimeout> | null = null;
  private persistEnabled: boolean;

  constructor(opts?: { persist?: boolean }) {
    this.persistEnabled = opts?.persist ?? (typeof window !== 'undefined' && typeof localStorage !== 'undefined');
    if (this.persistEnabled) {
      this.loadFromStorage();
    }
  }

  private nowMs(): number {
    return typeof performance !== 'undefined' ? performance.now() : Date.now();
  }

  private scheduleNotify(): void {
    if (this.listeners.size === 0) return;
    if (this.notifyTimer !== null) return;
    this.notifyTimer = setTimeout(() => {
      this.notifyTimer = null;
      for (const fn of this.listeners) {
        try {
          fn();
        } catch {
          // isolate subscriber failures
        }
      }
    }, 60);
  }

  private notifyImmediate(): void {
    if (this.notifyTimer !== null) {
      clearTimeout(this.notifyTimer);
      this.notifyTimer = null;
    }
    for (const fn of this.listeners) {
      try {
        fn();
      } catch {
        // isolate subscriber failures
      }
    }
  }

  private saveToStorage(): void {
    if (!this.persistEnabled || typeof localStorage === 'undefined') return;
    try {
      // Persist up to 5 latest sessions with capped entry count to fit cleanly within localStorage quota
      const compact = this.sessions.slice(-5).map((s) => ({
        ...s,
        entries: s.entries.slice(-500),
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
    } catch {
      // Ignore quota exceeded gracefully
    }
  }

  private loadFromStorage(): void {
    if (typeof localStorage === 'undefined') return;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        this.sessions = parsed.filter(
          (s): s is LogSession =>
            Boolean(s) && typeof s === 'object' && typeof (s as LogSession).id === 'string' && Array.isArray((s as LogSession).entries)
        );
      }
    } catch {
      // Ignore corrupt localStorage gracefully
    }
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private ensureActiveSession(): LogSession {
    if (this.activeSessionId) {
      const found = this.sessions.find((s) => s.id === this.activeSessionId);
      if (found) return found;
    }
    return this.startSession({
      title: '系统初始化与交互日志',
      kind: 'lifecycle',
    });
  }

  startSession(meta: {
    title: string;
    kind: LogSession['kind'];
    algoVersion?: string;
    robotId?: string;
    configSnapshot?: unknown;
  }): LogSession {
    const id = `ses_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const session: LogSession = {
      id,
      title: meta.title,
      kind: meta.kind,
      startedAt: new Date().toISOString(),
      algoVersion: meta.algoVersion,
      robotId: meta.robotId,
      configSnapshot: meta.configSnapshot !== undefined ? summarizeForLog(meta.configSnapshot) : undefined,
      entries: [],
    };
    this.sessions.push(session);
    if (this.sessions.length > MAX_STORED_SESSIONS) {
      this.sessions.splice(0, this.sessions.length - MAX_STORED_SESSIONS);
    }
    this.activeSessionId = id;
    this.activeSessionStartPerf = this.nowMs();
    this.callStack = [];
    this.seqCounter = 0;
    this.saveToStorage();
    this.notifyImmediate();
    return session;
  }

  endSession(summary?: LogSessionSummary): void {
    const session = this.sessions.find((s) => s.id === this.activeSessionId);
    if (!session) return;
    session.endedAt = new Date().toISOString();
    if (summary) {
      session.summary = { ...(session.summary ?? {}), ...(summarizeForLog(summary) as LogSessionSummary) };
    }
    this.callStack = [];
    this.saveToStorage();
    this.notifyImmediate();
  }

  importSession(session: LogSession): LogSession {
    this.sessions.push(session);
    if (this.sessions.length > MAX_STORED_SESSIONS) {
      this.sessions.splice(0, this.sessions.length - MAX_STORED_SESSIONS);
    }
    this.saveToStorage();
    this.notifyImmediate();
    return session;
  }

  clearSessions(): void {
    this.sessions = [];
    this.activeSessionId = null;
    this.callStack = [];
    this.seqCounter = 0;
    if (this.persistEnabled && typeof localStorage !== 'undefined') {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore
      }
    }
    this.notifyImmediate();
  }

  getSessions(): LogSession[] {
    return this.sessions.map((s) => ({
      ...s,
      entries: [...s.entries],
    }));
  }

  getActiveSession(): LogSession | null {
    if (!this.activeSessionId) return this.sessions[this.sessions.length - 1] ?? null;
    return this.sessions.find((s) => s.id === this.activeSessionId) ?? null;
  }

  getTotalEntryCount(): number {
    let total = 0;
    for (let i = 0; i < this.sessions.length; i++) {
      total += this.sessions[i].entries.length;
    }
    return total;
  }

  private pushEntry(entry: LogEntry): void {
    const session = this.ensureActiveSession();
    session.entries.push(entry);
    if (session.entries.length > MAX_ENTRIES_PER_SESSION) {
      session.entries.splice(0, session.entries.length - MAX_ENTRIES_PER_SESSION);
    }
    this.scheduleNotify();
  }

  log(
    level: LogLevel,
    module: string,
    fn: string,
    message: string,
    payload?: { input?: unknown; output?: unknown; durationMs?: number; error?: string }
  ): LogEntry {
    const session = this.ensureActiveSession();
    this.seqCounter += 1;
    const parentId = this.callStack.length > 0 ? this.callStack[this.callStack.length - 1] : undefined;
    const entry: LogEntry = {
      id: `${session.id}_${this.seqCounter}`,
      seq: this.seqCounter,
      sessionId: session.id,
      timestamp: new Date().toISOString(),
      relMs: roundNum(Math.max(0, this.nowMs() - this.activeSessionStartPerf)),
      level,
      module,
      fn,
      message,
      parentId,
      depth: this.callStack.length,
      durationMs: payload?.durationMs !== undefined ? roundNum(payload.durationMs) : undefined,
      input: payload?.input !== undefined ? summarizeForLog(payload.input) : undefined,
      output: payload?.output !== undefined ? summarizeForLog(payload.output) : undefined,
      error: payload?.error,
    };
    this.pushEntry(entry);
    return entry;
  }

  /** Record a completed call directly (useful for generator iterations or manual timing) */
  recordCall(
    module: string,
    fn: string,
    input: unknown,
    output: unknown,
    durationMs: number,
    message?: string
  ): LogEntry {
    return this.log('call', module, fn, message ?? `${module}.${fn}()`, {
      input,
      output,
      durationMs,
    });
  }

  /** Wrap a synchronous function call, capturing hierarchical depth, inputs, outputs, latency, and errors */
  traceCall<T>(
    module: string,
    fn: string,
    input: unknown,
    body: () => T,
    formatOutput?: (res: T) => unknown,
    message?: string
  ): T {
    const session = this.ensureActiveSession();
    this.seqCounter += 1;
    const id = `${session.id}_${this.seqCounter}`;
    const parentId = this.callStack.length > 0 ? this.callStack[this.callStack.length - 1] : undefined;
    const depth = this.callStack.length;
    const tStart = this.nowMs();
    const entry: LogEntry = {
      id,
      seq: this.seqCounter,
      sessionId: session.id,
      timestamp: new Date().toISOString(),
      relMs: roundNum(Math.max(0, tStart - this.activeSessionStartPerf)),
      level: 'call',
      module,
      fn,
      message: message ?? `${module}.${fn}()`,
      parentId,
      depth,
      input: summarizeForLog(input),
    };
    this.pushEntry(entry);
    this.callStack.push(id);
    try {
      const res = body();
      entry.durationMs = roundNum(Math.max(0, this.nowMs() - tStart));
      entry.output = summarizeForLog(formatOutput ? formatOutput(res) : res);
      this.scheduleNotify();
      return res;
    } catch (err) {
      entry.durationMs = roundNum(Math.max(0, this.nowMs() - tStart));
      entry.level = 'error';
      entry.error = err instanceof Error ? err.message : String(err);
      this.scheduleNotify();
      throw err;
    } finally {
      this.callStack.pop();
    }
  }

  /** Wrap an async function call, capturing hierarchical depth, inputs, outputs, latency, and errors */
  async traceAsync<T>(
    module: string,
    fn: string,
    input: unknown,
    body: () => Promise<T>,
    formatOutput?: (res: T) => unknown,
    message?: string
  ): Promise<T> {
    const session = this.ensureActiveSession();
    this.seqCounter += 1;
    const id = `${session.id}_${this.seqCounter}`;
    const parentId = this.callStack.length > 0 ? this.callStack[this.callStack.length - 1] : undefined;
    const depth = this.callStack.length;
    const tStart = this.nowMs();
    const entry: LogEntry = {
      id,
      seq: this.seqCounter,
      sessionId: session.id,
      timestamp: new Date().toISOString(),
      relMs: roundNum(Math.max(0, tStart - this.activeSessionStartPerf)),
      level: 'call',
      module,
      fn,
      message: message ?? `${module}.${fn}()`,
      parentId,
      depth,
      input: summarizeForLog(input),
    };
    this.pushEntry(entry);
    this.callStack.push(id);
    try {
      const res = await body();
      entry.durationMs = roundNum(Math.max(0, this.nowMs() - tStart));
      entry.output = summarizeForLog(formatOutput ? formatOutput(res) : res);
      this.scheduleNotify();
      return res;
    } catch (err) {
      entry.durationMs = roundNum(Math.max(0, this.nowMs() - tStart));
      entry.level = 'error';
      entry.error = err instanceof Error ? err.message : String(err);
      this.scheduleNotify();
      throw err;
    } finally {
      this.callStack.pop();
    }
  }
}

export const traceLogger = new TraceLogger();
