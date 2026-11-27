import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BarChart3,
  Check,
  Copy,
  Download,
  ExternalLink,
  FileUp,
  FolderGit2,
  GitBranch,
  Link2,
  LogOut,
  RefreshCw,
  Search,
  Terminal,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react';
import {
  buildDefaultLogPath,
  isSafeGitHubUrl,
  validateGitHubPushTarget,
} from '../utils/githubValidation';
import {
  computeFunctionProfile,
  exportSessionJson,
  exportSessionJsonl,
  filterLogEntries,
  parseImportedLogContent,
  traceLogger,
  type LogEntry,
  type LogLevel,
  type LogSession,
} from '../utils/traceLogger';

interface LogViewerModalProps {
  open: boolean;
  onClose: () => void;
  sessions: readonly LogSession[];
  activeSessionId: string | null;
}

interface GitHubStatusResponse {
  oauthConfigured: boolean;
  serverTokenConfigured: boolean;
  connected: boolean;
  authSource: 'oauth' | 'server_token' | null;
  user: { login: string; name: string; avatarUrl: string } | null;
  defaultRepo: string;
  callbackUrl: string;
}

interface GitHubRepoItem {
  fullName: string;
  defaultBranch: string;
  isPrivate: boolean;
}

interface PushSuccessRecord {
  repo: string;
  branch: string;
  filePath: string;
  htmlUrl: string;
  rawUrl: string;
  commitSha: string;
  commitUrl: string;
  bytes: number;
  pushedAt: string;
}

type ViewerTab = 'trace' | 'profile' | 'github';

const LEVEL_OPTIONS: readonly (LogLevel | 'all')[] = [
  'all',
  'call',
  'stage',
  'metric',
  'info',
  'warn',
  'error',
];

function levelBadgeClass(level: LogLevel): string {
  switch (level) {
    case 'call':
      return 'bg-cyan-500/15 text-cyan-300 border-cyan-500/35';
    case 'stage':
      return 'bg-indigo-500/20 text-indigo-300 border-indigo-400/40';
    case 'metric':
      return 'bg-emerald-500/15 text-emerald-300 border-emerald-500/35';
    case 'warn':
      return 'bg-amber-500/15 text-amber-300 border-amber-500/40';
    case 'error':
      return 'bg-rose-500/20 text-rose-300 border-rose-500/45';
    default:
      return 'bg-slate-700/40 text-slate-300 border-slate-600/50';
  }
}

function formatMs(ms?: number): string {
  if (ms === undefined || !Number.isFinite(ms)) return '—';
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms >= 10) return `${ms.toFixed(1)}ms`;
  return `${ms.toFixed(2)}ms`;
}

function formatJsonBlock(val: unknown): string {
  if (val === undefined) return '— (无)';
  try {
    return JSON.stringify(val, null, 2);
  } catch {
    return String(val);
  }
}

export const LogViewerModal: React.FC<LogViewerModalProps> = ({
  open,
  onClose,
  sessions,
  activeSessionId,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [tab, setTab] = useState<ViewerTab>('trace');
  const [selectedSessionId, setSelectedSessionId] = useState<string>('');
  const [levelFilter, setLevelFilter] = useState<LogLevel | 'all'>('all');
  const [moduleFilter, setModuleFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [importError, setImportError] = useState<string>('');
  const [importFeedback, setImportFeedback] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string>('');

  // GitHub OAuth session token held strictly in React memory (never localStorage/sessionStorage)
  const [ghSessionToken, setGhSessionToken] = useState<string | null>(null);
  const [ghStatus, setGhStatus] = useState<GitHubStatusResponse | null>(null);
  const [ghStatusLoading, setGhStatusLoading] = useState<boolean>(false);
  const [ghRepos, setGhRepos] = useState<GitHubRepoItem[]>([]);
  const [ghAuthError, setGhAuthError] = useState<string>('');

  // Push form state
  const [pushFormat, setPushFormat] = useState<'jsonl' | 'json'>('jsonl');
  const [pushRepo, setPushRepo] = useState<string>('');
  const [pushBranch, setPushBranch] = useState<string>('main');
  const [pushFilePath, setPushFilePath] = useState<string>('');
  const [pushCommitMsg, setPushCommitMsg] = useState<string>('');
  const [pushLoading, setPushLoading] = useState<boolean>(false);
  const [pushError, setPushError] = useState<string>('');
  const [pushHistory, setPushHistory] = useState<PushSuccessRecord[]>([]);

  // Fetch-from-GitHub form state
  const [fetchRepo, setFetchRepo] = useState<string>('');
  const [fetchBranch, setFetchBranch] = useState<string>('main');
  const [fetchFilePath, setFetchFilePath] = useState<string>('');
  const [fetchLoading, setFetchLoading] = useState<boolean>(false);
  const [fetchError, setFetchError] = useState<string>('');

  // Keep selectedSessionId synced to newest/active session when opening
  useEffect(() => {
    if (!open) return;
    if (
      !selectedSessionId ||
      !sessions.some((s) => s.id === selectedSessionId)
    ) {
      const nextId = activeSessionId ?? sessions[sessions.length - 1]?.id ?? '';
      setSelectedSessionId(nextId);
    }
  }, [open, sessions, activeSessionId, selectedSessionId]);

  const currentSession: LogSession | null = useMemo(() => {
    if (sessions.length === 0) return null;
    return (
      sessions.find((s) => s.id === selectedSessionId) ??
      sessions[sessions.length - 1] ??
      null
    );
  }, [sessions, selectedSessionId]);

  // Update default push path when session or format changes
  useEffect(() => {
    if (!currentSession) return;
    setPushFilePath(
      buildDefaultLogPath({
        algoVersion: currentSession.algoVersion,
        robotId: currentSession.robotId,
        kind: currentSession.kind,
        format: pushFormat,
      })
    );
  }, [currentSession?.id, pushFormat]);

  // Keep commit message entry count up to date as session entries accumulate
  useEffect(() => {
    if (!currentSession) return;
    setPushCommitMsg(
      `chore(logs): upload ${currentSession.algoVersion || 'calib'} ${currentSession.robotId || 'robot'} trace (${currentSession.entries.length} entries)`
    );
  }, [currentSession?.id, currentSession?.entries.length]);

  const availableModules = useMemo(() => {
    if (!currentSession) return [];
    const set = new Set<string>();
    for (const e of currentSession.entries) set.add(e.module);
    return Array.from(set).sort();
  }, [currentSession]);

  const filteredEntries = useMemo(() => {
    if (!currentSession) return [];
    return filterLogEntries(currentSession.entries, {
      level: levelFilter,
      module: moduleFilter,
      query: searchQuery,
    });
  }, [currentSession, levelFilter, moduleFilter, searchQuery]);

  const profileRows = useMemo(() => {
    if (!currentSession) return [];
    return computeFunctionProfile(currentSession.entries);
  }, [currentSession]);

  const selectedEntry: LogEntry | null = useMemo(() => {
    if (filteredEntries.length === 0) return null;
    if (selectedEntryId) {
      const hit = filteredEntries.find((e) => e.id === selectedEntryId);
      if (hit) return hit;
    }
    return filteredEntries[filteredEntries.length - 1] ?? null;
  }, [filteredEntries, selectedEntryId]);

  const authHeaders = useCallback((): Record<string, string> => {
    const headers: Record<string, string> = {};
    if (ghSessionToken) {
      headers.Authorization = `Bearer ${ghSessionToken}`;
    }
    return headers;
  }, [ghSessionToken]);

  const refreshGitHubStatus = useCallback(async () => {
    setGhStatusLoading(true);
    setGhAuthError('');
    try {
      const resp = await fetch('/api/github/status', { headers: authHeaders() });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = (await resp.json()) as GitHubStatusResponse;
      setGhStatus(data);
      if (data.defaultRepo && !pushRepo) {
        setPushRepo(data.defaultRepo);
      }
      if (data.defaultRepo && !fetchRepo) {
        setFetchRepo(data.defaultRepo);
      }
      if (data.connected) {
        const repoResp = await fetch('/api/github/repos', { headers: authHeaders() });
        if (repoResp.ok) {
          const rData = (await repoResp.json()) as { repos?: GitHubRepoItem[] };
          const list = rData.repos ?? [];
          setGhRepos(list);
          if (!pushRepo && list.length > 0) {
            setPushRepo(list[0].fullName);
            setPushBranch(list[0].defaultBranch || 'main');
          }
        }
      } else {
        setGhRepos([]);
      }
    } catch (err) {
      setGhAuthError(
        `无法获取 GitHub 后端状态: ${err instanceof Error ? err.message : String(err)}`
      );
    } finally {
      setGhStatusLoading(false);
    }
  }, [authHeaders, pushRepo, fetchRepo]);

  // Query GitHub backend status whenever modal opens or OAuth token updates
  useEffect(() => {
    if (open) {
      refreshGitHubStatus();
    }
  }, [open, ghSessionToken, refreshGitHubStatus]);

  // Listen for OAuth popup postMessage completion
  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      const origin = event.origin;
      const isTrusted =
        origin === window.location.origin ||
        origin.endsWith('.run.app') ||
        origin.includes('localhost');
      if (!isTrusted) return;
      const data = event.data as Record<string, unknown> | null;
      if (!data || typeof data !== 'object') return;
      if (data.type === 'GITHUB_OAUTH_SUCCESS' && typeof data.sessionToken === 'string') {
        setGhSessionToken(data.sessionToken);
        setGhAuthError('');
      } else if (data.type === 'GITHUB_OAUTH_ERROR' && typeof data.error === 'string') {
        setGhAuthError(data.error);
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const copyText = async (text: string, tag: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(tag);
      window.setTimeout(() => setCopiedId(''), 1800);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopiedId(tag);
      window.setTimeout(() => setCopiedId(''), 1800);
    }
  };

  const handleDownload = (format: 'json' | 'jsonl') => {
    if (!currentSession) return;
    const content =
      format === 'jsonl'
        ? exportSessionJsonl(currentSession)
        : exportSessionJson(currentSession);
    const filename = buildDefaultLogPath({
      algoVersion: currentSession.algoVersion,
      robotId: currentSession.robotId,
      kind: currentSession.kind,
      format,
    })
      .split('/')
      .pop()!;
    const blob = new Blob([content], {
      type: format === 'jsonl' ? 'application/x-ndjson;charset=utf-8' : 'application/json;charset=utf-8',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLocalFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    setImportError('');
    setImportFeedback('');
    try {
      const text = await file.text();
      const imported = parseImportedLogContent(text, file.name);
      traceLogger.importSession(imported);
      setSelectedSessionId(imported.id);
      setTab('trace');
      setImportFeedback(`已打开 ${file.name} (${imported.entries.length} 条记录)`);
      window.setTimeout(() => setImportFeedback(''), 3500);
    } catch (err) {
      setImportError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleConnectGitHub = async () => {
    setGhAuthError('');
    try {
      const resp = await fetch('/api/github/auth/url');
      const data = (await resp.json()) as { url?: string; error?: string };
      if (!resp.ok || !data.url) {
        setGhAuthError(data.error || '未能生成 GitHub 授权地址。');
        return;
      }
      // Open GitHub OAuth provider URL directly in popup (never container route)
      const popup = window.open(data.url, 'github_oauth_popup', 'width=620,height=720');
      if (!popup) {
        setGhAuthError('浏览器拦截了授权弹窗，请允许本页面弹出窗口后重试。');
      }
    } catch (err) {
      setGhAuthError(
        `请求 GitHub OAuth 失败: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  };

  const handleDisconnectGitHub = async () => {
    try {
      await fetch('/api/github/disconnect', {
        method: 'POST',
        headers: authHeaders(),
      });
    } catch {
      // ignore network error on logout
    }
    setGhSessionToken(null);
  };

  const handlePushToGitHub = async (e: React.FormEvent) => {
    e.preventDefault();
    setPushError('');
    if (!currentSession || currentSession.entries.length === 0) {
      setPushError('当前会话暂无可推送的日志记录，请先运行一次标定或导入日志。');
      return;
    }

    const validation = validateGitHubPushTarget({
      repoFullName: pushRepo,
      branch: pushBranch,
      filePath: pushFilePath,
    });
    if (!validation.ok) {
      setPushError(validation.error || '仓库或文件路径校验未通过。');
      return;
    }

    setPushLoading(true);
    try {
      const content =
        pushFormat === 'jsonl'
          ? exportSessionJsonl(currentSession)
          : exportSessionJson(currentSession);
      const resp = await fetch('/api/github/push-log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(),
        },
        body: JSON.stringify({
          repoFullName: `${validation.owner}/${validation.repo}`,
          branch: validation.branch,
          filePath: validation.filePath,
          commitMessage: pushCommitMsg.trim(),
          content,
        }),
      });
      const data = (await resp.json()) as {
        ok?: boolean;
        error?: string;
        owner?: string;
        repo?: string;
        branch?: string;
        filePath?: string;
        htmlUrl?: string;
        rawUrl?: string;
        commitSha?: string;
        commitUrl?: string;
        bytes?: number;
      };
      if (!resp.ok || !data.ok) {
        setPushError(data.error || `推送失败 (HTTP ${resp.status})`);
        return;
      }

      const rec: PushSuccessRecord = {
        repo: `${data.owner}/${data.repo}`,
        branch: data.branch || validation.branch!,
        filePath: data.filePath || validation.filePath!,
        htmlUrl: data.htmlUrl || '',
        rawUrl: data.rawUrl || '',
        commitSha: data.commitSha || '',
        commitUrl: data.commitUrl || '',
        bytes: data.bytes || content.length,
        pushedAt: new Date().toLocaleTimeString(),
      };
      setPushHistory((prev) => [rec, ...prev]);
      setFetchRepo(rec.repo);
      setFetchBranch(rec.branch);
      setFetchFilePath(rec.filePath);
    } catch (err) {
      setPushError(`推送请求异常: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPushLoading(false);
    }
  };

  const handleFetchFromGitHub = async (e: React.FormEvent) => {
    e.preventDefault();
    setFetchError('');
    const validation = validateGitHubPushTarget({
      repoFullName: fetchRepo,
      branch: fetchBranch,
      filePath: fetchFilePath,
    });
    if (!validation.ok) {
      setFetchError(validation.error || '仓库或文件路径格式无效。');
      return;
    }

    setFetchLoading(true);
    try {
      const qs = new URLSearchParams({
        repoFullName: `${validation.owner}/${validation.repo}`,
        branch: validation.branch!,
        filePath: validation.filePath!,
      });
      const resp = await fetch(`/api/github/fetch-log?${qs.toString()}`, {
        headers: authHeaders(),
      });
      const data = (await resp.json()) as {
        ok?: boolean;
        error?: string;
        content?: string;
      };
      if (!resp.ok || !data.ok || typeof data.content !== 'string') {
        setFetchError(data.error || `拉取失败 (HTTP ${resp.status})`);
        return;
      }
      const imported = parseImportedLogContent(
        data.content,
        `GitHub:${validation.owner}/${validation.repo}/${validation.filePath}`
      );
      traceLogger.importSession(imported);
      setSelectedSessionId(imported.id);
      setTab('trace');
      setImportFeedback(`已从 GitHub 打开 ${validation.filePath} (${imported.entries.length} 条记录)`);
      window.setTimeout(() => setImportFeedback(''), 3500);
    } catch (err) {
      setFetchError(`拉取请求异常: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setFetchLoading(false);
    }
  };

  const maxTotalMs = profileRows.reduce((m, r) => Math.max(m, r.totalMs), 1);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="运行追踪与函数调用日志分析器"
      data-testid="log-viewer-modal"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-3 sm:p-5"
    >
      <div className="w-full max-w-[1260px] h-[88vh] flex flex-col bg-[#0B0F17] border border-slate-700 rounded-md shadow-2xl overflow-hidden text-slate-100">
        {/* Top Modal Header */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-[#0D121B] border-b border-slate-800">
          <div className="flex flex-wrap items-center gap-2.5 min-w-0">
            <Terminal className="w-4 h-4 text-cyan-400 shrink-0" />
            <h2 className="text-[14px] font-semibold text-slate-100 whitespace-nowrap">
              运行日志与函数调用追踪分析器
            </h2>
            {currentSession && (
              <span
                data-testid="log-entry-count"
                className="text-[11px] font-mono px-1.5 py-0.5 rounded-[2px] bg-cyan-500/15 text-cyan-300 border border-cyan-500/35"
              >
                {currentSession.entries.length} 条记录
              </span>
            )}
            <select
              aria-label="选择日志会话"
              data-testid="log-session-select"
              value={currentSession?.id ?? ''}
              onChange={(e) => {
                setSelectedSessionId(e.target.value);
                setSelectedEntryId(null);
              }}
              className="max-w-[300px] bg-[#090C12] border border-slate-700 rounded-sm px-2 py-1 text-[11.5px] font-mono text-slate-200 focus:outline-none focus:border-cyan-500"
            >
              {sessions.length === 0 ? (
                <option value="">暂无日志会话</option>
              ) : (
                [...sessions].reverse().map((s) => (
                  <option key={s.id} value={s.id}>
                    [{s.kind}] {s.title} ({s.entries.length})
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 shrink-0">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.jsonl,.ndjson,.txt"
              onChange={handleLocalFileChange}
              className="hidden"
              data-testid="log-file-input"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11.5px] font-medium rounded-sm bg-slate-800 text-slate-200 border border-slate-700 hover:border-cyan-500/60 hover:text-white transition-colors"
            >
              <FileUp className="w-3.5 h-3.5 text-cyan-400" />
              打开本地日志
            </button>
            <button
              type="button"
              disabled={!currentSession || currentSession.entries.length === 0}
              onClick={() => handleDownload('jsonl')}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11.5px] font-mono rounded-sm bg-[#10151F] text-slate-300 border border-slate-700 hover:text-white hover:border-slate-500 disabled:opacity-40"
            >
              <Download className="w-3.5 h-3.5" />
              .JSONL
            </button>
            <button
              type="button"
              disabled={!currentSession || currentSession.entries.length === 0}
              onClick={() => handleDownload('json')}
              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11.5px] font-mono rounded-sm bg-[#10151F] text-slate-300 border border-slate-700 hover:text-white hover:border-slate-500 disabled:opacity-40"
            >
              <Download className="w-3.5 h-3.5" />
              .JSON
            </button>
            <button
              type="button"
              disabled={sessions.length === 0}
              onClick={() => {
                traceLogger.clearSessions();
                setSelectedEntryId(null);
              }}
              aria-label="清空全部日志"
              className="inline-flex items-center gap-1 px-2 py-1 text-[11.5px] rounded-sm bg-[#10151F] text-slate-400 border border-slate-800 hover:text-rose-300 hover:border-rose-500/40 disabled:opacity-40"
            >
              <Trash2 className="w-3.5 h-3.5" />
              清空
            </button>
            <button
              type="button"
              aria-label="关闭日志分析器"
              onClick={onClose}
              className="inline-flex items-center justify-center w-7 h-7 rounded-sm bg-slate-800/80 text-slate-300 hover:text-white hover:bg-slate-700 border border-slate-700"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Sub-Navigation Tabs + Status Notifications */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-1.5 bg-[#10151F] border-b border-slate-800">
          <div role="tablist" aria-label="日志分析视图" className="flex items-center gap-1.5">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'trace'}
              data-testid="log-tab-trace"
              onClick={() => setTab('trace')}
              className={`inline-flex items-center gap-1.5 px-3 py-1 text-[12px] font-medium rounded-sm border transition-colors ${
                tab === 'trace'
                  ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/60'
                  : 'bg-[#090C12] text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              调用追踪与输入输出
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'profile'}
              data-testid="log-tab-profile"
              onClick={() => setTab('profile')}
              className={`inline-flex items-center gap-1.5 px-3 py-1 text-[12px] font-medium rounded-sm border transition-colors ${
                tab === 'profile'
                  ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/60'
                  : 'bg-[#090C12] text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              函数耗时画像 ({profileRows.length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'github'}
              data-testid="log-tab-github"
              onClick={() => setTab('github')}
              className={`inline-flex items-center gap-1.5 px-3 py-1 text-[12px] font-medium rounded-sm border transition-colors ${
                tab === 'github'
                  ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/60'
                  : 'bg-[#090C12] text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              <FolderGit2 className="w-3.5 h-3.5" />
              GitHub 推送与拉取
            </button>
          </div>

          <div className="flex items-center gap-2 text-[11px] font-mono">
            {importFeedback && <span className="text-emerald-400">{importFeedback}</span>}
            {importError && <span className="text-rose-400">{importError}</span>}
            {currentSession?.summary && (
              <span className="text-slate-400 hidden md:inline">
                {currentSession.algoVersion?.toUpperCase() || ''}{' '}
                {currentSession.robotId || ''}{' '}
                {typeof currentSession.summary.finalTransMm === 'number'
                  ? `· ${currentSession.summary.finalTransMm.toFixed(2)}mm / ${(currentSession.summary.finalRotDeg ?? 0).toFixed(3)}°`
                  : ''}
              </span>
            )}
          </div>
        </div>

        {/* TAB 1: Call Trace + Input/Output Inspector */}
        {tab === 'trace' && (
          <div className="flex-1 flex flex-col min-h-0">
            {/* Filter Toolbar */}
            <div className="flex flex-wrap items-center gap-2 px-4 py-2 bg-[#0D121B] border-b border-slate-800">
              <div className="flex items-center gap-1 flex-wrap">
                {LEVEL_OPTIONS.map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    onClick={() => setLevelFilter(lvl)}
                    className={`px-2 py-0.5 text-[10.5px] font-mono uppercase rounded-[2px] border transition-colors ${
                      levelFilter === lvl
                        ? 'bg-cyan-500/20 text-cyan-200 border-cyan-500/60'
                        : 'bg-[#090C12] text-slate-400 border-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>

              <select
                aria-label="按模块筛选日志"
                value={moduleFilter}
                onChange={(e) => setModuleFilter(e.target.value)}
                className="bg-[#090C12] border border-slate-700 rounded-sm px-2 py-1 text-[11px] font-mono text-slate-300 focus:outline-none focus:border-cyan-500"
              >
                <option value="all">全部模块 ({availableModules.length})</option>
                {availableModules.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <div className="relative flex-1 min-w-[200px]">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="text"
                  aria-label="搜索函数名、参数或输出"
                  data-testid="log-search-input"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="搜索函数名、阶段、参数值或输出字段..."
                  className="w-full pl-8 pr-2.5 py-1 bg-[#090C12] border border-slate-700 rounded-sm text-[11.5px] font-mono text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <span className="text-[11px] font-mono text-slate-400 whitespace-nowrap">
                显示 {filteredEntries.length} / {currentSession?.entries.length ?? 0}
              </span>
            </div>

            {/* Split Master-Detail Area */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] min-h-0">
              {/* Left: Call Timeline List */}
              <div
                data-testid="log-entries-list"
                className="overflow-y-auto border-b lg:border-b-0 lg:border-r border-slate-800 bg-[#090C12] divide-y divide-slate-800/70"
              >
                {filteredEntries.length === 0 ? (
                  <div className="p-8 text-center text-[12px] font-mono text-slate-500">
                    暂无匹配的日志条目。点击顶部「运行单次标定」或「一键跑 6 组消融实验」即可自动记录全流程函数调用细节，或点击上方「打开本地日志」加载已有日志文件。
                  </div>
                ) : (
                  filteredEntries.map((entry) => {
                    const isSelected = selectedEntry?.id === entry.id;
                    return (
                      <button
                        key={entry.id}
                        type="button"
                        data-testid="log-entry-row"
                        onClick={() => setSelectedEntryId(entry.id)}
                        className={`w-full text-left px-3 py-1.5 flex items-center gap-2 text-[11px] font-mono transition-colors ${
                          isSelected
                            ? 'bg-cyan-500/15 text-slate-100'
                            : 'hover:bg-slate-800/45 text-slate-300'
                        }`}
                      >
                        <span className="w-8 text-right text-slate-500 shrink-0">
                          #{entry.seq}
                        </span>
                        <span className="w-16 text-right text-slate-500 shrink-0">
                          +{formatMs(entry.relMs)}
                        </span>
                        <span
                          className={`px-1.5 py-0 text-[9.5px] uppercase rounded-[2px] border shrink-0 ${levelBadgeClass(
                            entry.level
                          )}`}
                        >
                          {entry.level}
                        </span>
                        <div
                          className="min-w-0 flex-1 truncate"
                          style={{ paddingLeft: `${Math.min(4, entry.depth) * 10}px` }}
                        >
                          <span className="text-cyan-300 font-medium">
                            {entry.module}.{entry.fn}
                          </span>
                          {entry.message && entry.message !== `${entry.module}.${entry.fn}()` && (
                            <span className="ml-2 text-slate-400 font-sans">
                              {entry.message}
                            </span>
                          )}
                        </div>
                        <span className="w-14 text-right text-amber-300/90 shrink-0">
                          {formatMs(entry.durationMs)}
                        </span>
                      </button>
                    );
                  })
                )}
              </div>

              {/* Right: Detailed Input / Output JSON Inspector */}
              <div
                data-testid="log-io-inspector"
                className="overflow-y-auto p-3.5 bg-[#0B0F17] flex flex-col gap-3"
              >
                {!selectedEntry ? (
                  <div className="text-[12px] font-mono text-slate-500 p-4">
                    选择左侧任意函数调用或阶段事件，查看其完整序列化输入参数（Input）、返回值（Output）与耗时详情。
                  </div>
                ) : (
                  <>
                    <div className="flex items-start justify-between gap-2 pb-2 border-b border-slate-800">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`px-1.5 py-0.5 text-[10px] font-mono uppercase rounded-[2px] border ${levelBadgeClass(
                              selectedEntry.level
                            )}`}
                          >
                            {selectedEntry.level}
                          </span>
                          <span className="text-[13px] font-mono font-semibold text-cyan-300 break-all">
                            {selectedEntry.module}.{selectedEntry.fn}
                          </span>
                        </div>
                        <p className="mt-1 text-[12px] text-slate-300">
                          {selectedEntry.message || '—'}
                        </p>
                        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-0.5 text-[10.5px] font-mono text-slate-500">
                          <span>Seq #{selectedEntry.seq}</span>
                          <span>相对时间 +{formatMs(selectedEntry.relMs)}</span>
                          <span>耗时 {formatMs(selectedEntry.durationMs)}</span>
                          <span>调用深度 {selectedEntry.depth}</span>
                          <span>{selectedEntry.timestamp}</span>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => copyText(JSON.stringify(selectedEntry, null, 2), selectedEntry.id)}
                        className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-mono rounded-sm bg-slate-800 text-slate-300 border border-slate-700 hover:text-white shrink-0"
                      >
                        {copiedId === selectedEntry.id ? (
                          <Check className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                        {copiedId === selectedEntry.id ? '已复制' : '复制 JSON'}
                      </button>
                    </div>

                    {selectedEntry.error && (
                      <div className="p-2.5 rounded bg-rose-950/50 border border-rose-500/40 text-[11.5px] font-mono text-rose-200">
                        <div className="text-[10px] uppercase text-rose-400 mb-1">Exception / Error</div>
                        {selectedEntry.error}
                      </div>
                    )}

                    <div>
                      <div className="text-[11px] font-mono font-semibold text-amber-300 mb-1">
                        输入参数 (Input)
                      </div>
                      <pre className="p-2.5 rounded bg-[#090C12] border border-slate-800 text-[11px] font-mono text-slate-200 overflow-x-auto max-h-[240px] whitespace-pre-wrap break-words">
                        {formatJsonBlock(selectedEntry.input)}
                      </pre>
                    </div>

                    <div>
                      <div className="text-[11px] font-mono font-semibold text-emerald-300 mb-1">
                        输出结果 (Output)
                      </div>
                      <pre className="p-2.5 rounded bg-[#090C12] border border-slate-800 text-[11px] font-mono text-slate-200 overflow-x-auto max-h-[300px] whitespace-pre-wrap break-words">
                        {formatJsonBlock(selectedEntry.output)}
                      </pre>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Function Latency Profile */}
        {tab === 'profile' && (
          <div className="flex-1 overflow-y-auto p-4 bg-[#090C12]" data-testid="log-profile-panel">
            {profileRows.length === 0 ? (
              <div className="text-center text-[12px] font-mono text-slate-500 p-8">
                当前会话尚无带耗时的函数调用记录。
              </div>
            ) : (
              <table className="w-full text-left border-collapse font-mono text-[11.5px]">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                    <th className="py-2 px-2.5 font-medium">模块 · 函数</th>
                    <th className="py-2 px-2.5 font-medium text-right">调用次数</th>
                    <th className="py-2 px-2.5 font-medium text-right">总耗时</th>
                    <th className="py-2 px-2.5 font-medium text-right">平均耗时</th>
                    <th className="py-2 px-2.5 font-medium text-right">峰值耗时</th>
                    <th className="py-2 px-2.5 font-medium text-right">异常</th>
                    <th className="py-2 px-2.5 font-medium w-40">耗时占比</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {profileRows.map((row) => {
                    const pct = Math.max(2, (row.totalMs / Math.max(1e-3, maxTotalMs)) * 100);
                    return (
                      <tr
                        key={row.key}
                        onClick={() => {
                          setModuleFilter(row.module);
                          setSearchQuery(row.fn);
                          setLevelFilter('all');
                          setTab('trace');
                        }}
                        className="hover:bg-slate-800/45 cursor-pointer transition-colors"
                      >
                        <td className="py-2 px-2.5 text-cyan-300">{row.key}</td>
                        <td className="py-2 px-2.5 text-right text-slate-200">{row.calls}</td>
                        <td className="py-2 px-2.5 text-right text-amber-300">
                          {formatMs(row.totalMs)}
                        </td>
                        <td className="py-2 px-2.5 text-right text-slate-300">
                          {formatMs(row.avgMs)}
                        </td>
                        <td className="py-2 px-2.5 text-right text-slate-300">
                          {formatMs(row.maxMs)}
                        </td>
                        <td
                          className={`py-2 px-2.5 text-right ${
                            row.errors > 0 ? 'text-rose-400 font-semibold' : 'text-slate-500'
                          }`}
                        >
                          {row.errors}
                        </td>
                        <td className="py-2 px-2.5">
                          <div className="h-2 w-full bg-slate-800 rounded-[2px] overflow-hidden">
                            <div
                              className="h-full bg-cyan-400/80"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        )}

        {/* TAB 3: GitHub Push & Remote Open */}
        {tab === 'github' && (
          <div
            data-testid="log-github-panel"
            className="flex-1 overflow-y-auto p-4 bg-[#090C12] flex flex-col gap-4"
          >
            {/* Connection Status Banner */}
            <div className="p-3.5 rounded bg-[#0D121B] border border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[12.5px] font-semibold text-slate-100">
                  <FolderGit2 className="w-4 h-4 text-cyan-400" />
                  <span>GitHub 连接状态</span>
                  {ghStatus?.connected ? (
                    <span className="px-2 py-0.5 text-[10.5px] font-mono rounded-[2px] bg-emerald-500/15 text-emerald-300 border border-emerald-500/40">
                      已连接 · {ghStatus.authSource === 'oauth' ? 'OAuth' : '服务端 Token'}
                      {ghStatus.user ? ` (@${ghStatus.user.login})` : ''}
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 text-[10.5px] font-mono rounded-[2px] bg-amber-500/15 text-amber-300 border border-amber-500/40">
                      未授权推送（仍可直接拉取公开仓库日志或导出本地文件）
                    </span>
                  )}
                </div>
                {ghStatus && !ghStatus.connected && (
                  <div className="mt-2 text-[11px] font-mono text-slate-400 space-y-1">
                    <p>
                      如需推送到 GitHub：请在 AI Studio Secrets 中配置{' '}
                      <span className="text-slate-200">GITHUB_CLIENT_ID</span> +{' '}
                      <span className="text-slate-200">GITHUB_CLIENT_SECRET</span>（OAuth 登录），或直接配置服务端{' '}
                      <span className="text-slate-200">GITHUB_TOKEN</span>（凭据仅留存于服务端，前端绝不接触密钥）。
                    </p>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-slate-500">OAuth Callback URL:</span>
                      <code className="px-1.5 py-0.5 bg-[#090C12] border border-slate-800 rounded text-cyan-300">
                        {ghStatus.callbackUrl}
                      </code>
                      <button
                        type="button"
                        onClick={() => copyText(ghStatus.callbackUrl, 'cb_url')}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] rounded bg-slate-800 text-slate-300 hover:text-white border border-slate-700"
                      >
                        {copiedId === 'cb_url' ? (
                          <Check className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                        复制回调地址
                      </button>
                    </div>
                  </div>
                )}
                {ghAuthError && (
                  <p className="mt-1.5 text-[11.5px] font-mono text-rose-400">{ghAuthError}</p>
                )}
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  disabled={ghStatusLoading}
                  onClick={refreshGitHubStatus}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11.5px] font-mono rounded-sm bg-slate-800 text-slate-200 border border-slate-700 hover:bg-slate-700 disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${ghStatusLoading ? 'animate-spin' : ''}`} />
                  刷新状态
                </button>
                {ghStatus?.authSource === 'oauth' ? (
                  <button
                    type="button"
                    onClick={handleDisconnectGitHub}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11.5px] font-medium rounded-sm bg-rose-500/15 text-rose-300 border border-rose-500/40 hover:bg-rose-500/25"
                  >
                    <LogOut className="w-3.5 h-3.5" />
                    断开 OAuth
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleConnectGitHub}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11.5px] font-medium rounded-sm bg-cyan-500 text-slate-950 hover:bg-cyan-400"
                  >
                    <Link2 className="w-3.5 h-3.5" />
                    连接 GitHub (OAuth)
                  </button>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Push Current Session Form */}
              <form
                onSubmit={handlePushToGitHub}
                data-testid="github-push-form"
                className="p-4 rounded bg-[#0D121B] border border-slate-800 flex flex-col gap-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <h3 className="text-[13px] font-semibold text-slate-100 flex items-center gap-1.5">
                    <UploadCloud className="w-4 h-4 text-cyan-400" />
                    推送当前日志到 GitHub 仓库
                  </h3>
                  <div className="inline-flex rounded-sm border border-slate-700 overflow-hidden text-[10.5px] font-mono">
                    <button
                      type="button"
                      onClick={() => setPushFormat('jsonl')}
                      className={`px-2 py-0.5 ${
                        pushFormat === 'jsonl'
                          ? 'bg-cyan-500/25 text-cyan-200'
                          : 'bg-[#090C12] text-slate-400'
                      }`}
                    >
                      JSONL
                    </button>
                    <button
                      type="button"
                      onClick={() => setPushFormat('json')}
                      className={`px-2 py-0.5 ${
                        pushFormat === 'json'
                          ? 'bg-cyan-500/25 text-cyan-200'
                          : 'bg-[#090C12] text-slate-400'
                      }`}
                    >
                      JSON
                    </button>
                  </div>
                </div>

                <label className="flex flex-col gap-1 text-[11.5px]">
                  <span className="text-slate-400 font-mono">目标仓库 (owner/repo)</span>
                  <input
                    type="text"
                    list="github-repos-datalist"
                    value={pushRepo}
                    onChange={(e) => setPushRepo(e.target.value)}
                    placeholder="例如: your-org/hand-eye-calib-logs"
                    data-testid="github-push-repo"
                    className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-cyan-500"
                  />
                  <datalist id="github-repos-datalist">
                    {ghRepos.map((r) => (
                      <option key={r.fullName} value={r.fullName} />
                    ))}
                  </datalist>
                </label>

                <div className="grid grid-cols-1 sm:grid-cols-[120px_minmax(0,1fr)] gap-2.5">
                  <label className="flex flex-col gap-1 text-[11.5px]">
                    <span className="text-slate-400 font-mono flex items-center gap-1">
                      <GitBranch className="w-3 h-3" /> 分支
                    </span>
                    <input
                      type="text"
                      value={pushBranch}
                      onChange={(e) => setPushBranch(e.target.value)}
                      data-testid="github-push-branch"
                      className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-cyan-500"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-[11.5px]">
                    <span className="text-slate-400 font-mono">仓库内文件路径</span>
                    <input
                      type="text"
                      value={pushFilePath}
                      onChange={(e) => setPushFilePath(e.target.value)}
                      data-testid="github-push-path"
                      className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-cyan-500"
                    />
                  </label>
                </div>

                <label className="flex flex-col gap-1 text-[11.5px]">
                  <span className="text-slate-400 font-mono">Commit Message</span>
                  <input
                    type="text"
                    value={pushCommitMsg}
                    onChange={(e) => setPushCommitMsg(e.target.value)}
                    className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-cyan-500"
                  />
                </label>

                {pushError && (
                  <div
                    role="alert"
                    data-testid="github-push-error"
                    className="p-2 rounded bg-rose-950/50 border border-rose-500/40 text-[11.5px] font-mono text-rose-300"
                  >
                    {pushError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={pushLoading}
                  data-testid="github-push-submit"
                  className="mt-1 inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-[12px] font-medium rounded-sm bg-cyan-500 text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
                >
                  <UploadCloud className="w-4 h-4" />
                  {pushLoading
                    ? '正在推送至 GitHub...'
                    : `推送日志 (${currentSession?.entries.length ?? 0} 条) 到 GitHub`}
                </button>
              </form>

              {/* Pull / Open Remote GitHub Log Inside App */}
              <form
                onSubmit={handleFetchFromGitHub}
                data-testid="github-fetch-form"
                className="p-4 rounded bg-[#0D121B] border border-slate-800 flex flex-col gap-3"
              >
                <h3 className="text-[13px] font-semibold text-slate-100 flex items-center gap-1.5">
                  <Download className="w-4 h-4 text-emerald-400" />
                  从 GitHub 拉取日志并在当前 App 打开分析
                </h3>
                <p className="text-[11.5px] text-slate-400 leading-relaxed">
                  协作者或其他分析节点可将已推送至 GitHub 的 <code className="font-mono text-slate-200">.json</code> /{' '}
                  <code className="font-mono text-slate-200">.jsonl</code> 日志直接拉取到本分析器中，查看完整函数调用栈、输入输出与耗时分布。
                </p>

                <label className="flex flex-col gap-1 text-[11.5px]">
                  <span className="text-slate-400 font-mono">来源仓库 (owner/repo)</span>
                  <input
                    type="text"
                    value={fetchRepo}
                    onChange={(e) => setFetchRepo(e.target.value)}
                    placeholder="例如: your-org/hand-eye-calib-logs"
                    data-testid="github-fetch-repo"
                    className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-emerald-500"
                  />
                </label>

                <div className="grid grid-cols-1 sm:grid-cols-[120px_minmax(0,1fr)] gap-2.5">
                  <label className="flex flex-col gap-1 text-[11.5px]">
                    <span className="text-slate-400 font-mono">分支</span>
                    <input
                      type="text"
                      value={fetchBranch}
                      onChange={(e) => setFetchBranch(e.target.value)}
                      className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-emerald-500"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-[11.5px]">
                    <span className="text-slate-400 font-mono">日志文件路径</span>
                    <input
                      type="text"
                      value={fetchFilePath}
                      onChange={(e) => setFetchFilePath(e.target.value)}
                      placeholder="logs/calib_v3_panda_single_....jsonl"
                      className="px-2.5 py-1.5 bg-[#090C12] border border-slate-700 rounded-sm font-mono text-[12px] text-slate-100 focus:outline-none focus:border-emerald-500"
                    />
                  </label>
                </div>

                {fetchError && (
                  <div
                    role="alert"
                    data-testid="github-fetch-error"
                    className="p-2 rounded bg-rose-950/50 border border-rose-500/40 text-[11.5px] font-mono text-rose-300"
                  >
                    {fetchError}
                  </div>
                )}

                <button
                  type="submit"
                  disabled={fetchLoading}
                  data-testid="github-fetch-submit"
                  className="mt-auto inline-flex items-center justify-center gap-1.5 px-3.5 py-2 text-[12px] font-medium rounded-sm bg-emerald-500/20 text-emerald-200 border border-emerald-500/50 hover:bg-emerald-500/30 disabled:opacity-50"
                >
                  <Download className="w-4 h-4" />
                  {fetchLoading ? '正在拉取日志...' : '从 GitHub 拉取并打开查看'}
                </button>
              </form>
            </div>

            {/* Push History */}
            {pushHistory.length > 0 && (
              <div className="p-3.5 rounded bg-[#0D121B] border border-slate-800">
                <h4 className="text-[12px] font-semibold text-slate-200 mb-2">本次已推送记录</h4>
                <div className="space-y-2">
                  {pushHistory.map((item, idx) => (
                    <div
                      key={`${item.commitSha}_${idx}`}
                      className="flex flex-wrap items-center justify-between gap-2 p-2 rounded bg-[#090C12] border border-slate-800 text-[11px] font-mono"
                    >
                      <div className="min-w-0">
                        <span className="text-emerald-400">{item.repo}</span>
                        <span className="text-slate-500">@{item.branch} / </span>
                        <span className="text-cyan-300">{item.filePath}</span>
                        <span className="ml-2 text-slate-500">
                          ({(item.bytes / 1024).toFixed(1)} KB · {item.pushedAt})
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        {isSafeGitHubUrl(item.htmlUrl) && (
                          <a
                            href={item.htmlUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-cyan-300 hover:underline"
                          >
                            GitHub 查看 <ExternalLink className="w-3 h-3" />
                          </a>
                        )}
                        {isSafeGitHubUrl(item.rawUrl) && (
                          <button
                            type="button"
                            onClick={() => copyText(item.rawUrl, `raw_${idx}`)}
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 hover:text-white border border-slate-700"
                          >
                            {copiedId === `raw_${idx}` ? '已复制 Raw 链接' : '复制 Raw 地址'}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
