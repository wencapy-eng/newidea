/**
 * Pure validation helpers for GitHub repository targets and log file paths.
 * Shared by both client UI forms and Express backend routes to prevent path traversal,
 * URL injection, and malformed repository commits.
 */

const OWNER_RE = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/;
const REPO_RE = /^[A-Za-z0-9._-]{1,100}$/;
const BRANCH_RE = /^[A-Za-z0-9._/-]{1,120}$/;
const PATH_SEGMENT_RE = /^[A-Za-z0-9._-]+$/;
const ALLOWED_EXTENSIONS = ['.json', '.jsonl', '.md', '.txt'];

export interface GitHubTargetValidation {
  ok: boolean;
  error?: string;
  owner?: string;
  repo?: string;
  branch?: string;
  filePath?: string;
}

export function parseOwnerRepo(input: string): { owner: string; repo: string } | null {
  const cleaned = input
    .trim()
    .replace(/^https?:\/\/github\.com\//i, '')
    .replace(/\.git$/i, '')
    .replace(/^\/+|\/+$/g, '');
  const parts = cleaned.split('/');
  if (parts.length !== 2) return null;
  const [owner, repo] = parts;
  if (!OWNER_RE.test(owner) || !REPO_RE.test(repo) || repo === '.' || repo === '..') {
    return null;
  }
  return { owner, repo };
}

export function validateGitHubBranch(branch: string): string | null {
  const b = branch.trim();
  if (!b || !BRANCH_RE.test(b) || b.includes('..') || b.startsWith('/') || b.endsWith('/')) {
    return null;
  }
  return b;
}

export function validateGitHubLogPath(filePath: string): string | null {
  const trimmed = filePath.trim().replace(/^\/+/, '');
  if (!trimmed || trimmed.length > 240) return null;
  if (trimmed.includes('\\') || trimmed.includes('..') || trimmed.includes('//')) return null;

  const segments = trimmed.split('/');
  for (const seg of segments) {
    if (!seg || seg === '.' || seg === '..' || !PATH_SEGMENT_RE.test(seg)) {
      return null;
    }
  }

  const lower = trimmed.toLowerCase();
  if (!ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
    return null;
  }
  return trimmed;
}

export function validateGitHubPushTarget(params: {
  repoFullName: string;
  branch: string;
  filePath: string;
}): GitHubTargetValidation {
  const parsed = parseOwnerRepo(params.repoFullName);
  if (!parsed) {
    return { ok: false, error: '仓库格式无效，请输入 owner/repo（例如 alice/hand-eye-logs）' };
  }
  const branch = validateGitHubBranch(params.branch);
  if (!branch) {
    return { ok: false, error: '分支名称无效（仅允许字母、数字、.、_、-、/）' };
  }
  const filePath = validateGitHubLogPath(params.filePath);
  if (!filePath) {
    return {
      ok: false,
      error: '日志路径无效（必须为相对路径、禁止 ..，且扩展名需为 .json / .jsonl / .md / .txt）',
    };
  }
  return {
    ok: true,
    owner: parsed.owner,
    repo: parsed.repo,
    branch,
    filePath,
  };
}

export function buildDefaultLogPath(meta: {
  algoVersion?: string;
  robotId?: string;
  kind?: string;
  format: 'json' | 'jsonl';
}): string {
  const now = new Date();
  const stamp = [
    now.getUTCFullYear(),
    String(now.getUTCMonth() + 1).padStart(2, '0'),
    String(now.getUTCDate()).padStart(2, '0'),
    '_',
    String(now.getUTCHours()).padStart(2, '0'),
    String(now.getUTCMinutes()).padStart(2, '0'),
    String(now.getUTCSeconds()).padStart(2, '0'),
  ].join('');
  const ver = (meta.algoVersion || 'v3').replace(/[^a-zA-Z0-9_-]/g, '');
  const robot = (meta.robotId || 'panda').replace(/[^a-zA-Z0-9_-]/g, '');
  const kind = (meta.kind || 'run').replace(/[^a-zA-Z0-9_-]/g, '');
  return `logs/calib_${ver}_${robot}_${kind}_${stamp}.${meta.format}`;
}

export function isSafeGitHubUrl(url: string | undefined | null): boolean {
  if (!url) return false;
  try {
    const parsed = new URL(url);
    return (
      parsed.protocol === 'https:' &&
      (parsed.hostname === 'github.com' || parsed.hostname === 'raw.githubusercontent.com')
    );
  } catch {
    return false;
  }
}
