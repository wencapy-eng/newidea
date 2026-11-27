import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';
import {
  validateGitHubPushTarget,
} from './src/utils/githubValidation.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const GITHUB_API_BASE = 'https://api.github.com';
const GITHUB_AUTHORIZE_URL = 'https://github.com/login/oauth/authorize';
const GITHUB_TOKEN_URL = 'https://github.com/login/oauth/access_token';
const SESSION_TTL_MS = 8 * 60 * 60 * 1000;
const STATE_TTL_MS = 10 * 60 * 1000;

interface GitHubUserProfile {
  login: string;
  name: string;
  avatarUrl: string;
}

interface OAuthSessionRecord {
  accessToken: string;
  user: GitHubUserProfile;
  expiresAt: number;
}

const oauthSessions = new Map<string, OAuthSessionRecord>();
const pendingStates = new Map<string, number>();

function cleanupExpiredMaps(): void {
  const now = Date.now();
  for (const [k, exp] of pendingStates.entries()) {
    if (exp <= now) pendingStates.delete(k);
  }
  for (const [k, rec] of oauthSessions.entries()) {
    if (rec.expiresAt <= now) oauthSessions.delete(k);
  }
}

function getCanonicalCallbackUrl(): string {
  const appUrl = (process.env.APP_URL || '').trim().replace(/\/+$/, '');
  if (appUrl && /^https?:\/\//i.test(appUrl)) {
    return `${appUrl}/api/github/callback`;
  }
  return `http://localhost:${PORT}/api/github/callback`;
}

function extractBearerToken(req: Request): string | null {
  const auth = req.header('authorization');
  if (!auth) return null;
  const match = /^Bearer\s+([A-Za-z0-9_-]{16,128})$/.exec(auth.trim());
  return match ? match[1] : null;
}

function resolveGitHubCredentials(req: Request): {
  token: string;
  authSource: 'oauth' | 'server_token';
  user: GitHubUserProfile | null;
} | null {
  cleanupExpiredMaps();
  const bearer = extractBearerToken(req);
  if (bearer) {
    const sess = oauthSessions.get(bearer);
    if (sess && sess.expiresAt > Date.now()) {
      return {
        token: sess.accessToken,
        authSource: 'oauth',
        user: sess.user,
      };
    }
  }
  const serverToken = (process.env.GITHUB_TOKEN || '').trim();
  if (serverToken) {
    return {
      token: serverToken,
      authSource: 'server_token',
      user: null,
    };
  }
  return null;
}

function safeJsonForScript(payload: unknown): string {
  return JSON.stringify(payload).replace(/</g, '\\u003c').replace(/>/g, '\\u003e');
}

function renderOAuthCallbackHtml(payload: Record<string, unknown>, message: string): string {
  const serialized = safeJsonForScript(payload);
  const safeMsg = message.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return `<!doctype html>
<html lang="zh-CN">
  <head><meta charset="utf-8" /><title>GitHub 授权回调</title></head>
  <body style="background:#090C12;color:#e2e8f0;font-family:monospace;padding:24px;">
    <p>${safeMsg}</p>
    <script>
      (function () {
        var msg = ${serialized};
        if (window.opener) {
          window.opener.postMessage(msg, '*');
          window.close();
        }
      })();
    </script>
  </body>
</html>`;
}

function githubHeaders(token?: string, accept = 'application/vnd.github+json'): Record<string, string> {
  const headers: Record<string, string> = {
    Accept: accept,
    'User-Agent': 'hand-eye-calib-logger/1.0',
    'X-GitHub-Api-Version': '2022-11-28',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

async function fetchGitHubUser(token: string): Promise<GitHubUserProfile | null> {
  try {
    const resp = await fetch(`${GITHUB_API_BASE}/user`, {
      headers: githubHeaders(token),
    });
    if (!resp.ok) return null;
    const data = (await resp.json()) as {
      login?: string;
      name?: string | null;
      avatar_url?: string;
    };
    if (!data.login) return null;
    return {
      login: data.login,
      name: data.name || data.login,
      avatarUrl: data.avatar_url || '',
    };
  } catch {
    return null;
  }
}

async function createAppServer(): Promise<void> {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: '16mb' }));

  // 1. GitHub integration status (never leaks tokens or secrets)
  app.get('/api/github/status', async (req: Request, res: Response) => {
    const oauthConfigured = Boolean(
      (process.env.GITHUB_CLIENT_ID || '').trim() &&
        (process.env.GITHUB_CLIENT_SECRET || '').trim()
    );
    const serverTokenConfigured = Boolean((process.env.GITHUB_TOKEN || '').trim());
    const creds = resolveGitHubCredentials(req);
    let user = creds?.user ?? null;
    if (creds && creds.authSource === 'server_token' && !user) {
      user = await fetchGitHubUser(creds.token);
    }
    res.json({
      oauthConfigured,
      serverTokenConfigured,
      connected: Boolean(creds),
      authSource: creds?.authSource ?? null,
      user,
      defaultRepo: (process.env.GITHUB_DEFAULT_REPO || '').trim(),
      callbackUrl: getCanonicalCallbackUrl(),
    });
  });

  // 2. Construct GitHub OAuth authorize URL (popup opens provider URL directly)
  app.get('/api/github/auth/url', (_req: Request, res: Response) => {
    cleanupExpiredMaps();
    const clientId = (process.env.GITHUB_CLIENT_ID || '').trim();
    const clientSecret = (process.env.GITHUB_CLIENT_SECRET || '').trim();
    if (!clientId || !clientSecret) {
      res.status(400).json({
        error: '尚未配置 GITHUB_CLIENT_ID 与 GITHUB_CLIENT_SECRET，请在环境变量中设置后再使用 OAuth 登录。',
        callbackUrl: getCanonicalCallbackUrl(),
      });
      return;
    }

    const state = crypto.randomBytes(24).toString('hex');
    pendingStates.set(state, Date.now() + STATE_TTL_MS);

    const params = new URLSearchParams({
      client_id: clientId,
      redirect_uri: getCanonicalCallbackUrl(),
      scope: 'repo',
      state,
    });
    res.json({ url: `${GITHUB_AUTHORIZE_URL}?${params.toString()}` });
  });

  // 3. GitHub OAuth callback (handles trailing slash variations)
  const handleOAuthCallback = async (req: Request, res: Response) => {
    cleanupExpiredMaps();
    const code = typeof req.query.code === 'string' ? req.query.code.trim() : '';
    const state = typeof req.query.state === 'string' ? req.query.state.trim() : '';
    const oauthError = typeof req.query.error === 'string' ? req.query.error.trim() : '';

    if (oauthError) {
      res
        .status(400)
        .send(
          renderOAuthCallbackHtml(
            { type: 'GITHUB_OAUTH_ERROR', error: `GitHub 授权取消或失败 (${oauthError})` },
            'GitHub 授权未完成，请关闭此窗口。'
          )
        );
      return;
    }

    const stateExpiry = state ? pendingStates.get(state) : undefined;
    if (!code || !state || !stateExpiry || stateExpiry <= Date.now()) {
      res
        .status(400)
        .send(
          renderOAuthCallbackHtml(
            { type: 'GITHUB_OAUTH_ERROR', error: 'OAuth state 校验失败或已过期，请重试连接。' },
            'OAuth 校验失败，请返回主页面重试。'
          )
        );
      return;
    }
    pendingStates.delete(state);

    const clientId = (process.env.GITHUB_CLIENT_ID || '').trim();
    const clientSecret = (process.env.GITHUB_CLIENT_SECRET || '').trim();
    if (!clientId || !clientSecret) {
      res
        .status(500)
        .send(
          renderOAuthCallbackHtml(
            { type: 'GITHUB_OAUTH_ERROR', error: '服务端未配置 GitHub OAuth 凭据。' },
            '服务端未配置 GitHub OAuth 凭据。'
          )
        );
      return;
    }

    try {
      const tokenResp = await fetch(GITHUB_TOKEN_URL, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          client_id: clientId,
          client_secret: clientSecret,
          code,
          redirect_uri: getCanonicalCallbackUrl(),
        }),
      });
      const tokenJson = (await tokenResp.json()) as {
        access_token?: string;
        error?: string;
        error_description?: string;
      };
      if (!tokenResp.ok || !tokenJson.access_token) {
        res
          .status(400)
          .send(
            renderOAuthCallbackHtml(
              {
                type: 'GITHUB_OAUTH_ERROR',
                error: tokenJson.error_description || tokenJson.error || '换取 GitHub Token 失败',
              },
              'GitHub 令牌换取失败，请关闭窗口后重试。'
            )
          );
        return;
      }

      const user = (await fetchGitHubUser(tokenJson.access_token)) ?? {
        login: 'github-user',
        name: 'GitHub User',
        avatarUrl: '',
      };
      const sessionToken = crypto.randomBytes(32).toString('hex');
      oauthSessions.set(sessionToken, {
        accessToken: tokenJson.access_token,
        user,
        expiresAt: Date.now() + SESSION_TTL_MS,
      });

      res.send(
        renderOAuthCallbackHtml(
          { type: 'GITHUB_OAUTH_SUCCESS', sessionToken, user },
          'GitHub 授权成功，窗口将自动关闭。'
        )
      );
    } catch {
      res
        .status(500)
        .send(
          renderOAuthCallbackHtml(
            { type: 'GITHUB_OAUTH_ERROR', error: '连接 GitHub 授权服务失败，请稍后重试。' },
            '连接 GitHub 授权服务失败。'
          )
        );
    }
  };
  app.get(['/api/github/callback', '/api/github/callback/'], handleOAuthCallback);

  // 4. Disconnect OAuth session
  app.post('/api/github/disconnect', (req: Request, res: Response) => {
    const bearer = extractBearerToken(req);
    if (bearer) {
      oauthSessions.delete(bearer);
    }
    res.json({ ok: true });
  });

  // 5. List accessible repositories for connected account
  app.get('/api/github/repos', async (req: Request, res: Response) => {
    const creds = resolveGitHubCredentials(req);
    if (!creds) {
      res.status(401).json({ error: '请先连接 GitHub 账号或在服务端配置 GITHUB_TOKEN。' });
      return;
    }
    try {
      const resp = await fetch(
        `${GITHUB_API_BASE}/user/repos?sort=updated&per_page=30&affiliation=owner,collaborator,organization_member`,
        { headers: githubHeaders(creds.token) }
      );
      if (!resp.ok) {
        const errBody = await resp.text();
        res.status(resp.status).json({
          error: `获取仓库列表失败 (${resp.status}): ${errBody.slice(0, 180)}`,
        });
        return;
      }
      const list = (await resp.json()) as Array<{
        full_name: string;
        default_branch: string;
        private: boolean;
      }>;
      res.json({
        repos: list.map((r) => ({
          fullName: r.full_name,
          defaultBranch: r.default_branch || 'main',
          isPrivate: Boolean(r.private),
        })),
      });
    } catch {
      res.status(502).json({ error: '请求 GitHub 仓库列表时网络异常。' });
    }
  });

  // 6. Push structured log file to GitHub repository via Contents API
  app.post('/api/github/push-log', async (req: Request, res: Response) => {
    const creds = resolveGitHubCredentials(req);
    if (!creds) {
      res.status(401).json({
        error: '尚未授权 GitHub：请先点击「连接 GitHub (OAuth)」或在 AI Studio Secrets 中配置 GITHUB_TOKEN。',
      });
      return;
    }

    const body = (req.body ?? {}) as Record<string, unknown>;
    const repoFullName = typeof body.repoFullName === 'string' ? body.repoFullName : '';
    const branch = typeof body.branch === 'string' ? body.branch : 'main';
    const filePath = typeof body.filePath === 'string' ? body.filePath : '';
    const commitMessageRaw = typeof body.commitMessage === 'string' ? body.commitMessage.trim() : '';
    const content = typeof body.content === 'string' ? body.content : '';

    const check = validateGitHubPushTarget({ repoFullName, branch, filePath });
    if (!check.ok || !check.owner || !check.repo || !check.branch || !check.filePath) {
      res.status(400).json({ error: check.error || '目标仓库或路径参数无效。' });
      return;
    }
    if (!content.trim()) {
      res.status(400).json({ error: '日志内容不能为空。' });
      return;
    }
    const contentBytes = Buffer.byteLength(content, 'utf8');
    if (contentBytes > 14 * 1024 * 1024) {
      res.status(413).json({ error: '日志文件超过 14 MB 上限，请减少会话条目或改用本地导出。' });
      return;
    }

    const encodedPath = check.filePath.split('/').map(encodeURIComponent).join('/');
    const contentsUrl = `${GITHUB_API_BASE}/repos/${encodeURIComponent(check.owner)}/${encodeURIComponent(check.repo)}/contents/${encodedPath}`;
    const commitMessage =
      commitMessageRaw.slice(0, 200) || `chore(logs): upload calibration trace ${check.filePath}`;

    try {
      // Check existing file SHA so re-pushing to the same path performs an atomic update
      let existingSha: string | undefined;
      const getResp = await fetch(`${contentsUrl}?ref=${encodeURIComponent(check.branch)}`, {
        headers: githubHeaders(creds.token),
      });
      if (getResp.ok) {
        const existing = (await getResp.json()) as { sha?: string };
        if (typeof existing.sha === 'string') existingSha = existing.sha;
      }

      const putBody: Record<string, string> = {
        message: commitMessage,
        content: Buffer.from(content, 'utf8').toString('base64'),
        branch: check.branch,
      };
      if (existingSha) putBody.sha = existingSha;

      const putResp = await fetch(contentsUrl, {
        method: 'PUT',
        headers: {
          ...githubHeaders(creds.token),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(putBody),
      });

      const putJson = (await putResp.json()) as {
        message?: string;
        content?: { html_url?: string; download_url?: string; path?: string };
        commit?: { sha?: string; html_url?: string };
      };
      if (!putResp.ok) {
        res.status(putResp.status).json({
          error: `GitHub 推送失败 (${putResp.status}): ${putJson.message || '请确认仓库存在、分支正确且账号有写入权限。'}`,
        });
        return;
      }

      res.json({
        ok: true,
        owner: check.owner,
        repo: check.repo,
        branch: check.branch,
        filePath: putJson.content?.path || check.filePath,
        htmlUrl: putJson.content?.html_url || '',
        rawUrl:
          putJson.content?.download_url ||
          `https://raw.githubusercontent.com/${check.owner}/${check.repo}/${check.branch}/${check.filePath}`,
        commitSha: putJson.commit?.sha || '',
        commitUrl: putJson.commit?.html_url || '',
        bytes: contentBytes,
      });
    } catch {
      res.status(502).json({ error: '向 GitHub API 推送日志文件时网络异常。' });
    }
  });

  // 7. Fetch a log file from GitHub so collaborators can open & inspect it inside this app
  app.get('/api/github/fetch-log', async (req: Request, res: Response) => {
    const repoFullName = typeof req.query.repoFullName === 'string' ? req.query.repoFullName : '';
    const branch = typeof req.query.branch === 'string' ? req.query.branch : 'main';
    const filePath = typeof req.query.filePath === 'string' ? req.query.filePath : '';

    const check = validateGitHubPushTarget({ repoFullName, branch, filePath });
    if (!check.ok || !check.owner || !check.repo || !check.branch || !check.filePath) {
      res.status(400).json({ error: check.error || '仓库或日志文件路径格式无效。' });
      return;
    }

    const creds = resolveGitHubCredentials(req);
    const encodedPath = check.filePath.split('/').map(encodeURIComponent).join('/');
    const contentsUrl = `${GITHUB_API_BASE}/repos/${encodeURIComponent(check.owner)}/${encodeURIComponent(check.repo)}/contents/${encodedPath}?ref=${encodeURIComponent(check.branch)}`;

    try {
      const resp = await fetch(contentsUrl, {
        headers: githubHeaders(creds?.token, 'application/vnd.github.raw+json'),
      });
      if (!resp.ok) {
        const msg =
          resp.status === 404
            ? '未在目标仓库/分支找到该日志文件（若为私有仓库，请先连接 GitHub 授权）。'
            : `拉取 GitHub 日志失败 (${resp.status})`;
        res.status(resp.status).json({ error: msg });
        return;
      }
      const content = await resp.text();
      res.json({
        ok: true,
        repoFullName: `${check.owner}/${check.repo}`,
        branch: check.branch,
        filePath: check.filePath,
        content,
      });
    } catch {
      res.status(502).json({ error: '从 GitHub 拉取日志文件时网络异常。' });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build output not found.');
      }
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[server] listening on http://0.0.0.0:${PORT}`);
  });
}

createAppServer().catch((err) => {
  console.error('[server] startup failed:', err);
  process.exit(1);
});
