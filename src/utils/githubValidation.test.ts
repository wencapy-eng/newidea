import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDefaultLogPath,
  isSafeGitHubUrl,
  parseOwnerRepo,
  validateGitHubLogPath,
  validateGitHubPushTarget,
} from './githubValidation';

test('parseOwnerRepo accepts owner/repo and GitHub URLs while rejecting malformed inputs', () => {
  assert.deepEqual(parseOwnerRepo('octocat/hand-eye-logs'), {
    owner: 'octocat',
    repo: 'hand-eye-logs',
  });
  assert.deepEqual(parseOwnerRepo('https://github.com/octocat/hand-eye-logs.git'), {
    owner: 'octocat',
    repo: 'hand-eye-logs',
  });
  assert.equal(parseOwnerRepo('invalid-only-one'), null);
  assert.equal(parseOwnerRepo('../evil/repo'), null);
  assert.equal(parseOwnerRepo('owner/..'), null);
});

test('validateGitHubLogPath blocks path traversal and disallowed extensions', () => {
  assert.equal(validateGitHubLogPath('logs/v3_panda_run.json'), 'logs/v3_panda_run.json');
  assert.equal(validateGitHubLogPath('/logs/run.jsonl'), 'logs/run.jsonl');
  assert.equal(validateGitHubLogPath('../secrets.json'), null);
  assert.equal(validateGitHubLogPath('logs/../../etc/passwd.json'), null);
  assert.equal(validateGitHubLogPath('logs/script.sh'), null);
  assert.equal(validateGitHubLogPath('logs/bad file name.json'), null);
});

test('validateGitHubPushTarget and isSafeGitHubUrl enforce strict allowlists', () => {
  const valid = validateGitHubPushTarget({
    repoFullName: 'lab-user/calib-traces',
    branch: 'main',
    filePath: buildDefaultLogPath({ algoVersion: 'v3', robotId: 'panda', kind: 'single', format: 'jsonl' }),
  });
  assert.equal(valid.ok, true);
  assert.equal(valid.owner, 'lab-user');

  const invalid = validateGitHubPushTarget({
    repoFullName: 'lab-user/calib-traces',
    branch: 'main/..',
    filePath: 'logs/run.json',
  });
  assert.equal(invalid.ok, false);

  assert.equal(isSafeGitHubUrl('https://github.com/octocat/repo/blob/main/logs/a.json'), true);
  assert.equal(isSafeGitHubUrl('https://raw.githubusercontent.com/octocat/repo/main/logs/a.json'), true);
  assert.equal(isSafeGitHubUrl('javascript:alert(1)'), false);
  assert.equal(isSafeGitHubUrl('https://evil.example.com/github.com'), false);
});
