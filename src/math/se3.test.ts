import test from 'node:test';
import assert from 'node:assert/strict';
import {
  adjointSE3,
  applyAdjoint,
  composeSE3,
  computePoseError,
  expSE3,
  expSO3,
  invertSE3,
  jacobiEigen6x6,
  leftJacobianInvSO3,
  leftJacobianSO3,
  logSE3,
  logSO3,
  mat3Det,
  mat3Mul,
  mat3Transpose,
  solveLinearSystem,
  svd3x3,
  type Mat3,
  type Vec3,
  type Vec6,
} from './se3';

function assertNear(a: number, b: number, tol = 1e-9, msg?: string) {
  assert.ok(Math.abs(a - b) <= tol, `${msg ?? 'mismatch'}: ${a} vs ${b} (diff=${Math.abs(a - b)})`);
}

test('expSO3 and logSO3 round-trip across small, moderate, and near-pi angles', () => {
  const cases: Vec3[] = [
    [1e-8, -2e-8, 3e-8],
    [0.35, -0.72, 0.51],
    [1.4, 0.8, -0.9],
    [Math.PI - 5e-5, 0, 0],
    [0, -(Math.PI - 2e-5), 0],
  ];

  for (const phi of cases) {
    const R = expSO3(phi);
    assertNear(mat3Det(R), 1, 1e-9, 'det(R) must be 1');
    const RtR = mat3Mul(mat3Transpose(R), R);
    for (let i = 0; i < 9; i++) {
      assertNear(RtR[i], i % 4 === 0 ? 1 : 0, 1e-9, 'R^T R = I');
    }
    const back = logSO3(R);
    const Rback = expSO3(back);
    for (let i = 0; i < 9; i++) {
      assertNear(Rback[i], R[i], 1e-6, 'exp(log(R)) = R');
    }
  }
});

test('leftJacobianSO3 * leftJacobianInvSO3 equals identity', () => {
  const phis: Vec3[] = [
    [0, 0, 0],
    [1e-6, -2e-6, 1e-6],
    [0.42, -0.88, 1.15],
    [-1.2, 1.5, -0.7],
  ];
  for (const phi of phis) {
    const Jl = leftJacobianSO3(phi);
    const JlInv = leftJacobianInvSO3(phi);
    const prod = mat3Mul(Jl, JlInv);
    for (let i = 0; i < 9; i++) {
      assertNear(prod[i], i % 4 === 0 ? 1 : 0, 1e-9, 'Jl * Jl^-1 = I');
    }
  }
});

test('expSE3 / logSE3 round-trip and Adjoint conjugation identity', () => {
  const xi: Vec6 = [0.45, -0.62, 0.31, 0.25, -0.4, 1.15];
  const T = expSE3(xi);
  const back = logSE3(T);
  for (let i = 0; i < 6; i++) {
    assertNear(back[i], xi[i], 1e-9, `logSE3(expSE3(xi))[${i}]`);
  }

  // Adjoint conjugation: T * exp(eta) * T^-1 == exp(Ad_T * eta)
  const eta: Vec6 = [-0.21, 0.34, -0.18, 0.12, 0.08, -0.19];
  const conj = composeSE3(composeSE3(T, expSE3(eta)), invertSE3(T));
  const adEta = applyAdjoint(T, eta);
  const expAdEta = expSE3(adEta);
  const err = computePoseError(conj, expAdEta);
  assert.ok(err.rotDeg < 1e-9 && err.transMm < 1e-9, `Adjoint conjugation error: ${JSON.stringify(err)}`);

  // Matrix form adjointSE3 matches applyAdjoint
  const Ad = adjointSE3(T);
  for (let r = 0; r < 6; r++) {
    let s = 0;
    for (let c = 0; c < 6; c++) s += Ad[r * 6 + c] * eta[c];
    assertNear(s, adEta[r], 1e-12, `Ad matrix row ${r}`);
  }
});

test('svd3x3 reconstructs full-rank and rank-2 degenerate matrices accurately', () => {
  const fullRank: Mat3 = [
    1.2, -0.5, 0.8,
    0.3, 2.1, -0.4,
    -0.9, 0.7, 1.5,
  ];
  // Rank-2 matrix (outer product of two orthogonal vectors)
  const rank2: Mat3 = [
    1, 0, 0,
    0, 2, 0,
    0, 0, 0,
  ];
  const R = expSO3([0.4, -0.7, 0.2]);
  const rotatedRank2 = mat3Mul(R, rank2);

  for (const M of [fullRank, rank2, rotatedRank2]) {
    const { U, S, V } = svd3x3(M);
    const diagS: Mat3 = [S[0], 0, 0, 0, S[1], 0, 0, 0, S[2]];
    const recon = mat3Mul(mat3Mul(U, diagS), mat3Transpose(V));
    for (let i = 0; i < 9; i++) {
      assertNear(recon[i], M[i], 1e-8, `SVD reconstruction[${i}]`);
    }
    const UtU = mat3Mul(mat3Transpose(U), U);
    const VtV = mat3Mul(mat3Transpose(V), V);
    for (let i = 0; i < 9; i++) {
      assertNear(UtU[i], i % 4 === 0 ? 1 : 0, 1e-8, 'U orthonormal');
      assertNear(VtV[i], i % 4 === 0 ? 1 : 0, 1e-8, 'V orthonormal');
    }
  }
});

test('jacobiEigen6x6 and solveLinearSystem match exact analytical values', () => {
  // Construct symmetric 6x6 PSD matrix M = sum v_k v_k^T with known spectrum
  const diag = [12, 8, 5, 3, 1.5, 0.25];
  const M = new Float64Array(36);
  for (let i = 0; i < 6; i++) M[i * 6 + i] = diag[i];
  // Apply simple orthogonal similarity (Givens rotation in (0, 3) and (1, 5))
  const applyGivens = (A: Float64Array, p: number, q: number, ang: number) => {
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const B = new Float64Array(A);
    for (let r = 0; r < 6; r++) {
      B[r * 6 + p] = c * A[r * 6 + p] - s * A[r * 6 + q];
      B[r * 6 + q] = s * A[r * 6 + p] + c * A[r * 6 + q];
    }
    const C = new Float64Array(B);
    for (let col = 0; col < 6; col++) {
      C[p * 6 + col] = c * B[p * 6 + col] - s * B[q * 6 + col];
      C[q * 6 + col] = s * B[p * 6 + col] + c * B[q * 6 + col];
    }
    return C;
  };
  const rotated = applyGivens(applyGivens(M, 0, 3, 0.7), 1, 5, -0.45);
  const { eigenvalues } = jacobiEigen6x6(rotated);
  for (let i = 0; i < 6; i++) {
    assertNear(eigenvalues[i], diag[i], 1e-8, `eigenvalue[${i}]`);
  }

  // Test 6x6 Gaussian elimination with partial pivoting
  const xTrue = [1.5, -2.2, 0.7, 3.1, -0.4, 2.8];
  const b = new Float64Array(6);
  for (let r = 0; r < 6; r++) {
    let s = 0;
    for (let c = 0; c < 6; c++) s += rotated[r * 6 + c] * xTrue[c];
    b[r] = s;
  }
  const xSol = solveLinearSystem(rotated, b, 6);
  assert.ok(xSol !== null);
  for (let i = 0; i < 6; i++) {
    assertNear(xSol[i], xTrue[i], 1e-9, `linear solve[${i}]`);
  }
});
