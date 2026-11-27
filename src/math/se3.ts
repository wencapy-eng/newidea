/**
 * Exact SO(3) / SE(3) Lie group algebra, 3x3 SVD, 6x6 Jacobi eigensolver,
 * Gaussian elimination with partial pivoting, 3D spatial hash grid, and deterministic PRNG.
 *
 * Twist ordering convention throughout this codebase:
 *   xi = [phi(3), rho(3)] = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
 * Adjoint matrix:
 *   Ad_T = [ R,       0 ]
 *          [ [t]_x R, R ]
 */

export type Vec3 = [number, number, number];
export type Vec6 = [number, number, number, number, number, number];
/** Row-major 3x3 matrix (length 9) */
export type Mat3 = [
  number, number, number,
  number, number, number,
  number, number, number
];
/** Row-major 6x6 matrix (length 36) */
export type Mat6 = Float64Array;

export interface SE3 {
  R: Mat3;
  t: Vec3;
}

export function identityMat3(): Mat3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

export function identitySE3(): SE3 {
  return { R: identityMat3(), t: [0, 0, 0] };
}

export function cloneSE3(T: SE3): SE3 {
  return {
    R: [T.R[0], T.R[1], T.R[2], T.R[3], T.R[4], T.R[5], T.R[6], T.R[7], T.R[8]],
    t: [T.t[0], T.t[1], T.t[2]],
  };
}

export function dot3(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function norm3(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalize3(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]);
  if (n < 1e-15) return [0, 0, 0];
  return [a[0] / n, a[1] / n, a[2] / n];
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale3(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function skew(v: Vec3): Mat3 {
  return [
    0, -v[2], v[1],
    v[2], 0, -v[0],
    -v[1], v[0], 0,
  ];
}

export function mat3Mul(A: Mat3, B: Mat3): Mat3 {
  return [
    A[0] * B[0] + A[1] * B[3] + A[2] * B[6],
    A[0] * B[1] + A[1] * B[4] + A[2] * B[7],
    A[0] * B[2] + A[1] * B[5] + A[2] * B[8],

    A[3] * B[0] + A[4] * B[3] + A[5] * B[6],
    A[3] * B[1] + A[4] * B[4] + A[5] * B[7],
    A[3] * B[2] + A[4] * B[5] + A[5] * B[8],

    A[6] * B[0] + A[7] * B[3] + A[8] * B[6],
    A[6] * B[1] + A[7] * B[4] + A[8] * B[7],
    A[6] * B[2] + A[7] * B[5] + A[8] * B[8],
  ];
}

export function mat3Transpose(A: Mat3): Mat3 {
  return [
    A[0], A[3], A[6],
    A[1], A[4], A[7],
    A[2], A[5], A[8],
  ];
}

export function mat3MulVec(A: Mat3, v: Vec3): Vec3 {
  return [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
  ];
}

export function mat3TransposeMulVec(A: Mat3, v: Vec3): Vec3 {
  return [
    A[0] * v[0] + A[3] * v[1] + A[6] * v[2],
    A[1] * v[0] + A[4] * v[1] + A[7] * v[2],
    A[2] * v[0] + A[5] * v[1] + A[8] * v[2],
  ];
}

export function mat3Det(A: Mat3): number {
  return (
    A[0] * (A[4] * A[8] - A[5] * A[7]) -
    A[1] * (A[3] * A[8] - A[5] * A[6]) +
    A[2] * (A[3] * A[7] - A[4] * A[6])
  );
}

export function composeSE3(A: SE3, B: SE3): SE3 {
  const R = mat3Mul(A.R, B.R);
  const Rb = mat3MulVec(A.R, B.t);
  return {
    R,
    t: [Rb[0] + A.t[0], Rb[1] + A.t[1], Rb[2] + A.t[2]],
  };
}

export function invertSE3(T: SE3): SE3 {
  const Rt = mat3Transpose(T.R);
  const negRtt = mat3MulVec(Rt, T.t);
  return {
    R: Rt,
    t: [-negRtt[0], -negRtt[1], -negRtt[2]],
  };
}

export function transformPoint(T: SE3, p: Vec3): Vec3 {
  return [
    T.R[0] * p[0] + T.R[1] * p[1] + T.R[2] * p[2] + T.t[0],
    T.R[3] * p[0] + T.R[4] * p[1] + T.R[5] * p[2] + T.t[1],
    T.R[6] * p[0] + T.R[7] * p[1] + T.R[8] * p[2] + T.t[2],
  ];
}

export function inverseTransformPoint(T: SE3, p: Vec3): Vec3 {
  const dx = p[0] - T.t[0];
  const dy = p[1] - T.t[1];
  const dz = p[2] - T.t[2];
  return [
    T.R[0] * dx + T.R[3] * dy + T.R[6] * dz,
    T.R[1] * dx + T.R[4] * dy + T.R[7] * dz,
    T.R[2] * dx + T.R[5] * dy + T.R[8] * dz,
  ];
}

/** Rodrigues formula: exp([phi]_x) */
export function expSO3(phi: Vec3): Mat3 {
  const theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  const theta = Math.sqrt(theta2);
  let A: number;
  let B: number;
  if (theta < 1e-6) {
    A = 1 - theta2 / 6 + (theta2 * theta2) / 120;
    B = 0.5 - theta2 / 24 + (theta2 * theta2) / 720;
  } else {
    A = Math.sin(theta) / theta;
    B = (1 - Math.cos(theta)) / theta2;
  }
  const K = skew(phi);
  const K2 = mat3Mul(K, K);
  const I = identityMat3();
  const out = new Array(9) as Mat3;
  for (let i = 0; i < 9; i++) {
    out[i] = I[i] + A * K[i] + B * K2[i];
  }
  return out;
}

/** Exact logarithm map SO(3) -> so(3), robust near 0 and pi */
export function logSO3(R: Mat3): Vec3 {
  const trace = R[0] + R[4] + R[8];
  const cosTheta = Math.max(-1, Math.min(1, 0.5 * (trace - 1)));
  const theta = Math.acos(cosTheta);

  if (theta < 1e-6) {
    const scale = 0.5 * (1 + (theta * theta) / 6 + (7 * theta * theta * theta * theta) / 360);
    return [
      scale * (R[7] - R[5]),
      scale * (R[2] - R[6]),
      scale * (R[3] - R[1]),
    ];
  }

  if (Math.PI - theta < 1e-6) {
    // Near pi: extract unit axis from symmetric part (R + I)/2 = u u^T
    const d0 = Math.max(0, 0.5 * (R[0] + 1));
    const d1 = Math.max(0, 0.5 * (R[4] + 1));
    const d2 = Math.max(0, 0.5 * (R[8] + 1));
    let ax = 0;
    let ay = 0;
    let az = 0;
    if (d0 >= d1 && d0 >= d2) {
      ax = Math.sqrt(d0);
      ay = (R[1] + R[3]) / (4 * ax);
      az = (R[2] + R[6]) / (4 * ax);
    } else if (d1 >= d0 && d1 >= d2) {
      ay = Math.sqrt(d1);
      ax = (R[1] + R[3]) / (4 * ay);
      az = (R[5] + R[7]) / (4 * ay);
    } else {
      az = Math.sqrt(d2);
      ax = (R[2] + R[6]) / (4 * az);
      ay = (R[5] + R[7]) / (4 * az);
    }
    const u = normalize3([ax, ay, az]);
    const svec: Vec3 = [R[7] - R[5], R[2] - R[6], R[3] - R[1]];
    const sign = dot3(u, svec) < -1e-12 ? -1 : 1;
    return [sign * theta * u[0], sign * theta * u[1], sign * theta * u[2]];
  }

  const factor = theta / (2 * Math.sin(theta));
  return [
    factor * (R[7] - R[5]),
    factor * (R[2] - R[6]),
    factor * (R[3] - R[1]),
  ];
}

/** Exact SO(3) Left Jacobian J_l(phi) */
export function leftJacobianSO3(phi: Vec3): Mat3 {
  const theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  const theta = Math.sqrt(theta2);
  let c1: number;
  let c2: number;
  if (theta < 1e-5) {
    c1 = 0.5 - theta2 / 24 + (theta2 * theta2) / 720;
    c2 = 1 / 6 - theta2 / 120 + (theta2 * theta2) / 5040;
  } else {
    c1 = (1 - Math.cos(theta)) / theta2;
    c2 = (theta - Math.sin(theta)) / (theta2 * theta);
  }
  const K = skew(phi);
  const K2 = mat3Mul(K, K);
  const I = identityMat3();
  const out = new Array(9) as Mat3;
  for (let i = 0; i < 9; i++) {
    out[i] = I[i] + c1 * K[i] + c2 * K2[i];
  }
  return out;
}

/** Exact inverse SO(3) Left Jacobian J_l^{-1}(phi) */
export function leftJacobianInvSO3(phi: Vec3): Mat3 {
  const theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
  const theta = Math.sqrt(theta2);
  let c2: number;
  if (theta < 1e-5) {
    c2 = 1 / 12 + theta2 / 720 + (theta2 * theta2) / 30240;
  } else {
    const half = 0.5 * theta;
    const cotHalf = Math.cos(half) / Math.sin(half);
    c2 = (1 / theta2) * (1 - half * cotHalf);
  }
  const K = skew(phi);
  const K2 = mat3Mul(K, K);
  const I = identityMat3();
  const out = new Array(9) as Mat3;
  for (let i = 0; i < 9; i++) {
    out[i] = I[i] - 0.5 * K[i] + c2 * K2[i];
  }
  return out;
}

/** Exact SE(3) exponential map: xi = [phi(3), rho(3)] */
export function expSE3(xi: Vec6): SE3 {
  const phi: Vec3 = [xi[0], xi[1], xi[2]];
  const rho: Vec3 = [xi[3], xi[4], xi[5]];
  const R = expSO3(phi);
  const Jl = leftJacobianSO3(phi);
  const t = mat3MulVec(Jl, rho);
  return { R, t };
}

/** Exact SE(3) logarithm map returning [phi(3), rho(3)] */
export function logSE3(T: SE3): Vec6 {
  const phi = logSO3(T.R);
  const JlInv = leftJacobianInvSO3(phi);
  const rho = mat3MulVec(JlInv, T.t);
  return [phi[0], phi[1], phi[2], rho[0], rho[1], rho[2]];
}

/** 6x6 Adjoint matrix Ad_T = [ R, 0; [t]_x R, R ] (row-major length 36) */
export function adjointSE3(T: SE3): Mat6 {
  const out = new Float64Array(36);
  const txR = mat3Mul(skew(T.t), T.R);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const valR = T.R[r * 3 + c];
      out[r * 6 + c] = valR;
      out[(r + 3) * 6 + (c + 3)] = valR;
      out[(r + 3) * 6 + c] = txR[r * 3 + c];
    }
  }
  return out;
}

export function applyAdjoint(T: SE3, xi: Vec6): Vec6 {
  const w: Vec3 = [xi[0], xi[1], xi[2]];
  const v: Vec3 = [xi[3], xi[4], xi[5]];
  const Rw = mat3MulVec(T.R, w);
  const Rv = mat3MulVec(T.R, v);
  const tCrossRw = cross3(T.t, Rw);
  return [
    Rw[0], Rw[1], Rw[2],
    tCrossRw[0] + Rv[0], tCrossRw[1] + Rv[1], tCrossRw[2] + Rv[2],
  ];
}

/** Geodesic rotation error (deg) and Euclidean translation error (mm) between T_est and T_gt */
export function computePoseError(T_est: SE3, T_gt: SE3): { rotDeg: number; transMm: number } {
  const R_err = mat3Mul(mat3Transpose(T_gt.R), T_est.R);
  const phi = logSO3(R_err);
  const rotDeg = (norm3(phi) * 180) / Math.PI;
  // Compare camera position in base frame or t_cb directly; both are equal in Euclidean norm
  // Wait: ||t_est - t_gt|| vs ||p_cam_est - p_cam_gt||. Let's check:
  // p_cam = -R_cb^T t_cb. Usually hand-eye error reports ||t_est - t_gt|| (in camera frame)
  // OR ||p_cam_est - p_cam_gt||. Note that for small rotation error they are virtually identical;
  // standard hand-eye benchmark uses ||t_est - t_gt|| directly on T_cb.
  const dt = sub3(T_est.t, T_gt.t);
  const transMm = norm3(dt) * 1000;
  return { rotDeg, transMm };
}

/**
 * Symmetric n x n Jacobi eigensolver (returns eigenvalues sorted descending + orthonormal eigenvectors as columns).
 * Works cleanly for 3x3 (SVD) and 6x6 (FIM log det & lambda_min).
 */
export function jacobiEigenSymmetric(
  mat: ArrayLike<number>,
  n: number,
  maxSweeps = 60
): { eigenvalues: Float64Array; eigenvectors: Float64Array } {
  const A = new Float64Array(n * n);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    V[i * n + i] = 1;
    for (let j = 0; j < n; j++) {
      // Symmetrize input
      A[i * n + j] = 0.5 * (mat[i * n + j] + mat[j * n + i]);
    }
  }

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let maxOff = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        const absApq = Math.abs(apq);
        if (absApq > maxOff) maxOff = absApq;
        if (absApq < 1e-15) continue;

        const app = A[p * n + p];
        const aqq = A[q * n + q];
        const tau = (aqq - app) / (2 * apq);
        const t = tau >= 0
          ? 1 / (tau + Math.sqrt(1 + tau * tau))
          : -1 / (-tau + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        A[p * n + p] = app - t * apq;
        A[q * n + q] = aqq + t * apq;
        A[p * n + q] = 0;
        A[q * n + p] = 0;

        for (let r = 0; r < n; r++) {
          if (r === p || r === q) continue;
          const arp = A[r * n + p];
          const arq = A[r * n + q];
          const newArp = c * arp - s * arq;
          const newArq = s * arp + c * arq;
          A[r * n + p] = newArp;
          A[p * n + r] = newArp;
          A[r * n + q] = newArq;
          A[q * n + r] = newArq;
        }

        for (let r = 0; r < n; r++) {
          const vrp = V[r * n + p];
          const vrq = V[r * n + q];
          V[r * n + p] = c * vrp - s * vrq;
          V[r * n + q] = s * vrp + c * vrq;
        }
      }
    }
    if (maxOff < 1e-13) break;
  }

  const order = Array.from({ length: n }, (_, i) => i).sort(
    (i, j) => A[j * n + j] - A[i * n + i]
  );
  const eigenvalues = new Float64Array(n);
  const eigenvectors = new Float64Array(n * n);
  for (let col = 0; col < n; col++) {
    const src = order[col];
    eigenvalues[col] = A[src * n + src];
    for (let row = 0; row < n; row++) {
      eigenvectors[row * n + col] = V[row * n + src];
    }
  }
  return { eigenvalues, eigenvectors };
}

/** 6x6 symmetric matrix Jacobi eigenvalues (descending) for FIM D/E-optimality */
export function jacobiEigen6x6(M: Float64Array): {
  eigenvalues: Float64Array;
  eigenvectors: Float64Array;
} {
  return jacobiEigenSymmetric(M, 6, 60);
}

/**
 * Exact 3x3 SVD: H = U * diag(S) * V^T using Jacobi eigen-decomposition of H^T H.
 * Handles rank-2 and rank-1 degenerate cases cleanly (critical for Wahba problem).
 */
export function svd3x3(H: Mat3): { U: Mat3; S: Vec3; V: Mat3 } {
  const Ht = mat3Transpose(H);
  const HtH = mat3Mul(Ht, H);
  const { eigenvalues, eigenvectors } = jacobiEigenSymmetric(HtH, 3, 50);

  // Columns of V are right singular vectors
  const v0: Vec3 = normalize3([eigenvectors[0], eigenvectors[3], eigenvectors[6]]);
  const v1: Vec3 = normalize3([eigenvectors[1], eigenvectors[4], eigenvectors[7]]);
  let v2: Vec3 = normalize3([eigenvectors[2], eigenvectors[5], eigenvectors[8]]);

  // Ensure right-handed orthonormal V
  const v0xv1 = cross3(v0, v1);
  if (dot3(v0xv1, v2) < 0) {
    v2 = [-v2[0], -v2[1], -v2[2]];
  }
  const V: Mat3 = [
    v0[0], v1[0], v2[0],
    v0[1], v1[1], v2[1],
    v0[2], v1[2], v2[2],
  ];

  const s0 = Math.sqrt(Math.max(0, eigenvalues[0]));
  const s1 = Math.sqrt(Math.max(0, eigenvalues[1]));
  const s2 = Math.sqrt(Math.max(0, eigenvalues[2]));

  const eps = Math.max(1e-12, s0 * 1e-10);
  let u0: Vec3;
  let u1: Vec3;
  let u2: Vec3;

  if (s0 > eps) {
    const hv0 = mat3MulVec(H, v0);
    u0 = normalize3(hv0);
  } else {
    u0 = [1, 0, 0];
  }

  if (s1 > eps) {
    const hv1 = mat3MulVec(H, v1);
    // Gram-Schmidt orthogonalization against u0
    const proj = dot3(hv1, u0);
    u1 = normalize3([hv1[0] - proj * u0[0], hv1[1] - proj * u0[1], hv1[2] - proj * u0[2]]);
  } else {
    // Pick arbitrary vector orthogonal to u0
    const ref: Vec3 = Math.abs(u0[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    u1 = normalize3(cross3(u0, ref));
  }

  if (s2 > eps) {
    const hv2 = mat3MulVec(H, v2);
    const p0 = dot3(hv2, u0);
    const p1 = dot3(hv2, u1);
    const ortho: Vec3 = [
      hv2[0] - p0 * u0[0] - p1 * u1[0],
      hv2[1] - p0 * u0[1] - p1 * u1[1],
      hv2[2] - p0 * u0[2] - p1 * u1[2],
    ];
    if (norm3(ortho) > 1e-10) {
      u2 = normalize3(ortho);
    } else {
      u2 = normalize3(cross3(u0, u1));
    }
  } else {
    // Rank-2 or rank-1 degenerate completion: u2 = u0 x u1 (orthogonal complement)
    u2 = normalize3(cross3(u0, u1));
  }

  // Re-verify sign of u2 against H * v2 when s2 > eps
  if (s2 > eps) {
    const hv2 = mat3MulVec(H, v2);
    if (dot3(u2, hv2) < 0) {
      u2 = [-u2[0], -u2[1], -u2[2]];
    }
  }

  const U: Mat3 = [
    u0[0], u1[0], u2[0],
    u0[1], u1[1], u2[1],
    u0[2], u1[2], u2[2],
  ];

  return { U, S: [s0, s1, s2], V };
}

/**
 * Solve n x n linear system A x = b using Gaussian elimination with partial pivoting.
 * Returns null if singular.
 */
export function solveLinearSystem(A_in: ArrayLike<number>, b_in: ArrayLike<number>, n: number): Float64Array | null {
  const M = new Float64Array(n * (n + 1));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      M[r * (n + 1) + c] = A_in[r * n + c];
    }
    M[r * (n + 1) + n] = b_in[r];
  }

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let pivotVal = Math.abs(M[col * (n + 1) + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r * (n + 1) + col]);
      if (v > pivotVal) {
        pivotVal = v;
        pivotRow = r;
      }
    }
    if (pivotVal < 1e-14) return null;

    if (pivotRow !== col) {
      for (let c = col; c <= n; c++) {
        const tmp = M[col * (n + 1) + c];
        M[col * (n + 1) + c] = M[pivotRow * (n + 1) + c];
        M[pivotRow * (n + 1) + c] = tmp;
      }
    }

    const diag = M[col * (n + 1) + col];
    for (let r = col + 1; r < n; r++) {
      const factor = M[r * (n + 1) + col] / diag;
      if (factor === 0) continue;
      M[r * (n + 1) + col] = 0;
      for (let c = col + 1; c <= n; c++) {
        M[r * (n + 1) + c] -= factor * M[col * (n + 1) + c];
      }
    }
  }

  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let sum = M[r * (n + 1) + n];
    for (let c = r + 1; c < n; c++) {
      sum -= M[r * (n + 1) + c] * x[c];
    }
    x[r] = sum / M[r * (n + 1) + r];
  }
  return x;
}

/** Deterministic PRNG (Mulberry32) + Box-Muller Gaussian for reproducible paired Monte Carlo trials */
export class SeededRNG {
  private state: number;
  private spare: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  uniform(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  gaussian(mean = 0, std = 1): number {
    if (this.spare !== null) {
      const val = this.spare;
      this.spare = null;
      return mean + val * std;
    }
    let u = 0;
    let v = 0;
    while (u <= 1e-12) u = this.next();
    v = this.next();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    const z0 = mag * Math.cos(2.0 * Math.PI * v);
    const z1 = mag * Math.sin(2.0 * Math.PI * v);
    this.spare = z1;
    return mean + z0 * std;
  }
}

export interface PointNormalSample {
  p: Vec3;
  n: Vec3;
}

/** Fast 3D Spatial Hash Grid for nearest-neighbor queries in ICP, SOR, and Stage 2 CAD lookup */
export class SpatialHashGrid3D<T extends { p: Vec3 }> {
  private readonly invCell: number;
  private readonly buckets = new Map<number, T[]>();

  constructor(cellSize: number) {
    this.invCell = 1 / cellSize;
  }

  private hash(ix: number, iy: number, iz: number): number {
    // 32-bit integer spatial hash with large primes
    return ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
  }

  insert(item: T): void {
    const ix = Math.floor(item.p[0] * this.invCell);
    const iy = Math.floor(item.p[1] * this.invCell);
    const iz = Math.floor(item.p[2] * this.invCell);
    const key = this.hash(ix, iy, iz);
    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = [];
      this.buckets.set(key, bucket);
    }
    bucket.push(item);
  }

  insertAll(items: readonly T[]): void {
    for (let i = 0; i < items.length; i++) {
      this.insert(items[i]);
    }
  }

  findNearest(
    query: Vec3,
    maxDist: number,
    filter?: (cand: T) => boolean
  ): { item: T; distSq: number } | null {
    const cx = Math.floor(query[0] * this.invCell);
    const cy = Math.floor(query[1] * this.invCell);
    const cz = Math.floor(query[2] * this.invCell);
    const radiusCells = Math.max(1, Math.ceil(maxDist * this.invCell));
    let bestDistSq = maxDist * maxDist;
    let bestItem: T | null = null;

    for (let dx = -radiusCells; dx <= radiusCells; dx++) {
      for (let dy = -radiusCells; dy <= radiusCells; dy++) {
        for (let dz = -radiusCells; dz <= radiusCells; dz++) {
          const bucket = this.buckets.get(this.hash(cx + dx, cy + dy, cz + dz));
          if (!bucket) continue;
          for (let i = 0; i < bucket.length; i++) {
            const cand = bucket[i];
            const ex = cand.p[0] - query[0];
            const ey = cand.p[1] - query[1];
            const ez = cand.p[2] - query[2];
            const d2 = ex * ex + ey * ey + ez * ez;
            if (d2 < bestDistSq) {
              if (filter && !filter(cand)) continue;
              bestDistSq = d2;
              bestItem = cand;
            }
          }
        }
      }
    }

    return bestItem ? { item: bestItem, distSq: bestDistSq } : null;
  }

  findWithinRadius(query: Vec3, radius: number): T[] {
    const cx = Math.floor(query[0] * this.invCell);
    const cy = Math.floor(query[1] * this.invCell);
    const cz = Math.floor(query[2] * this.invCell);
    const radiusCells = Math.max(1, Math.ceil(radius * this.invCell));
    const r2 = radius * radius;
    const result: T[] = [];

    for (let dx = -radiusCells; dx <= radiusCells; dx++) {
      for (let dy = -radiusCells; dy <= radiusCells; dy++) {
        for (let dz = -radiusCells; dz <= radiusCells; dz++) {
          const bucket = this.buckets.get(this.hash(cx + dx, cy + dy, cz + dz));
          if (!bucket) continue;
          for (let i = 0; i < bucket.length; i++) {
            const cand = bucket[i];
            const ex = cand.p[0] - query[0];
            const ey = cand.p[1] - query[1];
            const ez = cand.p[2] - query[2];
            if (ex * ex + ey * ey + ez * ez <= r2) {
              result.push(cand);
            }
          }
        }
      }
    }
    return result;
  }
}

/** Convert CV camera pose T_cb (p^c = R_cb p^b + t_cb) into camera position in base frame {B} */
export function cameraCenterInBase(T_cb: SE3): Vec3 {
  const negRtt = mat3TransposeMulVec(T_cb.R, T_cb.t);
  return [-negRtt[0], -negRtt[1], -negRtt[2]];
}

/** Build a valid CV extrinsic T_cb looking from camPosB toward targetB with world Z-up (+Z_b) */
export function lookAtCV(camPosB: Vec3, targetB: Vec3, rollRad = 0): SE3 {
  // CV +Z_c points from camera to target
  const zc = normalize3(sub3(targetB, camPosB));
  const worldUp: Vec3 = [0, 0, 1];
  // CV +X_c points right: zc x worldUp
  let xc = cross3(zc, worldUp);
  if (norm3(xc) < 1e-6) {
    xc = normalize3(cross3(zc, [1, 0, 0]));
  } else {
    xc = normalize3(xc);
  }
  // CV +Y_c points down: zc x xc (right-handed: xc x yc = zc)
  let yc = normalize3(cross3(zc, xc));

  if (Math.abs(rollRad) > 1e-9) {
    const cr = Math.cos(rollRad);
    const sr = Math.sin(rollRad);
    const xcNew: Vec3 = [
      cr * xc[0] + sr * yc[0],
      cr * xc[1] + sr * yc[1],
      cr * xc[2] + sr * yc[2],
    ];
    const ycNew: Vec3 = [
      -sr * xc[0] + cr * yc[0],
      -sr * xc[1] + cr * yc[1],
      -sr * xc[2] + cr * yc[2],
    ];
    xc = xcNew;
    yc = ycNew;
  }

  // Rows of R_cb are xc^T, yc^T, zc^T
  const R: Mat3 = [
    xc[0], xc[1], xc[2],
    yc[0], yc[1], yc[2],
    zc[0], zc[1], zc[2],
  ];
  const Rpos = mat3MulVec(R, camPosB);
  return {
    R,
    t: [-Rpos[0], -Rpos[1], -Rpos[2]],
  };
}
