import React from 'react';
import type { AblationRow, AlgoVersion, CalibrationResult, ExtrinsicInitMode } from '../calib/types';

interface AblationTableProps {
  algoVersion?: AlgoVersion;
  trueAsyncDelayMs?: number;
  initMode?: ExtrinsicInitMode;
  previewInitErr?: { rotDeg: number; transMm: number } | null;
  singleResult: CalibrationResult | null;
  ablationRows: AblationRow[];
  isRunningBenchmark: boolean;
}

const INIT_CARD_META: Record<ExtrinsicInitMode, { title: string; sub: string }> = {
  stage1: { title: 'Stage 1 螺旋轴闭式解', sub: 'Wahba SVD + Plücker WLS' },
  controlled_random: { title: '初始外参 · 受控随机', sub: '真值可控 SE(3) 扰动' },
  full_random: { title: '初始外参 · 完全随机', sub: 'SO(3)×R³ 全空间盲采样' },
  zero: { title: '初始外参 · 0 初始化', sub: 'ξ⁽⁰⁾ = 0 · R = I₃, t = 0' },
};

export const AblationTable: React.FC<AblationTableProps> = ({
  algoVersion = 'v2',
  trueAsyncDelayMs = 25,
  initMode = 'stage1',
  previewInitErr,
  singleResult,
  ablationRows,
  isRunningBenchmark,
}) => {
  const effectiveInitMode = singleResult?.initMode ?? initMode;
  const isV3 = (singleResult?.algoVersion ?? algoVersion) === 'v3';
  const cardMeta =
    effectiveInitMode === 'stage1' && isV3
      ? { title: 'Stage 2 盲探针闭式解', sub: 'GNC-Welsch + 广义螺旋轴' }
      : INIT_CARD_META[effectiveInitMode];
  const shownInitTrans = singleResult
    ? singleResult.initTransMm
    : previewInitErr
      ? previewInitErr.transMm
      : null;
  const shownInitRot = singleResult
    ? singleResult.initRotDeg
    : previewInitErr
      ? previewInitErr.rotDeg
      : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Quantitative Single Calibration Metric Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
        <div className="bg-[#10151F] border border-slate-800 rounded-md px-3 py-2.5 min-w-0">
          <div className="text-[11px] text-slate-400 truncate">{cardMeta.title}</div>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-1 gap-y-0.5 font-mono">
            <span
              data-testid="metric-init-trans"
              className="text-[15px] font-semibold text-amber-300"
            >
              {shownInitTrans !== null ? shownInitTrans.toFixed(1) : '—'}
            </span>
            <span className="text-[10.5px] text-slate-400">mm</span>
            <span className="text-slate-600">/</span>
            <span
              data-testid="metric-init-rot"
              className="text-[12.5px] font-medium text-amber-200/90"
            >
              {shownInitRot !== null ? `${shownInitRot.toFixed(2)}°` : '—'}
            </span>
          </div>
          <div className="text-[10.5px] text-slate-500 mt-0.5 font-mono truncate">
            {cardMeta.sub}
          </div>
        </div>

        <div className="bg-[#10151F] border border-slate-800 rounded-md px-3 py-2.5 min-w-0">
          <div className="text-[11px] text-slate-400 truncate">最终标定误差 (平移 / 旋转)</div>
          <div className="mt-1 flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5 font-mono">
            <span
              data-testid="metric-final-trans"
              className="text-[16px] font-semibold text-cyan-300"
            >
              {singleResult ? singleResult.finalTransMm.toFixed(2) : '—'}
            </span>
            <span className="text-[11px] text-slate-400">mm</span>
            <span className="text-slate-600">/</span>
            <span
              data-testid="metric-final-rot"
              className="text-[13.5px] font-medium text-cyan-200/90"
            >
              {singleResult ? `${singleResult.finalRotDeg.toFixed(3)}°` : '—'}
            </span>
          </div>
          <div className="text-[10.5px] text-slate-500 mt-0.5 font-mono truncate">
            {singleResult
              ? singleResult.success
                ? '状态: 收敛合格 (<20mm & <2°)'
                : '状态: 误差偏大'
              : '未运行标定'}
          </div>
        </div>

        <div className="bg-[#10151F] border border-slate-800 rounded-md px-3.5 py-2.5">
          <div className="text-[11px] text-slate-400">累积 FIM 信息量 (log det)</div>
          <div className="mt-1 flex items-baseline gap-2 font-mono">
            <span
              data-testid="metric-fim-logdet"
              className="text-[18px] font-semibold text-slate-100"
            >
              {singleResult ? singleResult.fimLogDet.toFixed(2) : '—'}
            </span>
            <span className="text-[11px] text-slate-400">nats</span>
          </div>
          <div className="text-[10.5px] text-slate-500 mt-0.5 font-mono">
            λ_min = {singleResult ? singleResult.fimLambdaMin.toExponential(2) : '—'}
          </div>
        </div>

        <div className="bg-[#10151F] border border-slate-800 rounded-md px-3.5 py-2.5">
          <div className="text-[11px] text-slate-400">总计算与仿真耗时</div>
          <div className="mt-1 flex items-baseline gap-1.5 font-mono">
            <span className="text-[18px] font-semibold text-slate-100">
              {singleResult ? (singleResult.elapsedMs / 1000).toFixed(2) : '—'}
            </span>
            <span className="text-[11px] text-slate-400">s</span>
          </div>
          <div className="text-[10.5px] text-slate-500 mt-0.5 font-mono">
            {singleResult ? `${singleResult.executedActions.length} 步动作 × 16 帧` : '真实 WebGL2 + WASM'}
          </div>
        </div>
      </div>

      {/* V3.0 Physical Observation Decoupling & 7-DoF Time-Offset Strip */}
      {isV3 && singleResult?.v3Diagnostics && (
        <div
          data-testid="v3-diagnostics-strip"
          className="grid grid-cols-2 lg:grid-cols-4 gap-2 bg-[#0D121B] border border-cyan-500/30 rounded-md px-3 py-2 text-[11px] font-mono"
        >
          <div className="min-w-0">
            <div className="text-[10px] text-slate-400 font-sans">EM 观测类先验 (I / II / III)</div>
            <div className="mt-0.5 flex items-center gap-1.5 truncate">
              <span className="text-cyan-300">
                {(singleResult.v3Diagnostics.classPriors[0] * 100).toFixed(0)}%
              </span>
              <span className="text-slate-600">/</span>
              <span className="text-amber-300">
                {(singleResult.v3Diagnostics.classPriors[1] * 100).toFixed(0)}%
              </span>
              <span className="text-slate-600">/</span>
              <span className="text-pink-400">
                {(singleResult.v3Diagnostics.classPriors[2] * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div className="min-w-0">
            <div className="text-[10px] text-slate-400 font-sans">7-DoF 时间偏置辨识 δt̂</div>
            <div className="mt-0.5 text-emerald-300 truncate" data-testid="metric-v3-dt">
              {singleResult.v3Diagnostics.estimatedDelayMs !== null
                ? `${singleResult.v3Diagnostics.estimatedDelayMs.toFixed(1)} ms (真值 ${trueAsyncDelayMs}ms)`
                : '未开启'}
            </div>
          </div>

          <div className="min-w-0">
            <div className="text-[10px] text-slate-400 font-sans">Z-Buffer 静态遮挡剔除</div>
            <div className="mt-0.5 text-slate-200 truncate">
              {singleResult.v3Diagnostics.zBufferOccludedPct.toFixed(1)}% · M_occ{' '}
              {singleResult.v3Diagnostics.occ2DBandPixels}px
            </div>
          </div>

          <div className="min-w-0">
            <div className="text-[10px] text-slate-400 font-sans">盲探针 3 门控准入</div>
            <div className="mt-0.5 text-cyan-200 truncate">
              {singleResult.v3Diagnostics.probeLogs.filter((p) => p.accepted).length} /{' '}
              {singleResult.v3Diagnostics.probeLogs.length} 通过
            </div>
          </div>
        </div>
      )}

      {/* 6-Condition Paired Monte Carlo Ablation Benchmark Table */}
      <section className="bg-[#10151F] border border-slate-800 rounded-md overflow-hidden">
        <div className="flex items-center justify-between px-3.5 py-2 border-b border-slate-800">
          <h2 className="text-[12.5px] font-semibold text-slate-200">
            {isV3 ? 'V3.0 6 组配对 Monte Carlo 消融对比表' : 'V2.0 6 组配对 Monte Carlo 消融对比表'}
          </h2>
          <span className="text-[11px] font-mono text-slate-400">
            {ablationRows.length > 0
              ? `同真值相机位姿配对评测 · N = ${ablationRows[0].trials}`
              : isRunningBenchmark
                ? '正在执行配对消融基准...'
                : '点击顶部「一键跑 6 组消融实验」生成定量对比'}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse" data-testid="ablation-table">
            <thead>
              <tr className="border-b border-slate-800 text-[11px] text-slate-400 bg-[#0c1018]">
                <th className="py-2 px-3 font-medium">消融条件</th>
                <th className="py-2 px-2.5 font-medium text-right whitespace-nowrap">
                  旋转误差 mean±std (°)
                </th>
                <th className="py-2 px-2.5 font-medium text-right whitespace-nowrap">
                  平移误差 mean±std (mm)
                </th>
                <th className="py-2 px-2.5 font-medium text-right whitespace-nowrap">
                  中位平移 (mm)
                </th>
                <th className="py-2 px-2.5 font-medium text-right whitespace-nowrap">成功率</th>
                <th className="py-2 px-3 font-medium text-right whitespace-nowrap">耗时 (s)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70 text-[12px]">
              {ablationRows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-5 px-3 text-center text-slate-500 text-[12px]">
                    尚无消融实验记录。6 组条件将在完全相同的随机相机真值位姿上依次运行真实
                    MuJoCo + WebGL2 标定流程。
                  </td>
                </tr>
              ) : (
                ablationRows.map((row, idx) => (
                  <tr
                    key={row.id}
                    className={
                      idx === 0
                        ? 'bg-cyan-500/[0.06] text-slate-100'
                        : 'text-slate-300 hover:bg-slate-800/30'
                    }
                  >
                    <td className="py-2 px-3">
                      <div className="font-medium text-slate-100">{row.name}</div>
                      <div className="text-[10.5px] text-slate-400">{row.shortDesc}</div>
                    </td>
                    <td className="py-2 px-2.5 text-right font-mono whitespace-nowrap">
                      {row.rotMeanDeg.toFixed(3)} ± {row.rotStdDeg.toFixed(3)}
                    </td>
                    <td
                      className={`py-2 px-2.5 text-right font-mono whitespace-nowrap ${
                        idx === 0 ? 'text-cyan-300 font-semibold' : ''
                      }`}
                    >
                      {row.transMeanMm.toFixed(2)} ± {row.transStdMm.toFixed(2)}
                    </td>
                    <td className="py-2 px-2.5 text-right font-mono whitespace-nowrap">
                      {row.transMedianMm.toFixed(2)}
                    </td>
                    <td
                      className={`py-2 px-2.5 text-right font-mono whitespace-nowrap ${
                        row.successRate >= 80
                          ? 'text-emerald-400'
                          : row.successRate >= 50
                            ? 'text-amber-300'
                            : 'text-rose-400'
                      }`}
                    >
                      {row.successRate.toFixed(0)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-slate-400 whitespace-nowrap">
                      {(row.meanTimeMs / 1000).toFixed(2)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
