import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ClipboardCheck,
  Copy,
  Download,
  FlaskConical,
  Play,
  RotateCcw,
  Shuffle,
  Square,
  Terminal,
} from 'lucide-react';
import {
  formatAblationCSV,
  formatMarkdownReport,
  generateInitialHandEyePose,
  getInitModeLabel,
  runPairedAblationBenchmark,
  runSingleCalibration,
  sampleRandomCameraPose,
} from './calib/pipeline';
import {
  runPairedAblationBenchmarkV3,
  runSingleCalibrationV3,
} from './calib/v3/pipelineV3';
import type {
  AblationRow,
  AlgoVersion,
  CalibrationConfig,
  CalibrationResult,
  ErrorHistoryPoint,
  ExtrinsicInitMode,
  LMTracePoint,
  SyncDiagnostics,
} from './calib/types';
import { AblationTable } from './components/AblationTable';
import { ControlPanel } from './components/ControlPanel';
import { DiagnosticCanvases } from './components/DiagnosticCanvases';
import { LogViewerModal } from './components/LogViewerModal';
import type { SingleRunErrorSample } from './components/MonteCarloHistogram';
import { cameraCenterInBase, computePoseError, SeededRNG, type SE3 } from './math/se3';
import { MujocoPandaEngine } from './sim/mujocoEngine';
import { PandaRGBDScene } from './sim/rgbdScene';
import { getRobotSpec } from './sim/robotCatalog';
import { traceLogger, type LogSession } from './utils/traceLogger';

const DEFAULT_CONFIG: CalibrationConfig = {
  algoVersion: 'v3',
  robotId: 'panda',
  mode: 'modeB',
  strategy: 'fim',
  initMode: 'stage1',
  initRotErrDeg: 15,
  initTransErrMm: 120,
  numSteps: 6,
  mcTrials: 5,
  useFlyingPixelFilter: true,
  useVelocityField: true,
  useTimestampSync: true,
  enableDistractor: true,
  enableCustomTool: true,
  enableFrontPillar: false,
  enableFlexCable: false,
  estimateTimeOffset: true,
  sensor: {
    width: 320,
    height: 240,
    fovDeg: 58,
    sigmaZ0: 0.0014,
    asyncDelayMs: 25,
    enableFlyingPixels: true,
    enableHoles: true,
    edgeNoiseStrength: 1.0,
    materialNoiseStrength: 1.0,
    outlierRate: 0.004,
  },
};

export default function App() {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<MujocoPandaEngine | null>(null);
  const sceneRef = useRef<PandaRGBDScene | null>(null);
  const gtPoseRef = useRef<SE3 | null>(null);
  const externalInitPoseRef = useRef<SE3 | null>(null);
  const poseCounterRef = useRef<number>(1);
  const initResampleCounterRef = useRef<number>(1);
  const cancelRef = useRef<boolean>(false);

  const [config, setConfig] = useState<CalibrationConfig>(DEFAULT_CONFIG);
  const configRef = useRef<CalibrationConfig>(DEFAULT_CONFIG);
  configRef.current = config;

  const [engineStatus, setEngineStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [engineMessage, setEngineMessage] = useState<string>('正在初始化 MuJoCo WASM...');
  const [gtPosText, setGtPosText] = useState<string>('—');
  const [previewInitErr, setPreviewInitErr] = useState<{ rotDeg: number; transMm: number } | null>(null);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isRunningBenchmark, setIsRunningBenchmark] = useState<boolean>(false);
  const [progressText, setProgressText] = useState<string>('');
  const [progressPct, setProgressPct] = useState<number>(0);

  const [liveRgb, setLiveRgb] = useState<Uint8ClampedArray | null>(null);
  const [liveDepth, setLiveDepth] = useState<Float32Array | null>(null);
  const [liveMask, setLiveMask] = useState<Uint8Array | null>(null);
  const [v3PosteriorOverlay, setV3PosteriorOverlay] = useState<Uint8Array | null>(null);
  const [syncDiag, setSyncDiag] = useState<SyncDiagnostics | null>(null);
  const [lmTrace, setLmTrace] = useState<LMTracePoint[]>([]);
  const [errorHistory, setErrorHistory] = useState<ErrorHistoryPoint[]>([]);
  const [singleRunSamples, setSingleRunSamples] = useState<SingleRunErrorSample[]>([]);
  const [singleResult, setSingleResult] = useState<CalibrationResult | null>(null);
  const [ablationRows, setAblationRows] = useState<AblationRow[]>([]);
  const [copyFeedback, setCopyFeedback] = useState<string>('');
  const [logViewerOpen, setLogViewerOpen] = useState<boolean>(false);
  const [logSessions, setLogSessions] = useState<LogSession[]>(() => [...traceLogger.getSessions()]);
  const [activeLogSessionId, setActiveLogSessionId] = useState<string | null>(
    () => traceLogger.getActiveSession()?.id ?? null
  );
  const [totalLogCount, setTotalLogCount] = useState<number>(() => traceLogger.getTotalEntryCount());

  useEffect(() => {
    const syncLogs = () => {
      setLogSessions([...traceLogger.getSessions()]);
      setActiveLogSessionId(traceLogger.getActiveSession()?.id ?? null);
      setTotalLogCount(traceLogger.getTotalEntryCount());
    };
    return traceLogger.subscribe(syncLogs);
  }, []);

  const buildPreviewInitPose = useCallback((pose: SE3, cfg: CalibrationConfig): SE3 | null => {
    if (cfg.initMode === 'stage1') {
      externalInitPoseRef.current = null;
      setPreviewInitErr(null);
      return null;
    }
    const seed =
      (20261126 ^
        (poseCounterRef.current * 2654435761) ^
        (initResampleCounterRef.current * 40503) ^
        (cfg.initMode.length * 997)) >>>
      0;
    const initPose = generateInitialHandEyePose(
      cfg.initMode as Exclude<ExtrinsicInitMode, 'stage1'>,
      pose,
      new SeededRNG(seed),
      {
        rotErrDeg: cfg.initRotErrDeg,
        transErrMm: cfg.initTransErrMm,
      }
    );
    externalInitPoseRef.current = initPose;
    setPreviewInitErr(computePoseError(initPose, pose));
    return initPose;
  }, []);

  const captureIdlePreview = useCallback(
    (pose: SE3, cfg: CalibrationConfig) => {
      const engine = engineRef.current;
      const rgbdScene = sceneRef.current;
      if (!engine || !rgbdScene) return;

      const initPreview = buildPreviewInitPose(pose, cfg);
      engine.setState(engine.spec.homeQ, 0);
      rgbdScene.syncFromMujoco();
      rgbdScene.updateFrontPillarForCamera(pose);
      rgbdScene.setSceneVisibility({
        showCustomTool: cfg.enableCustomTool,
        showDistractor: cfg.enableDistractor,
        showFrontPillar: cfg.enableFrontPillar,
        showFlexCable: cfg.enableFlexCable,
      });
      rgbdScene.setFrustumPoses(pose, initPreview, null);
      rgbdScene.renderNow();

      const previewRng = new SeededRNG(42 + poseCounterRef.current);
      const { rgb, depth } = rgbdScene.captureVirtualRGBD(pose, cfg.sensor, previewRng);
      setLiveRgb(rgb);
      setLiveDepth(depth);
      setLiveMask(null);
      setV3PosteriorOverlay(null);

      const camPos = cameraCenterInBase(pose);
      setGtPosText(
        `(${camPos[0].toFixed(2)}, ${camPos[1].toFixed(2)}, ${camPos[2].toFixed(2)}) m`
      );
    },
    [buildPreviewInitPose]
  );

  // Initialize / switch real @mujoco/mujoco WASM + Three.js WebGL2 scene whenever robotId changes
  useEffect(() => {
    let active = true;
    const targetRobot = getRobotSpec(config.robotId);

    setEngineStatus('loading');
    setEngineMessage(`正在加载 ${targetRobot.shortName} 物理模型与材质...`);
    setSingleResult(null);
    setAblationRows([]);
    setLmTrace([]);
    setErrorHistory([]);
    setSingleRunSamples([]);
    setSyncDiag(null);
    setProgressText('');
    setProgressPct(0);

    async function boot() {
      const tBoot = performance.now();
      try {
        const engine = await traceLogger.traceAsync(
          'sim',
          'MujocoPandaEngine.create',
          { robotId: config.robotId },
          () => MujocoPandaEngine.create(config.robotId),
          (eng) => ({ versionLabel: eng.versionLabel, dof: eng.spec.dof, name: eng.spec.name }),
          `初始化 MuJoCo WASM (${targetRobot.shortName})`
        );
        if (!active) {
          engine.dispose();
          return;
        }
        engineRef.current = engine;

        const container = viewportRef.current;
        if (!container) {
          throw new Error('3D 渲染视口容器未挂载。');
        }

        const rgbdScene = new PandaRGBDScene(
          container,
          engine,
          configRef.current.sensor.width,
          configRef.current.sensor.height,
          configRef.current.sensor.fovDeg
        );
        sceneRef.current = rgbdScene;

        const initRng = new SeededRNG(20260401 + config.robotId.length * 17);
        const initialPose = traceLogger.traceCall(
          'app',
          'sampleRandomCameraPose',
          { robotId: config.robotId, seed: 20260401 + config.robotId.length * 17 },
          () => sampleRandomCameraPose(initRng, engine.spec),
          (p) => ({ pose: p, camCenterBase: cameraCenterInBase(p) })
        );
        gtPoseRef.current = initialPose;

        setEngineStatus('ready');
        setEngineMessage(`${engine.versionLabel} · WebGL2 材质光学深度`);
        traceLogger.log(
          'info',
          'app',
          'engineReady',
          `${engine.versionLabel} 就绪 (${(performance.now() - tBoot).toFixed(0)}ms)`,
          { durationMs: performance.now() - tBoot }
        );

        captureIdlePreview(initialPose, configRef.current);
      } catch (err) {
        if (!active) return;
        const msg = err instanceof Error ? err.message : String(err);
        setEngineStatus('error');
        setEngineMessage(`MuJoCo WASM 加载失败: ${msg}`);
        traceLogger.log('error', 'app', 'boot', `MuJoCo WASM 加载失败: ${msg}`, { error: msg });
      }
    }

    boot();

    return () => {
      active = false;
      cancelRef.current = true;
      sceneRef.current?.dispose();
      sceneRef.current = null;
      engineRef.current?.dispose();
      engineRef.current = null;
    };
  }, [config.robotId, captureIdlePreview]);

  // Reflect user-driven scene visibility, sensor noise, and initial extrinsic preview changes while idle
  const initSettingsKey = `${config.initMode}|${config.initRotErrDeg}|${config.initTransErrMm}`;
  const prevInitKeyRef = useRef<string>(initSettingsKey);
  const previewSettingsKey = `${config.enableCustomTool}|${config.enableDistractor}|${config.enableFrontPillar}|${config.enableFlexCable}|${initSettingsKey}|${config.sensor.sigmaZ0}|${config.sensor.enableFlyingPixels}|${config.sensor.enableHoles}|${config.sensor.edgeNoiseStrength}|${config.sensor.materialNoiseStrength}|${config.sensor.outlierRate}`;
  const prevPreviewKeyRef = useRef<string>(previewSettingsKey);

  useEffect(() => {
    if (prevPreviewKeyRef.current === previewSettingsKey) return;
    prevPreviewKeyRef.current = previewSettingsKey;
    if (engineStatus !== 'ready' || isRunning || !gtPoseRef.current) return;
    if (prevInitKeyRef.current !== initSettingsKey) {
      prevInitKeyRef.current = initSettingsKey;
      setSingleResult(null);
      setLmTrace([]);
      setErrorHistory([]);
      setV3PosteriorOverlay(null);
    }
    captureIdlePreview(gtPoseRef.current, configRef.current);
  }, [engineStatus, isRunning, previewSettingsKey, initSettingsKey, captureIdlePreview]);

  const handleSwitchAlgoVersion = (nextVer: AlgoVersion) => {
    if (isRunning || nextVer === config.algoVersion) return;
    setConfig((prev) => ({ ...prev, algoVersion: nextVer }));
    setSingleResult(null);
    setLmTrace([]);
    setErrorHistory([]);
    setSingleRunSamples([]);
    setSyncDiag(null);
    setV3PosteriorOverlay(null);
    setAblationRows([]);
    setProgressText('');
    setProgressPct(0);
    if (gtPoseRef.current && sceneRef.current) {
      const initPreview = buildPreviewInitPose(gtPoseRef.current, { ...config, algoVersion: nextVer });
      sceneRef.current.setFrustumPoses(gtPoseRef.current, initPreview, null);
      sceneRef.current.renderNow();
    }
  };

  const handleResampleInit = () => {
    if (engineStatus !== 'ready' || isRunning || !gtPoseRef.current || !sceneRef.current) return;
    initResampleCounterRef.current += 1;
    const initPreview = buildPreviewInitPose(gtPoseRef.current, config);
    sceneRef.current.setFrustumPoses(gtPoseRef.current, initPreview, null);
    sceneRef.current.renderNow();
    setSingleResult(null);
    setLmTrace([]);
    setErrorHistory([]);
  };

  const handleRandomCameraPose = () => {
    const engine = engineRef.current;
    if (engineStatus !== 'ready' || isRunning || !engine) return;
    poseCounterRef.current += 1;
    const rng = new SeededRNG(Date.now() ^ (poseCounterRef.current * 2654435761));
    const nextPose = sampleRandomCameraPose(rng, engine.spec);
    gtPoseRef.current = nextPose;
    setSingleResult(null);
    setLmTrace([]);
    setErrorHistory([]);
    setSyncDiag(null);
    captureIdlePreview(nextPose, config);
  };

  const handleRunSingle = async () => {
    const engine = engineRef.current;
    const rgbdScene = sceneRef.current;
    const gtPose = gtPoseRef.current;
    if (!engine || !rgbdScene || !gtPose || isRunning) return;

    const verTag = config.algoVersion === 'v3' ? 'V3.0' : 'V2.0';
    cancelRef.current = false;
    setIsRunning(true);
    setIsRunningBenchmark(false);
    setV3PosteriorOverlay(null);
    setLmTrace([]);
    setErrorHistory([]);
    setProgressText(
      `[${verTag}] 启动 ${engine.spec.shortName} 标定 [${getInitModeLabel(config.initMode, config.initRotErrDeg, config.initTransErrMm)}]...`
    );
    setProgressPct(0.02);

    traceLogger.startSession({
      title: `${verTag} 单次标定 · ${engine.spec.shortName}`,
      kind: 'single',
      algoVersion: config.algoVersion,
      robotId: config.robotId,
      configSnapshot: config,
    });

    try {
      const seed = 7777 + poseCounterRef.current * 131;
      const runner =
        config.algoVersion === 'v3' ? runSingleCalibrationV3 : runSingleCalibration;
      const res = await runner(engine, rgbdScene, gtPose, config, seed, {
        shouldCancel: () => cancelRef.current,
        externalInitPose: config.initMode !== 'stage1' ? externalInitPoseRef.current : null,
        onLiveUpdate: (upd) => {
          setProgressText(upd.phaseText);
          setProgressPct(upd.progress);
          if (upd.rgb) setLiveRgb(upd.rgb);
          if (upd.depth) setLiveDepth(upd.depth);
          if (upd.mask) setLiveMask(upd.mask);
          if (upd.v3PosteriorOverlay !== undefined) {
            setV3PosteriorOverlay(upd.v3PosteriorOverlay);
          }
          if (upd.syncDiagnostics) setSyncDiag(upd.syncDiagnostics);
          if (upd.lmTrace) setLmTrace(upd.lmTrace);
          if (upd.errorHistory) setErrorHistory(upd.errorHistory);
          if (upd.stage1Pose) {
            rgbdScene.setFrustumPoses(gtPose, upd.stage1Pose, upd.stage2Pose ?? null);
          }
        },
      });

      if (!cancelRef.current) {
        setSingleResult(res);
        setLmTrace(res.lmTrace);
        if (res.errorHistory) setErrorHistory(res.errorHistory);
        setSingleRunSamples((prev) => [
          ...prev,
          { transErrMm: res.finalTransMm, rotErrDeg: res.finalRotDeg },
        ]);
        if (res.syncDiagnostics) setSyncDiag(res.syncDiagnostics);
        if (res.lastRgb) setLiveRgb(res.lastRgb);
        if (res.lastDepth) setLiveDepth(res.lastDepth);
        if (res.lastMask) setLiveMask(res.lastMask);
        setV3PosteriorOverlay(res.v3Diagnostics?.posteriorOverlay ?? null);
        setProgressText(
          `[${verTag}] ${engine.spec.shortName} 标定完成 · 最终误差 ${res.finalTransMm.toFixed(2)} mm / ${res.finalRotDeg.toFixed(3)}°`
        );
        setProgressPct(1);
        traceLogger.endSession({
          algoVersion: config.algoVersion,
          robotId: config.robotId,
          mode: config.mode,
          initMode: config.initMode,
          finalTransMm: res.finalTransMm,
          finalRotDeg: res.finalRotDeg,
          success: res.success,
          elapsedMs: res.elapsedMs,
        });
      } else {
        setProgressText('已中止标定');
        traceLogger.log('warn', 'app', 'handleRunSingle', '用户手动中止单次标定');
        traceLogger.endSession({ cancelled: true });
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      traceLogger.log('error', 'app', 'handleRunSingle', `标定运行异常: ${msg}`, { error: msg });
      traceLogger.endSession({ error: msg });
      throw err;
    } finally {
      setIsRunning(false);
    }
  };

  const handleRunAblation = async () => {
    const engine = engineRef.current;
    const rgbdScene = sceneRef.current;
    if (!engine || !rgbdScene || isRunning) return;

    const verTag = config.algoVersion === 'v3' ? 'V3.0' : 'V2.0';
    cancelRef.current = false;
    setIsRunning(true);
    setIsRunningBenchmark(true);
    setAblationRows([]);
    setV3PosteriorOverlay(null);
    setProgressText(`[${verTag}] 初始化 ${engine.spec.shortName} 6 组配对 Monte Carlo 消融实验...`);
    setProgressPct(0.01);

    traceLogger.startSession({
      title: `${verTag} 6组消融 · ${engine.spec.shortName} (N=${config.mcTrials})`,
      kind: 'ablation',
      algoVersion: config.algoVersion,
      robotId: config.robotId,
      configSnapshot: config,
    });

    try {
      const benchRunner =
        config.algoVersion === 'v3' ? runPairedAblationBenchmarkV3 : runPairedAblationBenchmark;
      const rows = await benchRunner(engine, rgbdScene, config, {
        baseSeed: 20261126,
        shouldCancel: () => cancelRef.current,
        onProgress: (info) => {
          setProgressText(info.text);
          setProgressPct(info.progress);
          setAblationRows(info.rows);
          if (info.latestResult) {
            setSingleResult(info.latestResult);
            setLmTrace(info.latestResult.lmTrace);
            setErrorHistory(info.latestResult.errorHistory ?? []);
            if (info.latestResult.syncDiagnostics) {
              setSyncDiag(info.latestResult.syncDiagnostics);
            }
            if (info.latestResult.lastRgb) setLiveRgb(info.latestResult.lastRgb);
            if (info.latestResult.lastDepth) setLiveDepth(info.latestResult.lastDepth);
            if (info.latestResult.lastMask) setLiveMask(info.latestResult.lastMask);
            setV3PosteriorOverlay(info.latestResult.v3Diagnostics?.posteriorOverlay ?? null);
            gtPoseRef.current = info.latestResult.T_cb_gt;
            const cPos = cameraCenterInBase(info.latestResult.T_cb_gt);
            setGtPosText(`(${cPos[0].toFixed(2)}, ${cPos[1].toFixed(2)}, ${cPos[2].toFixed(2)}) m`);
          }
        },
      });

      if (!cancelRef.current) {
        setAblationRows(rows);
        setProgressText(`[${verTag}] ${engine.spec.shortName} 6 组配对消融基准全部完成 (N=${config.mcTrials})`);
        setProgressPct(1);
        traceLogger.endSession({
          algoVersion: config.algoVersion,
          robotId: config.robotId,
          ablationRowCount: rows.length,
        });
      } else {
        setProgressText('已中止消融实验');
        traceLogger.log('warn', 'app', 'handleRunAblation', '用户手动中止消融实验');
        traceLogger.endSession({ ablationRowCount: rows.length, cancelled: true });
      }
    } finally {
      setIsRunning(false);
      setIsRunningBenchmark(false);
    }
  };

  const handleCancel = () => {
    cancelRef.current = true;
  };

  const handleExportCSV = () => {
    if (ablationRows.length === 0 && !singleResult) return;
    const rowsToExport: AblationRow[] =
      ablationRows.length > 0
        ? ablationRows
        : singleResult
          ? [
              {
                id: 'single_run',
                name: config.mode === 'modeB' ? 'Single Run (Mode B)' : 'Single Run (Mode A)',
                shortDesc: `${config.strategy.toUpperCase()} / ${config.numSteps} steps`,
                trials: 1,
                rotMeanDeg: singleResult.finalRotDeg,
                rotStdDeg: 0,
                transMeanMm: singleResult.finalTransMm,
                transStdMm: 0,
                transMedianMm: singleResult.finalTransMm,
                successRate: singleResult.success ? 100 : 0,
                meanTimeMs: singleResult.elapsedMs,
                fimLogDetMean: singleResult.fimLogDet,
                trialRotDegs: [singleResult.finalRotDeg],
                trialTransMms: [singleResult.finalTransMm],
              },
            ]
          : [];

    const csv = formatAblationCSV(rowsToExport, config);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${config.robotId}_hand_eye_ablation_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyMarkdown = async () => {
    const md = formatMarkdownReport(config, singleResult, ablationRows);
    try {
      await navigator.clipboard.writeText(md);
      setCopyFeedback('已复制 Markdown');
      window.setTimeout(() => setCopyFeedback(''), 2200);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = md;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopyFeedback('已复制 Markdown');
      window.setTimeout(() => setCopyFeedback(''), 2200);
    }
  };

  const canExport = ablationRows.length > 0 || singleResult !== null;
  const activeRobot = getRobotSpec(config.robotId);

  return (
    <div
      className="min-h-screen flex flex-col bg-[#090C12] text-slate-100"
      data-engine-status={engineStatus}
      data-robot-id={config.robotId}
      data-algo-version={config.algoVersion}
    >
      {/* Top Scientific Instrument Header */}
      <header className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-[#0D121B] border-b border-slate-800">
        <div className="flex flex-wrap items-center gap-3 min-w-0">
          <h1 className="text-[15px] font-semibold tracking-tight text-slate-50 whitespace-nowrap">
            免分割 RGB-D 主动运动手眼标定平台
          </h1>

          {/* Side-by-Side V2.0 / V3.0 Algorithm Core Switcher */}
          <div
            role="radiogroup"
            aria-label="算法核心版本切换"
            className="inline-flex items-center p-0.5 bg-[#090C12] border border-slate-700 rounded-sm shrink-0"
          >
            <button
              type="button"
              data-testid="algo-version-v2"
              disabled={isRunning}
              onClick={() => handleSwitchAlgoVersion('v2')}
              className={`px-2.5 py-1 text-[11.5px] font-mono font-medium rounded-[2px] transition-colors whitespace-nowrap ${
                config.algoVersion === 'v2'
                  ? 'bg-amber-500/25 text-amber-200 border border-amber-400/70'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              } disabled:opacity-50`}
            >
              V2.0 基础框架
            </button>
            <button
              type="button"
              data-testid="algo-version-v3"
              disabled={isRunning}
              onClick={() => handleSwitchAlgoVersion('v3')}
              className={`px-2.5 py-1 text-[11.5px] font-mono font-medium rounded-[2px] transition-colors whitespace-nowrap ${
                config.algoVersion === 'v3'
                  ? 'bg-cyan-500/25 text-cyan-200 border border-cyan-400/70'
                  : 'text-slate-400 hover:text-slate-200 border border-transparent'
              } disabled:opacity-50`}
            >
              V3.0 统一引擎
            </button>
          </div>

          <span aria-hidden="true" className="text-slate-700 hidden xl:inline">
            |
          </span>
          <div
            data-testid="mujoco-status"
            className={`text-[11.5px] font-mono truncate ${
              engineStatus === 'ready'
                ? 'text-emerald-400'
                : engineStatus === 'error'
                  ? 'text-rose-400 font-semibold'
                  : 'text-amber-300'
            }`}
          >
            {engineStatus === 'ready' && `● ${engineMessage}`}
            {engineStatus === 'loading' && `▲ ${engineMessage}`}
            {engineStatus === 'error' && `✖ ${engineMessage}`}
          </div>
        </div>

        {/* Primary Action Toolbar */}
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <button
            type="button"
            disabled={engineStatus !== 'ready' || isRunning}
            onClick={handleRunSingle}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-[12px] font-medium rounded-sm bg-cyan-500 text-slate-950 hover:bg-cyan-400 transition-colors whitespace-nowrap shrink-0 disabled:opacity-45 disabled:cursor-not-allowed"
          >
            <Play className="w-3.5 h-3.5" />
            运行单次标定
          </button>

          <button
            type="button"
            disabled={engineStatus !== 'ready' || isRunning}
            onClick={handleRandomCameraPose}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-sm bg-slate-800/90 text-slate-200 border border-slate-700 hover:bg-slate-700 transition-colors whitespace-nowrap shrink-0 disabled:opacity-45 disabled:cursor-not-allowed"
          >
            <Shuffle className="w-3.5 h-3.5" />
            随机相机位姿
          </button>

          <button
            type="button"
            disabled={engineStatus !== 'ready' || isRunning}
            onClick={handleRunAblation}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-sm bg-slate-800/90 text-cyan-200 border border-cyan-500/40 hover:bg-cyan-500/15 transition-colors whitespace-nowrap shrink-0 disabled:opacity-45 disabled:cursor-not-allowed"
          >
            <FlaskConical className="w-3.5 h-3.5" />
            一键跑 6 组消融实验
          </button>

          {isRunning && (
            <button
              type="button"
              onClick={handleCancel}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[12px] font-medium rounded-sm bg-rose-500/20 text-rose-300 border border-rose-500/50 hover:bg-rose-500/30 transition-colors whitespace-nowrap shrink-0"
            >
              <Square className="w-3 h-3" />
              中止
            </button>
          )}

          <button
            type="button"
            disabled={!canExport}
            onClick={handleExportCSV}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] font-medium rounded-sm bg-[#10151F] text-slate-300 border border-slate-700 hover:text-white hover:border-slate-500 transition-colors whitespace-nowrap shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-3.5 h-3.5" />
            导出 CSV
          </button>

          <button
            type="button"
            disabled={!canExport}
            onClick={handleCopyMarkdown}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] font-medium rounded-sm bg-[#10151F] text-slate-300 border border-slate-700 hover:text-white hover:border-slate-500 transition-colors whitespace-nowrap shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {copyFeedback ? (
              <ClipboardCheck className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
            {copyFeedback || '复制 Markdown 报告'}
          </button>

          <button
            type="button"
            data-testid="open-log-viewer"
            onClick={() => setLogViewerOpen(true)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] font-medium rounded-sm bg-[#10151F] text-cyan-200 border border-cyan-500/45 hover:bg-cyan-500/15 hover:border-cyan-400 transition-colors whitespace-nowrap shrink-0"
          >
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            运行日志
            {totalLogCount > 0 && (
              <span
                data-testid="toolbar-log-badge"
                className="px-1 py-0 text-[10px] font-mono rounded-[2px] bg-cyan-500/20 text-cyan-300"
              >
                {totalLogCount}
              </span>
            )}
          </button>
        </div>
      </header>

      {/* Prominent Error Banner if MuJoCo WASM Failed (zero silent mock fallback) */}
      {engineStatus === 'error' && (
        <div
          role="alert"
          className="mx-4 mt-3 px-4 py-3 rounded-md bg-rose-950/70 border border-rose-500/60 text-rose-200 text-[13px] font-mono"
        >
          {engineMessage} — 请确认浏览器支持 WebAssembly 与 WebGL2，系统严禁切换至模拟数据。
        </div>
      )}

      {/* Three-Column Scientific Workstation */}
      <main className="flex-1 grid grid-cols-1 xl:grid-cols-[282px_minmax(0,1fr)_476px] gap-3 p-3 min-h-0">
        {/* Left Column: Robot Selector, Controls & RealSense Material/Edge Noise */}
        <ControlPanel
          config={config}
          disabled={isRunning}
          previewInitErr={previewInitErr}
          onResampleInit={handleResampleInit}
          onChange={setConfig}
        />

        {/* Center Column: Interactive 3D Viewport + Quantitative Metrics + 6-Condition Ablation Table */}
        <div className="flex flex-col gap-3 min-w-0">
          {/* 3D MuJoCo + Three.js Viewport */}
          <section className="relative bg-[#0B0F17] border border-slate-800 rounded-md overflow-hidden flex flex-col">
            {/* Viewport Top Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2 bg-[#10151F] border-b border-slate-800 text-[11.5px]">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono">
                <span className="text-slate-300 font-sans font-semibold">
                  3D 场景 · {activeRobot.shortName} (Z-up)
                </span>
                <span className="text-emerald-400">■ 绿色=相机真值</span>
                <span className="text-amber-400">
                  ■ 黄色=
                  {config.initMode === 'stage1'
                    ? config.algoVersion === 'v3'
                      ? 'Stage 2 闭式解'
                      : 'Stage 1 闭式解'
                    : '初始外参 T_cb⁽⁰⁾'}
                </span>
                <span className="text-cyan-400">
                  ■ 青色={config.algoVersion === 'v3' ? 'Stage 4 退火 EM' : 'Stage 2 精修解'}
                </span>
              </div>
              <div className="flex items-center gap-3 font-mono text-[11px] text-slate-400">
                <span>GT 位置: {gtPosText}</span>
                <button
                  type="button"
                  onClick={() => sceneRef.current?.resetViewCamera()}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-sm bg-slate-800 text-slate-300 hover:text-white border border-slate-700 transition-colors"
                >
                  <RotateCcw className="w-3 h-3" />
                  重置视角
                </button>
              </div>
            </div>

            {/* Dedicated WebGL Mount Container (never unmounted by HUD state updates) */}
            <div className="relative w-full h-[345px] sm:h-[372px]">
              <div
                ref={viewportRef}
                data-testid="viewport-3d"
                className="w-full h-full cursor-grab active:cursor-grabbing"
              />

              {/* Solid surface progress overlay (no backdrop-blur over WebGL) */}
              {(isRunning || progressText) && (
                <div className="absolute left-3 right-3 bottom-3 bg-[#090C12]/90 border border-slate-700/90 rounded px-3 py-2 pointer-events-none">
                  <div className="flex items-center justify-between gap-3 text-[11.5px] font-mono">
                    <span className="text-slate-200 truncate">{progressText}</span>
                    <span className="text-cyan-300 shrink-0">
                      {Math.round(progressPct * 100)}%
                    </span>
                  </div>
                  <div className="mt-1.5 h-1 w-full bg-slate-800 rounded-sm overflow-hidden">
                    <div
                      className="h-full bg-cyan-400 transition-all duration-150"
                      style={{ width: `${Math.max(2, Math.min(100, progressPct * 100))}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* Metrics Strip + 6-Condition Paired Ablation Table */}
          <AblationTable
            algoVersion={config.algoVersion}
            trueAsyncDelayMs={config.sensor.asyncDelayMs}
            initMode={config.initMode}
            previewInitErr={previewInitErr}
            singleResult={singleResult}
            ablationRows={ablationRows}
            isRunningBenchmark={isRunningBenchmark}
          />
        </div>

        {/* Right Column: Live Virtual RGB, Turbo Depth + Mask / V3 EM Posterior, Sync Waveforms, Live Trans/Rot Error History & MC Histogram */}
        <DiagnosticCanvases
          algoVersion={config.algoVersion}
          width={config.sensor.width}
          height={config.sensor.height}
          rgb={liveRgb}
          depth={liveDepth}
          mask={liveMask}
          v3PosteriorOverlay={v3PosteriorOverlay}
          syncDiagnostics={syncDiag}
          lmTrace={lmTrace}
          errorHistory={errorHistory}
          ablationRows={ablationRows}
          singleRunSamples={singleRunSamples}
          isRunning={isRunning}
          isRunningBenchmark={isRunningBenchmark}
        />
      </main>

      <LogViewerModal
        open={logViewerOpen}
        onClose={() => setLogViewerOpen(false)}
        sessions={logSessions}
        activeSessionId={activeLogSessionId}
      />
    </div>
  );
}
