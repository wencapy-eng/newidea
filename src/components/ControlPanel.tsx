import React from 'react';
import type { CalibrationConfig, ExtrinsicInitMode } from '../calib/types';
import { getRobotSpec, ROBOT_LIST, type RobotId } from '../sim/robotCatalog';

interface ControlPanelProps {
  config: CalibrationConfig;
  disabled: boolean;
  previewInitErr?: { rotDeg: number; transMm: number } | null;
  onResampleInit?: () => void;
  onChange: (next: CalibrationConfig) => void;
}

interface ToggleRowProps {
  id: string;
  label: string;
  desc: string;
  checked: boolean;
  disabled: boolean;
  onToggle: (val: boolean) => void;
}

function ToggleRow({ id, label, desc, checked, disabled, onToggle }: ToggleRowProps) {
  return (
    <label
      htmlFor={id}
      className={`flex items-start justify-between gap-3 py-2 border-b border-slate-800/70 last:border-b-0 ${
        disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      }`}
    >
      <div className="min-w-0">
        <div className="text-[12.5px] font-medium text-slate-200 leading-snug">{label}</div>
        <div className="text-[11px] text-slate-400 leading-tight mt-0.5">{desc}</div>
      </div>
      <div className="shrink-0 pt-0.5">
        <input
          id={id}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="sr-only peer"
        />
        <span
          aria-hidden="true"
          className={`inline-flex items-center w-9 h-5 rounded-sm border transition-colors ${
            checked
              ? 'bg-cyan-500/25 border-cyan-400/70 justify-end'
              : 'bg-slate-900 border-slate-700 justify-start'
          } p-0.5`}
        >
          <span
            className={`block w-3.5 h-3.5 rounded-[2px] transition-colors ${
              checked ? 'bg-cyan-400' : 'bg-slate-500'
            }`}
          />
        </span>
      </div>
    </label>
  );
}

interface NoisePreset {
  id: string;
  label: string;
  patch: Partial<CalibrationConfig['sensor']>;
}

const NOISE_PRESETS: readonly NoisePreset[] = [
  {
    id: 'std',
    label: '标准 D435i',
    patch: {
      sigmaZ0: 0.0014,
      edgeNoiseStrength: 1.0,
      materialNoiseStrength: 1.0,
      outlierRate: 0.004,
      enableFlyingPixels: true,
      enableHoles: true,
    },
  },
  {
    id: 'material',
    label: '金属·黑体退化',
    patch: {
      sigmaZ0: 0.0018,
      edgeNoiseStrength: 1.5,
      materialNoiseStrength: 2.2,
      outlierRate: 0.012,
      enableFlyingPixels: true,
      enableHoles: true,
    },
  },
  {
    id: 'extreme',
    label: '极端边缘大噪声',
    patch: {
      sigmaZ0: 0.0022,
      edgeNoiseStrength: 2.6,
      materialNoiseStrength: 1.8,
      outlierRate: 0.025,
      enableFlyingPixels: true,
      enableHoles: true,
    },
  },
];

interface InitModeOption {
  id: ExtrinsicInitMode;
  label: string;
  desc: string;
}

const INIT_MODE_OPTIONS: readonly InitModeOption[] = [
  { id: 'stage1', label: 'Stage 1 闭式解', desc: '无先验 · 螺旋轴解' },
  { id: 'controlled_random', label: '受控随机初始化', desc: '真值可控 SE(3) 扰动' },
  { id: 'full_random', label: '完全随机初始化', desc: 'SO(3)×R³ 盲采样' },
  { id: 'zero', label: '0 初始化 (I₄)', desc: 'ξ⁽⁰⁾=0 · R=I₃, t=0' },
];

export const ControlPanel: React.FC<ControlPanelProps> = ({
  config,
  disabled,
  previewInitErr,
  onResampleInit,
  onChange,
}) => {
  const update = (patch: Partial<CalibrationConfig>) => onChange({ ...config, ...patch });
  const updateSensor = (patch: Partial<CalibrationConfig['sensor']>) =>
    onChange({ ...config, sensor: { ...config.sensor, ...patch } });

  const activeRobot = getRobotSpec(config.robotId);
  const isV3 = config.algoVersion === 'v3';

  const handleSelectRobot = (robotId: RobotId) => {
    if (disabled || robotId === config.robotId) return;
    update({ robotId });
  };

  return (
    <aside
      aria-label="实验参数与消融控制"
      className="flex flex-col gap-3 bg-[#10151F] border border-slate-800 rounded-md p-3.5 overflow-y-auto"
    >
      {/* Section 0: Multi-Robot Mechanism & CAD Selector */}
      <section>
        <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-800">
          <h2 className="text-[12px] font-semibold text-slate-300 tracking-wide">
            机器人构型与本体模型
          </h2>
          <span className="text-[10.5px] font-mono text-cyan-300">7 款构型</span>
        </div>

        <div className="grid grid-cols-2 gap-1.5" role="radiogroup" aria-label="选择机器人模型">
          {ROBOT_LIST.map((rob) => {
            const isSelected = rob.id === config.robotId;
            return (
              <button
                key={rob.id}
                type="button"
                data-testid={`robot-btn-${rob.id}`}
                disabled={disabled}
                onClick={() => handleSelectRobot(rob.id)}
                className={`text-left px-2 py-1.5 rounded-sm border transition-colors ${
                  isSelected
                    ? 'bg-cyan-500/18 border-cyan-400/70 text-cyan-100'
                    : 'bg-[#090C12] border-slate-800 text-slate-300 hover:border-slate-600 hover:text-slate-100'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <div className="text-[11.5px] font-medium leading-tight truncate">
                  {rob.shortName}
                </div>
                <div className="text-[10px] font-mono text-slate-400 leading-tight mt-0.5 truncate">
                  {rob.dof}-DoF · {rob.mechanism === 'parallel_delta' ? '闭链并联' : rob.cadAssetId ? '官方CAD' : '异形串联'}
                </div>
              </button>
            );
          })}
        </div>

        <div className="mt-2 px-2 py-1.5 rounded-sm bg-[#090C12] border border-slate-800/80 text-[10.5px] text-slate-400 leading-snug">
          <span className="text-slate-200 font-medium">{activeRobot.name}</span>
          <span className="mx-1.5 text-slate-600">|</span>
          <span className="font-mono text-slate-300">{activeRobot.materialTag}</span>
        </div>
      </section>

      {/* Section 1: Calibration Pipeline & Planning */}
      <section className="pt-0.5">
        <h2 className="text-[12px] font-semibold text-slate-300 tracking-wide mb-2.5 pb-1.5 border-b border-slate-800">
          标定模式与主动探索策略
        </h2>

        <div className="space-y-3">
          <div>
            <div className="text-[11.5px] text-slate-400 mb-1.5">求解模式</div>
            <div className="grid grid-cols-2 gap-1 p-1 bg-[#090C12] border border-slate-800 rounded">
              <button
                type="button"
                disabled={disabled}
                onClick={() => update({ mode: 'modeB' })}
                className={`px-2 py-1.5 text-[11.5px] font-medium rounded-sm transition-colors whitespace-nowrap ${
                  config.mode === 'modeB'
                    ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/50'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                } disabled:opacity-50`}
              >
                {isV3 ? '模式 B · 退火 EM' : '模式 B · 流形精修'}
              </button>
              <button
                type="button"
                disabled={disabled}
                onClick={() => update({ mode: 'modeA' })}
                className={`px-2 py-1.5 text-[11.5px] font-medium rounded-sm transition-colors whitespace-nowrap ${
                  config.mode === 'modeA'
                    ? 'bg-amber-500/20 text-amber-200 border border-amber-500/50'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                } disabled:opacity-50`}
              >
                {isV3 ? '模式 A · 广义螺旋' : '模式 A · 无 Mesh'}
              </button>
            </div>
          </div>

          <div>
            <div className="text-[11.5px] text-slate-400 mb-1.5">动作探索规划</div>
            <div className="grid grid-cols-2 gap-1 p-1 bg-[#090C12] border border-slate-800 rounded">
              <button
                type="button"
                disabled={disabled}
                onClick={() => update({ strategy: 'fim' })}
                className={`px-2 py-1.5 text-[11.5px] font-medium rounded-sm transition-colors whitespace-nowrap ${
                  config.strategy === 'fim'
                    ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/50'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                } disabled:opacity-50`}
              >
                {isV3 ? 'Z-Buf 无量纲 FIM' : 'FIM D/E-最优'}
              </button>
              <button
                type="button"
                disabled={disabled}
                onClick={() => update({ strategy: 'random' })}
                className={`px-2 py-1.5 text-[11.5px] font-medium rounded-sm transition-colors whitespace-nowrap ${
                  config.strategy === 'random'
                    ? 'bg-slate-700/60 text-slate-100 border border-slate-500/60'
                    : 'text-slate-400 hover:text-slate-200 border border-transparent'
                } disabled:opacity-50`}
              >
                Random 随机对照
              </button>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[11.5px] text-slate-400">初始手眼外参初始化方式</span>
              {config.initMode !== 'stage1' && config.initMode !== 'zero' && onResampleInit && (
                <button
                  type="button"
                  data-testid="btn-resample-init"
                  disabled={disabled}
                  onClick={onResampleInit}
                  className="text-[10.5px] font-mono text-amber-300 hover:text-amber-200 underline underline-offset-2 disabled:opacity-40"
                >
                  重采样初值
                </button>
              )}
            </div>
            <div className="grid grid-cols-2 gap-1.5" role="radiogroup" aria-label="初始手眼外参初始化方式">
              {INIT_MODE_OPTIONS.map((opt) => {
                const isSelected = config.initMode === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    data-testid={`init-mode-${opt.id}`}
                    disabled={disabled}
                    onClick={() => update({ initMode: opt.id })}
                    className={`text-left px-2 py-1.5 rounded-sm border transition-colors ${
                      isSelected
                        ? opt.id === 'stage1'
                          ? 'bg-cyan-500/18 border-cyan-400/70 text-cyan-100'
                          : 'bg-amber-500/18 border-amber-400/70 text-amber-100'
                        : 'bg-[#090C12] border-slate-800 text-slate-300 hover:border-slate-600 hover:text-slate-100'
                    } disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    <div className="text-[11px] font-medium leading-tight truncate">
                      {opt.label}
                    </div>
                    <div className="text-[9.5px] font-mono text-slate-400 leading-tight mt-0.5 truncate">
                      {opt.desc}
                    </div>
                  </button>
                );
              })}
            </div>

            {config.initMode === 'controlled_random' && (
              <div className="mt-2 p-2 rounded-sm bg-[#090C12] border border-amber-500/30 space-y-2">
                <div>
                  <div className="flex items-center justify-between text-[11px] mb-0.5">
                    <label htmlFor="slider-init-rot" className="text-slate-300">
                      初始旋转扰动 Δθ
                    </label>
                    <span className="font-mono text-amber-300 font-semibold">
                      {config.initRotErrDeg.toFixed(0)}°
                    </span>
                  </div>
                  <input
                    id="slider-init-rot"
                    data-testid="slider-init-rot"
                    type="range"
                    min={5}
                    max={60}
                    step={1}
                    disabled={disabled}
                    value={config.initRotErrDeg}
                    onChange={(e) => update({ initRotErrDeg: Number(e.target.value) })}
                    className="w-full lab-slider"
                  />
                </div>
                <div>
                  <div className="flex items-center justify-between text-[11px] mb-0.5">
                    <label htmlFor="slider-init-trans" className="text-slate-300">
                      初始平移扰动 Δt
                    </label>
                    <span className="font-mono text-amber-300 font-semibold">
                      {config.initTransErrMm.toFixed(0)} mm
                    </span>
                  </div>
                  <input
                    id="slider-init-trans"
                    data-testid="slider-init-trans"
                    type="range"
                    min={20}
                    max={350}
                    step={10}
                    disabled={disabled}
                    value={config.initTransErrMm}
                    onChange={(e) => update({ initTransErrMm: Number(e.target.value) })}
                    className="w-full lab-slider"
                  />
                </div>
              </div>
            )}

            {config.initMode !== 'stage1' && (
              <div className="mt-1.5 px-2 py-1.5 rounded-sm bg-[#090C12] border border-slate-800/80 text-[10px] font-mono text-slate-400 leading-snug">
                {previewInitErr ? (
                  <div className="text-amber-300">
                    当前初值误差: {previewInitErr.transMm.toFixed(1)} mm / {previewInitErr.rotDeg.toFixed(1)}°
                  </div>
                ) : null}
                <div className="mt-0.5 text-slate-400">
                  {config.mode === 'modeB'
                    ? '绕过 Stage 1，直接从该初始外参启动 Stage 2 流形优化'
                    : '提示: 外部初值仅作用于模式 B；模式 A 仍执行闭式解'}
                </div>
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-steps" className="text-slate-300">
                总探索步数 (含 3 步冷启动)
              </label>
              <span className="font-mono text-cyan-300 font-semibold">{config.numSteps} 步</span>
            </div>
            <input
              id="slider-steps"
              type="range"
              min={4}
              max={8}
              step={1}
              disabled={disabled}
              value={config.numSteps}
              onChange={(e) => update({ numSteps: Number(e.target.value) })}
              className="w-full lab-slider"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
              <span>4</span>
              <span>6</span>
              <span>8</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-trials" className="text-slate-300">
                Monte Carlo 每组试验次数
              </label>
              <span className="font-mono text-cyan-300 font-semibold">N = {config.mcTrials}</span>
            </div>
            <input
              id="slider-trials"
              type="range"
              min={3}
              max={20}
              step={1}
              disabled={disabled}
              value={config.mcTrials}
              onChange={(e) => update({ mcTrials: Number(e.target.value) })}
              className="w-full lab-slider"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
              <span>3 (快速)</span>
              <span>10</span>
              <span>20 (论文级)</span>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Algorithm & Scene Ablation Switches */}
      <section className="pt-0.5">
        <h2 className="text-[12px] font-semibold text-slate-300 tracking-wide mb-1.5 pb-1.5 border-b border-slate-800">
          算法模块与场景干扰开关
        </h2>
        <div>
          <ToggleRow
            id="toggle-flying-filter"
            label={isV3 ? '纯光学权重与双域滤波' : '飞点滤波与物理不确定性加权'}
            desc={isV3 ? '2D M_occ + 3D V_static + 梯度与入射角权重' : '边缘剔除 + 2px 腐蚀 + 3D SOR + σ₀²z⁴ 加权'}
            checked={config.useFlyingPixelFilter}
            disabled={disabled}
            onToggle={(v) => update({ useFlyingPixelFilter: v })}
          />
          <ToggleRow
            id="toggle-vel-field"
            label={isV3 ? '5帧 SG 速度场 + Class II 复用' : '动态法向速度场残差'}
            desc={isV3 ? '非标刚体静默几何、保留动态速度场反对称约束' : 'Stage 2 联合 ∂D/∂t 速度场与静态点到面'}
            checked={config.useVelocityField}
            disabled={disabled}
            onToggle={(v) => update({ useVelocityField: v })}
          />
          <ToggleRow
            id="toggle-sync-filter"
            label={isV3 ? '多碎片时序因果同步聚合' : '时间戳同步相关性滤波'}
            desc={isV3 ? '放弃单连通域假设，聚合全部 ρ≥0.40 碎片' : 'BFS 连通域与编码器 |q̇| 互相关 ρ ≥ 0.42'}
            checked={config.useTimestampSync}
            disabled={disabled}
            onToggle={(v) => update({ useTimestampSync: v })}
          />
          {isV3 && (
            <ToggleRow
              id="toggle-estimate-dt"
              label="时间偏置 δt 联合辨识 (7-DoF)"
              desc="Stage 4 增广状态 [δφ, δρ, δt] 自动辨识异步延迟"
              checked={config.estimateTimeOffset}
              disabled={disabled}
              onToggle={(v) => update({ estimateTimeOffset: v })}
            />
          )}
          <ToggleRow
            id="toggle-distractor"
            label="异步动态干扰物 (Class IV)"
            desc="独立正弦频率滑移干扰体 distractor_body"
            checked={config.enableDistractor}
            disabled={disabled}
            onToggle={(v) => update({ enableDistractor: v })}
          />
          <ToggleRow
            id="toggle-custom-tool"
            label="未知非标刚体夹具 (Class II)"
            desc="渲染进 RGB-D，严格从 CAD 表面采样点集剔除"
            checked={config.enableCustomTool}
            disabled={disabled}
            onToggle={(v) => update({ enableCustomTool: v })}
          />
          <ToggleRow
            id="toggle-front-pillar"
            label="前侧静态遮挡立柱 (Class IV)"
            desc="视线前方铝型材立柱，垂直切割机械臂视野"
            checked={config.enableFrontPillar}
            disabled={disabled}
            onToggle={(v) => update({ enableFrontPillar: v })}
          />
          <ToggleRow
            id="toggle-flex-cable"
            label="耦合柔性甩动线缆 (Class III)"
            desc="随臂运动并弹性甩动的非刚体线束，无 CAD 模型"
            checked={config.enableFlexCable}
            disabled={disabled}
            onToggle={(v) => update({ enableFlexCable: v })}
          />
        </div>
      </section>

      {/* Section 3: Virtual RealSense D435i Material & Edge Noise Stress Model */}
      <section className="pt-0.5">
        <h2 className="text-[12px] font-semibold text-slate-300 tracking-wide mb-2 pb-1.5 border-b border-slate-800">
          RealSense D435i 真实材质与边缘噪声
        </h2>

        {/* One-Click Robustness Stress Presets */}
        <div className="grid grid-cols-3 gap-1 mb-3">
          {NOISE_PRESETS.map((preset) => {
            const isActive =
              Math.abs(config.sensor.edgeNoiseStrength - (preset.patch.edgeNoiseStrength ?? 1)) < 0.05 &&
              Math.abs(config.sensor.materialNoiseStrength - (preset.patch.materialNoiseStrength ?? 1)) < 0.05;
            return (
              <button
                key={preset.id}
                type="button"
                data-testid={`noise-preset-${preset.id}`}
                disabled={disabled}
                onClick={() => updateSensor(preset.patch)}
                className={`px-1.5 py-1 text-[10.5px] font-medium rounded-sm border transition-colors whitespace-nowrap ${
                  isActive
                    ? 'bg-amber-500/20 border-amber-400/70 text-amber-200'
                    : 'bg-[#090C12] border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-600'
                } disabled:opacity-50`}
              >
                {preset.label}
              </button>
            );
          })}
        </div>

        <div className="space-y-2.5">
          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-sigma" className="text-slate-300">
                距离平方深度噪声 σ₀
              </label>
              <span className="font-mono text-slate-200">{config.sensor.sigmaZ0.toFixed(4)}</span>
            </div>
            <input
              id="slider-sigma"
              type="range"
              min={0.0002}
              max={0.0032}
              step={0.0001}
              disabled={disabled}
              value={config.sensor.sigmaZ0}
              onChange={(e) => updateSensor({ sigmaZ0: Number(e.target.value) })}
              className="w-full lab-slider"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-edge-noise" className="text-slate-300">
                深度跳变边缘大噪声
              </label>
              <span className="font-mono text-amber-300">
                {config.sensor.edgeNoiseStrength.toFixed(1)}×
              </span>
            </div>
            <input
              id="slider-edge-noise"
              type="range"
              min={0}
              max={3.0}
              step={0.2}
              disabled={disabled}
              value={config.sensor.edgeNoiseStrength}
              onChange={(e) => updateSensor({ edgeNoiseStrength: Number(e.target.value) })}
              className="w-full lab-slider"
            />
            <div className="text-[10px] text-slate-500 mt-0.5">
              1px+2px 轮廓跳变双环飞点插值 + 重尾视差抖动
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-mat-noise" className="text-slate-300">
                材质光学退化 (反光/黑体)
              </label>
              <span className="font-mono text-amber-300">
                {config.sensor.materialNoiseStrength.toFixed(1)}×
              </span>
            </div>
            <input
              id="slider-mat-noise"
              type="range"
              min={0}
              max={2.5}
              step={0.1}
              disabled={disabled}
              value={config.sensor.materialNoiseStrength}
              onChange={(e) => updateSensor({ materialNoiseStrength: Number(e.target.value) })}
              className="w-full lab-slider"
            />
            <div className="text-[10px] text-slate-500 mt-0.5">
              金属多径漂移与高光空洞 · 哑黑 IR 吸光信噪比衰减
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-outlier" className="text-slate-300">
                立体匹配离群点率
              </label>
              <span className="font-mono text-slate-200">
                {(config.sensor.outlierRate * 100).toFixed(1)}%
              </span>
            </div>
            <input
              id="slider-outlier"
              type="range"
              min={0}
              max={0.05}
              step={0.002}
              disabled={disabled}
              value={config.sensor.outlierRate}
              onChange={(e) => updateSensor({ outlierRate: Number(e.target.value) })}
              className="w-full lab-slider"
            />
          </div>

          <div>
            <div className="flex items-center justify-between text-[11.5px] mb-1">
              <label htmlFor="slider-delay" className="text-slate-300">
                相机-编码器异步延迟
              </label>
              <span className="font-mono text-slate-200">{config.sensor.asyncDelayMs} ms</span>
            </div>
            <input
              id="slider-delay"
              type="range"
              min={0}
              max={50}
              step={5}
              disabled={disabled}
              value={config.sensor.asyncDelayMs}
              onChange={(e) => updateSensor({ asyncDelayMs: Number(e.target.value) })}
              className="w-full lab-slider"
            />
          </div>

          <div className="pt-0.5">
            <ToggleRow
              id="toggle-sensor-fp"
              label="边缘飞点退化 (Flying Pixels)"
              desc="深度跳变 > 5.5cm 处前景/背景随机插值"
              checked={config.sensor.enableFlyingPixels}
              disabled={disabled}
              onToggle={(v) => updateSensor({ enableFlyingPixels: v })}
            />
            <ToggleRow
              id="toggle-sensor-holes"
              label="掠射角·反光·黑体深度空洞"
              desc=">78° 掠射角、高光金属与深黑表面随机置零"
              checked={config.sensor.enableHoles}
              disabled={disabled}
              onToggle={(v) => updateSensor({ enableHoles: v })}
            />
          </div>
        </div>
      </section>
    </aside>
  );
};
