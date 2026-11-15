import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronUp,
  RotateCcw,
  Shield,
  SlidersHorizontal,
  Upload,
  X,
} from 'lucide-react';
import {
  ATTACK_METHODS,
  AttackMethod,
  DEFENSE_METHODS,
  DefenseMethod,
  MODELS,
  PredictionState,
  SAMPLES,
  computeMockMetrics,
  renderMockPipelineDataUrls,
} from './mockData';

interface PanelCardProps {
  title: string;
  bandBg: string;
  badgeText: string;
  imageUrl: string;
  pred: PredictionState;
  barColor: string;
}

function ComparisonCard({
  title,
  bandBg,
  badgeText,
  imageUrl,
  pred,
  barColor,
}: PanelCardProps) {
  const pct = Math.max(0, Math.min(100, Math.round(pred.confidence * 100)));

  let truthLabel = '';
  if (pred.isCorrect === true) truthLabel = ' · GROUND TRUTH MATCH';
  else if (pred.isCorrect === false) truthLabel = ' · MISCLASSIFIED';

  return (
    <article className="bg-white border-2 border-[#1E242B] rounded-lg nb-shadow overflow-hidden flex flex-col">
      {/* Subtle Pastel Header Band */}
      <div
        className={`border-b-2 border-[#1E242B] px-4 py-2.5 flex items-center justify-between gap-2 ${bandBg}`}
      >
        <h2 className="font-display font-extrabold text-base tracking-tight text-[#1E242B] whitespace-nowrap">
          {title}
        </h2>
        <span className="font-mono-tabular text-[11px] font-bold bg-white border-[1.5px] border-[#1E242B] rounded px-2 py-0.5 text-[#1E242B] whitespace-nowrap truncate max-w-[170px]">
          {badgeText}
        </span>
      </div>

      {/* Consistent Square Preview */}
      <div className="p-4 pb-3 bg-white">
        <div className="border-2 border-[#1E242B] rounded-md bg-[#181A1F] aspect-square w-full overflow-hidden flex items-center justify-center">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt={`${title} sample preview`}
              className="w-full h-full object-cover block"
            />
          ) : (
            <div className="text-white font-mono-tabular text-xs">
              RENDERING PREVIEW
            </div>
          )}
        </div>
      </div>

      {/* Prediction & Metadata */}
      <div className="px-4 pb-4 pt-1 flex-1 flex flex-col justify-between gap-3">
        <div>
          <div className="font-mono-tabular text-[11px] font-bold uppercase tracking-wider text-[#5A626E]">
            Predicted Class
          </div>
          <div className="flex items-baseline justify-between gap-2 mt-0.5">
            <div className="font-display font-extrabold text-xl leading-tight text-[#1E242B] truncate">
              {pred.predictedClass}
            </div>
            <div className="font-mono-tabular font-bold text-lg text-[#1E242B] shrink-0">
              {(pred.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {/* Crisp confidence bar */}
          <div className="w-full h-3 bg-[#F3F2EC] border-[1.5px] border-[#1E242B] rounded-sm overflow-hidden mt-2">
            <div
              className="h-full border-r-[1.5px] border-[#1E242B] transition-[width] duration-150"
              style={{ width: `${pct}%`, backgroundColor: barColor }}
            />
          </div>

          <div className="font-mono-tabular text-[11px] font-bold text-[#333A44] mt-2">
            {pred.statusTag}
            {truthLabel}
          </div>
        </div>

        <div className="bg-[#FAF9F5] border-[1.5px] border-[#1E242B] rounded-md px-3 py-2 font-mono-tabular text-xs leading-relaxed text-[#2C333D]">
          {pred.metadataLines.map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      </div>
    </article>
  );
}

export default function App() {
  const [sampleId, setSampleId] = useState<string>(SAMPLES[0].id);
  const [modelId, setModelId] = useState<string>(SAMPLES[0].modelId);
  const [attackMethod, setAttackMethod] = useState<AttackMethod>('FGSM');
  const [epsilon, setEpsilon] = useState<number>(0.08);
  const [defenseMethod, setDefenseMethod] = useState<DefenseMethod>('PGD-AT');

  // Advanced Settings
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false);
  const [pgdSteps, setPgdSteps] = useState<number>(10);
  const [pgdAlpha, setPgdAlpha] = useState<number>(0.01);
  const [pgdRestarts, setPgdRestarts] = useState<number>(1);
  const [blurRadius, setBlurRadius] = useState<number>(1.8);
  const [bitDepth, setBitDepth] = useState<number>(3);
  const [visualAmp, setVisualAmp] = useState<number>(1.2);

  // Uploaded Image State
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadedImgEl, setUploadedImgEl] = useState<HTMLImageElement | null>(
    null
  );

  // Rendered Preview Data URLs
  const [previews, setPreviews] = useState<{
    originalUrl: string;
    attackedUrl: string;
    defendedUrl: string;
    meanSeed: number;
  }>({
    originalUrl: '',
    attackedUrl: '',
    defendedUrl: '',
    meanSeed: 0,
  });

  useEffect(() => {
    const out = renderMockPipelineDataUrls({
      sourceImage: uploadedImgEl,
      sampleId,
      attackMethod,
      epsilon,
      defenseMethod,
      blurRadius,
      bitDepth,
      visualAmp,
    });
    setPreviews(out);
  }, [
    uploadedImgEl,
    sampleId,
    attackMethod,
    epsilon,
    defenseMethod,
    blurRadius,
    bitDepth,
    visualAmp,
  ]);

  const analysis = useMemo(
    () =>
      computeMockMetrics({
        modelId,
        attackMethod,
        epsilon,
        defenseMethod,
        sampleId,
        uploadedImageName: uploadedFileName,
        uploadedMeanSeed: uploadedImgEl ? previews.meanSeed : null,
        pgdSteps,
        pgdAlpha,
        pgdRestarts,
        blurRadius,
        bitDepth,
        visualAmp,
      }),
    [
      modelId,
      attackMethod,
      epsilon,
      defenseMethod,
      sampleId,
      uploadedFileName,
      uploadedImgEl,
      previews.meanSeed,
      pgdSteps,
      pgdAlpha,
      pgdRestarts,
      blurRadius,
      bitDepth,
      visualAmp,
    ]
  );

  const handleSampleChange = (newSampleId: string) => {
    setSampleId(newSampleId);
    const found = SAMPLES.find((s) => s.id === newSampleId);
    if (found) {
      setModelId(found.modelId);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        setUploadedImgEl(img);
        setUploadedFileName(file.name);
      };
      if (typeof reader.result === 'string') {
        img.src = reader.result;
      }
    };
    reader.readAsDataURL(file);
  };

  const clearUpload = () => {
    setUploadedImgEl(null);
    setUploadedFileName(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const resetDefaults = () => {
    clearUpload();
    setSampleId(SAMPLES[0].id);
    setModelId(SAMPLES[0].modelId);
    setAttackMethod('FGSM');
    setEpsilon(0.08);
    setDefenseMethod('PGD-AT');
    setPgdSteps(10);
    setPgdAlpha(0.01);
    setPgdRestarts(1);
    setBlurRadius(1.8);
    setBitDepth(3);
    setVisualAmp(1.2);
  };

  const confDelta =
    analysis.defendedPred.confidence - analysis.attackedPred.confidence;
  const confDeltaFormatted = `${confDelta >= 0 ? '+' : ''}${(
    confDelta * 100
  ).toFixed(1)}%`;

  return (
    <div className="min-h-screen bg-[#FAF9F5] text-[#1E242B] flex flex-col lg:flex-row">
      {/* LEFT CONFIGURATION SIDEBAR */}
      <aside className="w-full lg:w-[330px] shrink-0 bg-white border-b-2 lg:border-b-0 lg:border-r-2 border-[#1E242B] p-5 flex flex-col gap-4">
        <div className="bg-[#FEF3C7] border-2 border-[#1E242B] rounded-md nb-shadow-sm px-3.5 py-2.5 flex items-center justify-between gap-2">
          <div>
            <div className="font-display font-extrabold text-sm tracking-tight leading-tight">
              CONFIGURATION
            </div>
            <div className="font-mono-tabular text-[11px] font-bold text-[#4B5563] mt-0.5">
              MOCK CONTROLS · LIVE UPDATE
            </div>
          </div>
          <SlidersHorizontal className="w-4 h-4 shrink-0" strokeWidth={2.25} />
        </div>

        {/* 1. Image Input & Sample Selector */}
        <section className="flex flex-col gap-2">
          <label className="font-display font-bold text-xs uppercase tracking-wider text-[#333A44]">
            1. Input Image
          </label>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleFileUpload}
            className="hidden"
            id="image-upload-input"
          />

          <div className="flex items-stretch gap-2">
            <label
              htmlFor="image-upload-input"
              className="flex-1 cursor-pointer bg-[#FAF9F5] hover:bg-[#FEF3C7] border-2 border-[#1E242B] rounded-md nb-shadow-btn px-3 py-2 flex items-center justify-center gap-2 font-display font-bold text-sm transition-colors"
            >
              <Upload className="w-4 h-4 shrink-0" strokeWidth={2.25} />
              <span className="truncate">
                {uploadedFileName ? uploadedFileName : 'Upload Custom Image'}
              </span>
            </label>
            {uploadedFileName && (
              <button
                type="button"
                onClick={clearUpload}
                aria-label="Clear uploaded image"
                className="bg-[#FECACA] hover:bg-[#FCA5A5] border-2 border-[#1E242B] rounded-md nb-shadow-btn px-2.5 flex items-center justify-center cursor-pointer"
              >
                <X className="w-4 h-4" strokeWidth={2.25} />
              </button>
            )}
          </div>

          <div className="flex flex-col gap-1 mt-0.5">
            <span className="font-mono-tabular text-[11px] font-bold text-[#5A626E]">
              DEMO SAMPLE PRESET
            </span>
            <select
              value={sampleId}
              disabled={Boolean(uploadedImgEl)}
              onChange={(e) => handleSampleChange(e.target.value)}
              className="w-full bg-white disabled:bg-[#F1F5F9] disabled:opacity-60 border-2 border-[#1E242B] rounded-md nb-shadow-sm px-3 py-2 font-semibold text-sm cursor-pointer focus:outline-none"
            >
              {SAMPLES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
        </section>

        <hr className="border-t border-[#D6D3C9]" />

        {/* 2. Model Selection */}
        <section className="flex flex-col gap-1.5">
          <label
            htmlFor="model-select"
            className="font-display font-bold text-xs uppercase tracking-wider text-[#333A44]"
          >
            2. Model Selection
          </label>
          <select
            id="model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="w-full bg-white border-2 border-[#1E242B] rounded-md nb-shadow-sm px-3 py-2 font-semibold text-sm cursor-pointer focus:outline-none"
          >
            {Object.values(MODELS).map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>
          <div className="font-mono-tabular text-[11px] text-[#5A626E]">
            {MODELS[modelId].dataset} · Clean Acc: {MODELS[modelId].cleanAccuracy}
          </div>
        </section>

        <hr className="border-t border-[#D6D3C9]" />

        {/* 3. Attack Method & Epsilon */}
        <section className="flex flex-col gap-2.5">
          <div className="font-display font-bold text-xs uppercase tracking-wider text-[#333A44]">
            3. Adversarial Attack
          </div>

          <div className="grid grid-cols-2 gap-2">
            {ATTACK_METHODS.map((method) => {
              const active = attackMethod === method;
              return (
                <button
                  key={method}
                  type="button"
                  onClick={() => setAttackMethod(method)}
                  className={`py-2 px-3 border-2 border-[#1E242B] rounded-md font-display font-extrabold text-sm cursor-pointer nb-shadow-btn ${
                    active
                      ? 'bg-[#FECACA] text-[#1E242B]'
                      : 'bg-white text-[#1E242B] hover:bg-[#FAF9F5]'
                  }`}
                >
                  {method}
                </button>
              );
            })}
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label
                htmlFor="epsilon-slider"
                className="font-mono-tabular text-xs font-bold"
              >
                EPSILON (ε)
              </label>
              <span className="font-mono-tabular text-xs font-bold bg-[#FEF3C7] border-[1.5px] border-[#1E242B] rounded px-2 py-0.5">
                {epsilon.toFixed(2)}
              </span>
            </div>
            <input
              id="epsilon-slider"
              type="range"
              min={0}
              max={0.3}
              step={0.01}
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="nb-range"
            />
            <div className="flex justify-between font-mono-tabular text-[10px] font-bold text-[#5A626E] mt-1">
              <span>0.00 (CLEAN)</span>
              <span>0.15</span>
              <span>0.30 (STRONG)</span>
            </div>
          </div>
        </section>

        <hr className="border-t border-[#D6D3C9]" />

        {/* 4. Defense Method */}
        <section className="flex flex-col gap-2">
          <div className="font-display font-bold text-xs uppercase tracking-wider text-[#333A44]">
            4. Defense Method
          </div>
          <div className="grid grid-cols-1 gap-2">
            {DEFENSE_METHODS.map((def) => {
              const active = defenseMethod === def;
              return (
                <button
                  key={def}
                  type="button"
                  onClick={() => setDefenseMethod(def)}
                  className={`w-full text-left px-3 py-2 border-2 border-[#1E242B] rounded-md font-display font-bold text-sm cursor-pointer flex items-center justify-between nb-shadow-btn ${
                    active
                      ? 'bg-[#D1FAE5] text-[#1E242B]'
                      : 'bg-white text-[#1E242B] hover:bg-[#FAF9F5]'
                  }`}
                >
                  <span>{def}</span>
                  <span className="font-mono-tabular text-[11px] font-bold text-[#333A44]">
                    {active ? 'ACTIVE' : 'SELECT'}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        {/* 5. Advanced Settings Accordion */}
        <section className="border-2 border-[#1E242B] rounded-md bg-[#FAF9F5] nb-shadow-sm overflow-hidden">
          <button
            type="button"
            onClick={() => setAdvancedOpen((prev) => !prev)}
            className="w-full px-3.5 py-2 flex items-center justify-between font-display font-bold text-xs uppercase tracking-wider cursor-pointer bg-[#FAF9F5] hover:bg-[#FEF3C7] transition-colors"
          >
            <span>Advanced Settings</span>
            {advancedOpen ? (
              <ChevronUp className="w-4 h-4" strokeWidth={2.25} />
            ) : (
              <ChevronDown className="w-4 h-4" strokeWidth={2.25} />
            )}
          </button>

          {advancedOpen && (
            <div className="p-3.5 pt-2.5 border-t-2 border-[#1E242B] flex flex-col gap-3 bg-white">
              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>PGD STEPS</span>
                  <span>{pgdSteps}</span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={40}
                  step={1}
                  value={pgdSteps}
                  onChange={(e) => setPgdSteps(parseInt(e.target.value, 10))}
                  className="nb-range"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>PGD STEP SIZE (α)</span>
                  <span>{pgdAlpha.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min={0.002}
                  max={0.05}
                  step={0.002}
                  value={pgdAlpha}
                  onChange={(e) => setPgdAlpha(parseFloat(e.target.value))}
                  className="nb-range"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>PGD RESTARTS</span>
                  <span>{pgdRestarts}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={5}
                  step={1}
                  value={pgdRestarts}
                  onChange={(e) => setPgdRestarts(parseInt(e.target.value, 10))}
                  className="nb-range"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>GAUSSIAN BLUR σ</span>
                  <span>{blurRadius.toFixed(1)}px</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={4.0}
                  step={0.1}
                  value={blurRadius}
                  onChange={(e) => setBlurRadius(parseFloat(e.target.value))}
                  className="nb-range"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>BIT-DEPTH BITS</span>
                  <span>{bitDepth}-bit</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={6}
                  step={1}
                  value={bitDepth}
                  onChange={(e) => setBitDepth(parseInt(e.target.value, 10))}
                  className="nb-range"
                />
              </div>

              <div>
                <div className="flex justify-between font-mono-tabular text-xs font-bold mb-1">
                  <span>VISUAL NOISE AMP</span>
                  <span>{visualAmp.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={2.5}
                  step={0.1}
                  value={visualAmp}
                  onChange={(e) => setVisualAmp(parseFloat(e.target.value))}
                  className="nb-range"
                />
              </div>
            </div>
          )}
        </section>

        <button
          type="button"
          onClick={resetDefaults}
          className="mt-auto bg-white hover:bg-[#FEF3C7] border-2 border-[#1E242B] rounded-md nb-shadow-btn py-2 px-4 font-display font-bold text-sm flex items-center justify-center gap-2 cursor-pointer"
        >
          <RotateCcw className="w-4 h-4" strokeWidth={2.25} />
          <span>Reset Demo Defaults</span>
        </button>
      </aside>

      {/* MAIN WORKSPACE */}
      <main className="flex-1 min-w-0 p-5 md:p-7 flex flex-col gap-5 max-w-[1440px]">
        {/* Header Bar */}
        <header className="bg-white border-2 border-[#1E242B] rounded-lg nb-shadow px-5 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 bg-[#FEF3C7] border-2 border-[#1E242B] rounded-md flex items-center justify-center shrink-0">
              <Shield className="w-5 h-5 text-[#1E242B]" strokeWidth={2.25} />
            </div>
            <div>
              <h1 className="font-display font-black text-2xl tracking-tight leading-none">
                Adversarial Vision Lab
              </h1>
              <p className="text-sm font-medium text-[#4B5563] mt-1">
                Single-page attack &amp; defense comparison workspace · Layout inspired by Irene Adversarial Vision Lab
              </p>
            </div>
          </div>

          <div className="bg-[#FEF3C7] border-2 border-[#1E242B] rounded-md nb-shadow-sm px-3 py-1.5 font-mono-tabular font-bold text-xs whitespace-nowrap shrink-0">
            DEMO MODE · MOCK DATA ONLY
          </div>
        </header>

        {/* Three Large Side-by-Side Image Comparison Cards */}
        <section
          aria-label="Original, Attacked, and Defended visual comparison"
          className="grid grid-cols-1 md:grid-cols-3 gap-5"
        >
          <ComparisonCard
            title="ORIGINAL"
            bandBg="bg-[#F1F5F9]"
            badgeText="CLEAN"
            imageUrl={previews.originalUrl}
            pred={analysis.originalPred}
            barColor="#60A5FA"
          />

          <ComparisonCard
            title="ATTACKED"
            bandBg="bg-[#FFE4DF]"
            badgeText={`${attackMethod} · ε=${epsilon.toFixed(2)}`}
            imageUrl={previews.attackedUrl}
            pred={analysis.attackedPred}
            barColor="#F87171"
          />

          <ComparisonCard
            title="DEFENDED"
            bandBg="bg-[#DCFCE7]"
            badgeText={defenseMethod.toUpperCase()}
            imageUrl={previews.defendedUrl}
            pred={analysis.defendedPred}
            barColor="#34D399"
          />
        </section>

        {/* Analysis Summary Section */}
        <section className="bg-white border-2 border-[#1E242B] rounded-lg nb-shadow overflow-hidden">
          <div className="bg-[#FEF3C7] border-b-2 border-[#1E242B] px-5 py-2.5 flex flex-wrap items-center justify-between gap-2">
            <h2 className="font-display font-extrabold text-base tracking-tight">
              ANALYSIS SUMMARY
            </h2>
            <span className="font-mono-tabular text-[11px] font-bold bg-white border-[1.5px] border-[#1E242B] rounded px-2.5 py-0.5">
              SIMULATED METRICS · NO REAL INFERENCE
            </span>
          </div>

          <div className="p-5 flex flex-col gap-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3.5">
              <div className="bg-[#FAF9F5] border-[1.5px] border-[#1E242B] rounded-md px-3.5 py-2.5">
                <div className="font-mono-tabular text-[11px] font-bold uppercase text-[#5A626E]">
                  Selected Attack
                </div>
                <div className="font-display font-extrabold text-base mt-0.5">
                  {attackMethod} (ε = {epsilon.toFixed(2)})
                </div>
              </div>

              <div className="bg-[#FAF9F5] border-[1.5px] border-[#1E242B] rounded-md px-3.5 py-2.5">
                <div className="font-mono-tabular text-[11px] font-bold uppercase text-[#5A626E]">
                  Selected Defense
                </div>
                <div className="font-display font-extrabold text-base mt-0.5">
                  {defenseMethod}
                </div>
              </div>

              <div className="bg-[#FAF9F5] border-[1.5px] border-[#1E242B] rounded-md px-3.5 py-2.5">
                <div className="font-mono-tabular text-[11px] font-bold uppercase text-[#5A626E]">
                  Mock Prediction Shift
                </div>
                <div className="font-display font-extrabold text-base mt-0.5 leading-snug">
                  {analysis.originalPred.predictedClass} →{' '}
                  {analysis.attackedPred.predictedClass} →{' '}
                  {analysis.defendedPred.predictedClass}
                </div>
              </div>

              <div className="bg-[#FAF9F5] border-[1.5px] border-[#1E242B] rounded-md px-3.5 py-2.5">
                <div className="font-mono-tabular text-[11px] font-bold uppercase text-[#5A626E]">
                  Defense Confidence Delta
                </div>
                <div className="font-mono-tabular font-bold text-base mt-0.5">
                  {confDeltaFormatted} vs Attacked
                </div>
              </div>
            </div>

            <div className="border-t border-[#D6D3C9] pt-3.5">
              <h3 className="font-display font-extrabold text-base text-[#1E242B]">
                {analysis.summaryHeadline}
              </h3>
              <p className="text-sm text-[#333A44] leading-relaxed mt-1">
                {analysis.summaryExplanation}
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
