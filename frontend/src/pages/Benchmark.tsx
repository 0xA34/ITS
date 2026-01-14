import { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  listModels,
  downloadModel,
  switchModel,
  runBenchmark,
  runBenchmarkAll,
  getDeviceInfo,
  type ModelInfo,
  type BenchmarkResult,
  type DeviceInfo,
} from "@/lib/api";

interface BenchmarkResultWithInfo extends BenchmarkResult {
  model_info?: {
    name: string;
    version: string;
    size: string;
    params_millions: number;
    expected_map50: number;
    expected_fps: number;
  };
  comparison?: {
    expected_fps: number;
    actual_fps: number;
    fps_difference: number;
  };
}

export default function BenchmarkPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [currentModel, setCurrentModel] = useState<string>("");
  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const [benchmarking, setBenchmarking] = useState<string | null>(null);
  const [benchmarkResults, setBenchmarkResults] = useState<
    Record<string, BenchmarkResultWithInfo>
  >({});
  const [error, setError] = useState<string | null>(null);

  // Benchmark settings
  const [benchmarkRuns, setBenchmarkRuns] = useState(50);
  const [warmupRuns, setWarmupRuns] = useState(10);
  const [imageSize, setImageSize] = useState(640);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [modelsData, device] = await Promise.all([
        listModels(),
        getDeviceInfo(),
      ]);
      setModels(modelsData.models);
      setCurrentModel(modelsData.current_model);
      setDeviceInfo(device);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleDownload = async (modelKey: string) => {
    try {
      setDownloading(modelKey);
      setError(null);
      const result = await downloadModel(modelKey);
      if (result.success) {
        await loadData();
      } else {
        setError(result.error || "Download failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(null);
    }
  };

  const handleSwitch = async (modelKey: string) => {
    try {
      setSwitching(modelKey);
      setError(null);
      const result = await switchModel(modelKey);
      if (result.success) {
        setCurrentModel(result.current_model || modelKey);
        await loadData();
      } else {
        setError(result.error || "Switch failed");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Switch failed");
    } finally {
      setSwitching(null);
    }
  };

  const handleBenchmark = async (modelKey: string) => {
    try {
      setBenchmarking(modelKey);
      setError(null);
      const result = await runBenchmark(
        modelKey,
        imageSize,
        warmupRuns,
        benchmarkRuns
      );
      if (result.success) {
        setBenchmarkResults((prev) => ({
          ...prev,
          [modelKey]: {
            ...result.benchmark,
            model_info: result.model_info,
            comparison: result.comparison,
          },
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setBenchmarking(null);
    }
  };

  const handleBenchmarkAll = async () => {
    try {
      setBenchmarking("all");
      setError(null);
      const result = await runBenchmarkAll(imageSize, warmupRuns, benchmarkRuns);
      if (result.success) {
        const newResults: Record<string, BenchmarkResultWithInfo> = {};
        for (const r of result.results) {
          if (!r.error) {
            newResults[r.model_name] = r;
          }
        }
        setBenchmarkResults((prev) => ({ ...prev, ...newResults }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark all failed");
    } finally {
      setBenchmarking(null);
    }
  };

  const downloadedModels = models.filter((m) => m.downloaded);
  const notDownloadedModels = models.filter((m) => !m.downloaded);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Model Benchmark</h1>
          </div>
          <a
            href="/"
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition"
          >
            ← Quay lại
          </a>
        </div>

        {/* Error display */}
        {error && (
          <div className="bg-red-500/20 border border-red-500 text-red-300 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Device Info */}
        {deviceInfo && (
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="text-green-500">●</span> Device Information
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-700/50 rounded p-3">
                <p className="text-gray-400 text-sm">Device</p>
                <p className="font-semibold">
                  {deviceInfo.cuda_available ? "CUDA (GPU)" : "CPU"}
                </p>
              </div>
              {deviceInfo.cuda_device_name && (
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">GPU</p>
                  <p className="font-semibold">{deviceInfo.cuda_device_name}</p>
                </div>
              )}
              {deviceInfo.cuda_memory_total_mb && (
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">VRAM Total</p>
                  <p className="font-semibold">
                    {(deviceInfo.cuda_memory_total_mb / 1024).toFixed(1)} GB
                  </p>
                </div>
              )}
              {deviceInfo.cuda_memory_allocated_mb !== undefined && (
                <div className="bg-gray-700/50 rounded p-3">
                  <p className="text-gray-400 text-sm">VRAM Used</p>
                  <p className="font-semibold">
                    {deviceInfo.cuda_memory_allocated_mb.toFixed(0)} MB
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Benchmark Settings */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">⚙️ Benchmark Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-gray-400 text-sm mb-1">
                Benchmark Runs
              </label>
              <input
                type="number"
                value={benchmarkRuns}
                onChange={(e) => setBenchmarkRuns(Number(e.target.value))}
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={10}
                max={500}
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-1">
                Warmup Runs
              </label>
              <input
                type="number"
                value={warmupRuns}
                onChange={(e) => setWarmupRuns(Number(e.target.value))}
                className="w-full bg-gray-700 rounded px-3 py-2"
                min={1}
                max={50}
              />
            </div>
            <div>
              <label className="block text-gray-400 text-sm mb-1">
                Image Size
              </label>
              <select
                value={imageSize}
                onChange={(e) => setImageSize(Number(e.target.value))}
                className="w-full bg-gray-700 rounded px-3 py-2"
              >
                <option value={320}>320</option>
                <option value={416}>416</option>
                <option value={640}>640</option>
                <option value={1280}>1280</option>
              </select>
            </div>
          </div>
          <div className="mt-4">
            <Button
              onClick={handleBenchmarkAll}
              disabled={benchmarking !== null || downloadedModels.length === 0}
              className="bg-blue-600 hover:bg-blue-700"
            >
              {benchmarking === "all"
                ? "Đang benchmark tất cả..."
                : `🚀 Benchmark All Downloaded (${downloadedModels.length} models)`}
            </Button>
          </div>
        </div>

        {/* Current Model */}
        <div className="bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-blue-500/30 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-2">
            🎯 Model đang sử dụng: {currentModel}
          </h2>
          <p className="text-gray-400">
            {models.find((m) => m.key === currentModel)?.description}
          </p>
        </div>

        {/* Downloaded Models */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            ✅ Downloaded Models ({downloadedModels.length})
          </h2>
          <div className="grid gap-4">
            {downloadedModels.map((model) => (
              <ModelCard
                key={model.key}
                model={model}
                isCurrent={model.key === currentModel}
                benchmarkResult={benchmarkResults[model.key]}
                onSwitch={() => handleSwitch(model.key)}
                onBenchmark={() => handleBenchmark(model.key)}
                switching={switching === model.key}
                benchmarking={benchmarking === model.key}
              />
            ))}
          </div>
        </div>

        {/* Not Downloaded Models */}
        {notDownloadedModels.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              ⬇️ Available for Download ({notDownloadedModels.length})
            </h2>
            <div className="grid gap-4">
              {notDownloadedModels.map((model) => (
                <ModelCard
                  key={model.key}
                  model={model}
                  isCurrent={false}
                  onDownload={() => handleDownload(model.key)}
                  downloading={downloading === model.key}
                />
              ))}
            </div>
          </div>
        )}

        {/* Benchmark Results Comparison Table */}
        {Object.keys(benchmarkResults).length > 0 && (
          <div className="mt-8 bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">
              📊 Benchmark Comparison Table
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="py-3 px-4">Model</th>
                    <th className="py-3 px-4">Params (M)</th>
                    <th className="py-3 px-4">FPS</th>
                    <th className="py-3 px-4">Expected FPS</th>
                    <th className="py-3 px-4">Inference (ms)</th>
                    <th className="py-3 px-4">Memory (MB)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(benchmarkResults)
                    .sort(([, a], [, b]) => b.fps - a.fps)
                    .map(([key, result]) => (
                      <tr key={key} className="border-b border-gray-700/50">
                        <td className="py-3 px-4 font-medium">
                          {result.model_info?.name || key}
                        </td>
                        <td className="py-3 px-4">
                          {result.model_info?.params_millions.toFixed(1)}
                        </td>
                        <td className="py-3 px-4">
                          <span className="text-green-400 font-semibold">
                            {result.fps.toFixed(1)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-gray-400">
                          {result.model_info?.expected_fps}
                        </td>
                        <td className="py-3 px-4">
                          {result.avg_inference_time_ms.toFixed(2)}
                        </td>
                        <td className="py-3 px-4">
                          {result.memory_used_mb.toFixed(0)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Model Card Component
interface ModelCardProps {
  model: ModelInfo;
  isCurrent: boolean;
  benchmarkResult?: BenchmarkResultWithInfo;
  onSwitch?: () => void;
  onBenchmark?: () => void;
  onDownload?: () => void;
  switching?: boolean;
  benchmarking?: boolean;
  downloading?: boolean;
}

function ModelCard({
  model,
  isCurrent,
  benchmarkResult,
  onSwitch,
  onBenchmark,
  onDownload,
  switching,
  benchmarking,
  downloading,
}: ModelCardProps) {
  const sizeColors: Record<string, string> = {
    nano: "bg-green-500/20 text-green-400",
    small: "bg-blue-500/20 text-blue-400",
    medium: "bg-yellow-500/20 text-yellow-400",
    large: "bg-orange-500/20 text-orange-400",
    xlarge: "bg-red-500/20 text-red-400",
  };

  return (
    <div
      className={`bg-gray-800 rounded-lg p-5 border ${
        isCurrent ? "border-blue-500" : "border-gray-700"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h3 className="text-lg font-semibold">{model.name}</h3>
            <span
              className={`px-2 py-0.5 rounded text-xs font-medium ${
                sizeColors[model.size] || "bg-gray-500/20"
              }`}
            >
              {model.size}
            </span>
            {isCurrent && (
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-500 text-white">
                Active
              </span>
            )}
          </div>
          <p className="text-gray-400 text-sm mb-3">{model.description}</p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <span className="text-gray-500">Version:</span>{" "}
              <span className="text-white">{model.version}</span>
            </div>
            <div>
              <span className="text-gray-500">Params:</span>{" "}
              <span className="text-white">{model.params_millions}M</span>
            </div>
            <div>
              <span className="text-gray-500">Expected mAP@0.5:</span>{" "}
              <span className="text-white">
                {(model.expected_map50 * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="text-gray-500">Expected FPS:</span>{" "}
              <span className="text-white">{model.expected_fps}</span>
            </div>
          </div>

          {/* Benchmark Results */}
          {benchmarkResult && (
            <div className="mt-4 pt-4 border-t border-gray-700">
              <h4 className="text-sm font-medium text-gray-400 mb-2">
                📊 Benchmark Results
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="bg-gray-700/50 rounded p-2">
                  <span className="text-gray-500 block text-xs">FPS</span>
                  <span className="text-green-400 font-bold text-lg">
                    {benchmarkResult.fps.toFixed(1)}
                  </span>
                </div>
                <div className="bg-gray-700/50 rounded p-2">
                  <span className="text-gray-500 block text-xs">
                    Inference Time
                  </span>
                  <span className="text-white font-semibold">
                    {benchmarkResult.avg_inference_time_ms.toFixed(2)} ms
                  </span>
                </div>
                <div className="bg-gray-700/50 rounded p-2">
                  <span className="text-gray-500 block text-xs">Memory</span>
                  <span className="text-white font-semibold">
                    {benchmarkResult.memory_used_mb.toFixed(0)} MB
                  </span>
                </div>
                <div className="bg-gray-700/50 rounded p-2">
                  <span className="text-gray-500 block text-xs">
                    vs Expected
                  </span>
                  <span
                    className={`font-semibold ${
                      benchmarkResult.comparison &&
                      benchmarkResult.comparison.fps_difference >= 0
                        ? "text-green-400"
                        : "text-red-400"
                    }`}
                  >
                    {benchmarkResult.comparison
                      ? `${
                          benchmarkResult.comparison.fps_difference >= 0
                            ? "+"
                            : ""
                        }${benchmarkResult.comparison.fps_difference.toFixed(
                          1
                        )} FPS`
                      : "N/A"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2">
          {model.downloaded ? (
            <>
              {!isCurrent && onSwitch && (
                <Button
                  onClick={onSwitch}
                  disabled={switching}
                  className="bg-blue-600 hover:bg-blue-700 text-sm"
                >
                  {switching ? "..." : "Use This"}
                </Button>
              )}
              {onBenchmark && (
                <Button
                  onClick={onBenchmark}
                  disabled={benchmarking}
                  className="bg-purple-600 hover:bg-purple-700 text-sm"
                >
                  {benchmarking ? "Running..." : "Benchmark"}
                </Button>
              )}
            </>
          ) : (
            onDownload && (
              <Button
                onClick={onDownload}
                disabled={downloading}
                className="bg-green-600 hover:bg-green-700 text-sm"
              >
                {downloading ? "Downloading..." : "Download"}
              </Button>
            )
          )}
        </div>
      </div>
    </div>
  );
}
