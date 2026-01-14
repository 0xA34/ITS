const API_BASE =
  (import.meta as any)?.env?.VITE_API_BASE?.toString?.() ||
  (import.meta as any)?.env?.VITE_API_URL?.toString?.() ||
  "http://localhost:8000";

export interface Camera {
  id: string;
  name: string;
  location: string;
  url: string;
}

export async function fetchCameras(
  limit = 200,
  skip = 0,
): Promise<{ items: Camera[]; total: number }> {
  const response = await fetch(
    `${API_BASE}/api/cameras?limit=${limit}&skip=${skip}`,
  );
  return response.json();
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  bbox: BoundingBox;
  class_name: string;
  class_id: number;
  confidence: number;
  track_id: number | null;
}

export interface DetectionResult {
  detections: Detection[];
  vehicle_count: Record<string, number>;
  total_count: number;
  frame_width: number;
  frame_height: number;
  processing_time_ms: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface ZonePolygon {
  id: string;
  name: string;
  points: Point[];
  is_parking_zone: boolean;
  is_traffic_light: boolean;
  is_red_light: boolean;
  is_stop_line: boolean;
  linked_traffic_light_id: string | null;
  is_counting_line?: boolean;
  counting_direction?: string;
  color: string;
}

export interface ParkingViolation {
  track_id: number;
  vehicle_class: string;
  zone_id: string;
  zone_name: string;
  duration_seconds: number;
  bbox: BoundingBox;
}

export interface CountingRecord {
  line_id: string;
  line_name: string;
  track_id: number;
  vehicle_class: string;
  direction: string;
  timestamp: string;
}

export interface LineCounts {
  line_id: string;
  line_name: string;
  total: number;
  count_in: number;
  count_out: number;
  by_class: Record<string, { in: number; out: number }>;
}

export interface DetectResponse {
  success: boolean;
  result: DetectionResult | null;
  violations: ParkingViolation[];
  error: string | null;
  model_info?: { model_key: string; model_name: string };
}

export interface TrafficStats {
  camera_id: string;
  timestamp: string;
  vehicle_counts: Record<string, number>;
  total_vehicles: number;
  parking_violations: ParkingViolation[];
  zones_occupancy: Record<string, number>;
}
//////////////// search
export type SearchSyncResponse = {
  mode: "sync";
  status: "success" | "empty";
  source?: any;
  detections: Array<any>;
  summary?: Record<string, number>;
  annotated?: { jpegBase64: string; width: number; height: number } | null;
};

export type SearchAsyncResponse = {
  mode: "async";
  status: "accepted";
  jobId: string;
  source: { kind: "video"; url: string; id?: string; name?: string };
};

export type SearchResponse = SearchSyncResponse | SearchAsyncResponse;

export async function postSearch(text: string, files: File[]): Promise<SearchResponse> {
  const fd = new FormData();
  fd.append("text", text ?? "");

  // backend nhận "files" (list)
  for (const f of files) fd.append("files", f);

  const res = await fetch("http://localhost:8000/api/search", {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`POST /api/search failed: ${res.status} ${msg}`);
  }

  return res.json();
}

export async function uploadImage(file: File) {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch("http://localhost:8000/api/upload-image", {
    method: "POST",
    body: fd,
  });

  if (!res.ok) throw new Error("upload failed");
  return res.json(); // { url: "..."}
}

////////////////
export async function detectVehicles(
  imageUrl: string,
  cameraId?: string,
  includeZones: boolean = false,
): Promise<DetectResponse> {
  const response = await fetch(`${API_BASE}/api/detection/detect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_url: imageUrl,
      camera_id: cameraId,
      include_zones: includeZones,
    }),
  });
  return response.json();
}

export async function getZones(
  cameraId: string,
): Promise<{ camera_id: string; zones: ZonePolygon[] }> {
  const response = await fetch(
    `${API_BASE}/api/detection/zones/${encodeURIComponent(cameraId)}`,
  );
  return response.json();
}

export async function saveZones(
  cameraId: string,
  zones: ZonePolygon[],
): Promise<{ success: boolean }> {
  const response = await fetch(
    `${API_BASE}/api/detection/zones/${encodeURIComponent(cameraId)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: cameraId, zones }),
    },
  );
  return response.json();
}

export async function addZone(
  cameraId: string,
  zone: ZonePolygon,
): Promise<{ success: boolean; zones: ZonePolygon[] }> {
  const response = await fetch(
    `${API_BASE}/api/detection/zones/${encodeURIComponent(cameraId)}/add`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(zone),
    },
  );
  return response.json();
}

export async function deleteZone(
  cameraId: string,
  zoneId: string,
): Promise<{ success: boolean; zones: ZonePolygon[] }> {
  const response = await fetch(
    `${API_BASE}/api/detection/zones/${encodeURIComponent(cameraId)}/${encodeURIComponent(zoneId)}`,
    {
      method: "DELETE",
    },
  );
  return response.json();
}

export async function resetTracker(
  cameraId: string,
): Promise<{ success: boolean }> {
  const response = await fetch(
    `${API_BASE}/api/detection/tracker/${encodeURIComponent(cameraId)}/reset`,
    {
      method: "POST",
    },
  );
  return response.json();
}

export async function getTrafficStats(
  cameraId: string,
  imageUrl: string,
): Promise<TrafficStats> {
  const params = new URLSearchParams({ image_url: imageUrl });
  const response = await fetch(
    `${API_BASE}/api/detection/stats/${encodeURIComponent(cameraId)}?${params}`,
  );
  return response.json();
}

export function createDetectionWebSocket(cameraId: string): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(
    `${wsBase}/api/detection/stream/${encodeURIComponent(cameraId)}`,
  );
}

// ==================== BENCHMARK API ====================

export interface ModelInfo {
  key: string;
  name: string;
  filename: string;
  version: string;
  size: string;
  description: string;
  params_millions: number;
  expected_map50: number;
  expected_fps: number;
  downloaded: boolean;
  path: string;
}

export interface BenchmarkResult {
  model_name: string;
  device: string;
  image_size: number;
  warmup_runs: number;
  benchmark_runs: number;
  avg_inference_time_ms: number;
  std_inference_time_ms: number;
  min_inference_time_ms: number;
  max_inference_time_ms: number;
  fps: number;
  memory_used_mb: number;
  timestamp: string;
}

export interface DeviceInfo {
  device: string;
  cuda_available: boolean;
  cuda_device_name?: string;
  cuda_device_count?: number;
  cuda_memory_total_mb?: number;
  cuda_memory_allocated_mb?: number;
  cuda_memory_cached_mb?: number;
}

export interface ListModelsResponse {
  models: ModelInfo[];
  current_model: string;
  device: string;
}

export interface DownloadModelResponse {
  success: boolean;
  model_key?: string;
  path?: string;
  error?: string;
}

export interface SwitchModelResponse {
  success: boolean;
  current_model?: string;
  previous_model?: string;
  error?: string;
}

export interface RunBenchmarkResponse {
  success: boolean;
  benchmark: BenchmarkResult;
  model_info: {
    name: string;
    version: string;
    size: string;
    params_millions: number;
    expected_map50: number;
    expected_fps: number;
  };
  comparison: {
    expected_fps: number;
    actual_fps: number;
    fps_difference: number;
  };
}

export interface BenchmarkAllResult {
  model_name: string;
  device: string;
  image_size: number;
  warmup_runs: number;
  benchmark_runs: number;
  avg_inference_time_ms: number;
  std_inference_time_ms: number;
  min_inference_time_ms: number;
  max_inference_time_ms: number;
  fps: number;
  memory_used_mb: number;
  timestamp: string;
  error?: string;
}

export interface RunBenchmarkAllResponse {
  success: boolean;
  device: string;
  total_models: number;
  results: BenchmarkAllResult[];
}

export interface ComparisonTableEntry {
  model_key: string;
  model_name: string;
  version: string;
  size: string;
  params_millions: number;
  expected_map50: number;
  expected_fps: number;
  downloaded: boolean;
  benchmarked: boolean;
  actual_fps?: number;
  actual_inference_ms?: number;
  fps_difference?: number;
}

export interface ComparisonTableResponse {
  comparison: ComparisonTableEntry[];
  device: string;
  current_model: string;
}

/**
 * List all available models with their information
 */
export async function listModels(): Promise<ListModelsResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/models`);
  if (!response.ok) {
    throw new Error(`Failed to list models: ${response.statusText}`);
  }
  return response.json();
}

/**
 * List only downloaded models
 */
export async function listDownloadedModels(): Promise<{
  models: ModelInfo[];
  current_model: string;
}> {
  const response = await fetch(`${API_BASE}/api/benchmark/models/downloaded`);
  if (!response.ok) {
    throw new Error(`Failed to list downloaded models: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get information about the currently active model
 */
export async function getCurrentModel(): Promise<{
  key: string;
  name: string;
  version: string;
  size: string;
  description: string;
  params_millions: number;
  expected_map50: number;
  expected_fps: number;
  device: string;
}> {
  const response = await fetch(`${API_BASE}/api/benchmark/models/current`);
  if (!response.ok) {
    throw new Error(`Failed to get current model: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Download a model if not already present
 */
export async function downloadModel(
  modelKey: string,
): Promise<DownloadModelResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_key: modelKey }),
  });
  if (!response.ok) {
    const error = await response.json();
    return { success: false, error: error.detail || response.statusText };
  }
  return response.json();
}

/**
 * Switch to a different model
 */
export async function switchModel(
  modelKey: string,
): Promise<SwitchModelResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/models/switch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_key: modelKey }),
  });
  if (!response.ok) {
    const error = await response.json();
    return { success: false, error: error.detail || response.statusText };
  }
  return response.json();
}

/**
 * Unload a model from memory
 */
export async function unloadModel(
  modelKey: string,
): Promise<{ success: boolean; error?: string }> {
  const response = await fetch(`${API_BASE}/api/benchmark/models/unload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_key: modelKey }),
  });
  if (!response.ok) {
    const error = await response.json();
    return { success: false, error: error.detail || response.statusText };
  }
  return response.json();
}

/**
 * Run benchmark on a specific model
 */
export async function runBenchmark(
  modelKey?: string,
  imageSize: number = 640,
  warmupRuns: number = 10,
  benchmarkRuns: number = 50,
): Promise<RunBenchmarkResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_key: modelKey,
      image_size: imageSize,
      warmup_runs: warmupRuns,
      benchmark_runs: benchmarkRuns,
    }),
  });
  if (!response.ok) {
    throw new Error(`Benchmark failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Run benchmark on all downloaded models
 */
export async function runBenchmarkAll(
  imageSize: number = 640,
  warmupRuns: number = 10,
  benchmarkRuns: number = 50,
): Promise<RunBenchmarkAllResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/run/all`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_size: imageSize,
      warmup_runs: warmupRuns,
      benchmark_runs: benchmarkRuns,
    }),
  });
  if (!response.ok) {
    throw new Error(`Benchmark all failed: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get all cached benchmark results
 */
export async function getBenchmarkResults(): Promise<{
  results: Record<string, BenchmarkResult>;
  device: string;
}> {
  const response = await fetch(`${API_BASE}/api/benchmark/results`);
  if (!response.ok) {
    throw new Error(`Failed to get benchmark results: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get comparison table of all models
 */
export async function getComparisonTable(): Promise<ComparisonTableResponse> {
  const response = await fetch(`${API_BASE}/api/benchmark/comparison`);
  if (!response.ok) {
    throw new Error(`Failed to get comparison table: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Get device information (CPU/GPU)
 */
export async function getDeviceInfo(): Promise<DeviceInfo> {
  const response = await fetch(`${API_BASE}/api/benchmark/device`);
  if (!response.ok) {
    throw new Error(`Failed to get device info: ${response.statusText}`);
  }
  return response.json();
}
