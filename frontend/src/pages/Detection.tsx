import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import Hls from "hls.js";
import { Button } from "@/components/ui/button";
import DetectionViewer from "@/components/DetectionViewer";
import PolygonEditor from "@/components/PolygonEditor";
import TrafficDashboard from "@/components/TrafficDashboard";

import {
    detectVehicles,
    getZones,
    addZone,
    deleteZone,
    saveZones,
    type DetectionResult,
    type ZonePolygon,
    type ParkingViolation,
    type LineCounts,
} from "@/lib/api";

const API_BASE =
    (import.meta as any)?.env?.VITE_API_BASE?.toString?.() ||
    (import.meta as any)?.env?.VITE_API_URL?.toString?.() ||
    "http://localhost:8000";

const WS_BASE = API_BASE.replace(/^http/, "ws");

export default function DetectionPage() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();

    const cameraId = searchParams.get("id") || "";
    const cameraUrl = searchParams.get("url") || "";
    const cameraName = searchParams.get("name") || "Camera";

    const containerRef = useRef<HTMLDivElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const hlsRef = useRef<Hls | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const [containerSize, setContainerSize] = useState({ width: 800, height: 450 });
    const [isDetecting, setIsDetecting] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [showDashboard, setShowDashboard] = useState(true);
    const [showDensitySection, setShowDensitySection] = useState(false);
    const [debugInfo, setDebugInfo] = useState("");

    const [result, setResult] = useState<DetectionResult | null>(null);
    const [violations, setViolations] = useState<ParkingViolation[]>([]);
    const [zones, setZones] = useState<ZonePolygon[]>([]);
    const [frameSize, setFrameSize] = useState({ width: 1920, height: 1080 });
    const [timestamp, setTimestamp] = useState(Date.now());
    const [syncedFrame, setSyncedFrame] = useState<string | null>(null);
    const [currentModel, setCurrentModel] = useState<string>("");
    const [lineCounts, setLineCounts] = useState<Record<string, LineCounts>>({});

    const [classFilter, setClassFilter] = useState<Record<string, boolean>>({
        person: true,
        car: true,
        motorcycle: true,
        bus: true,
        truck: true,
        bicycle: true,
    });

    const [featureFilter, setFeatureFilter] = useState<Record<string, boolean>>({
        traffic_light: true,
        parking_zone: true,
        counting_line: true,
        stop_line: true,
    });

    const detectIntervalRef = useRef<number | null>(null);
    const urlLower = cameraUrl.toLowerCase();
    const isVideo =
        urlLower.includes("/api/media/") ||
        urlLower.includes(".m3u8") ||
        urlLower.includes(".mp4") ||
        urlLower.includes(".mov") ||
        urlLower.includes(".mkv") ||
        urlLower.includes(".avi") ||
        urlLower.includes(".webm");

    const isM3u8 = urlLower.includes(".m3u8");

    const getProxiedImageUrl = useCallback(() => {
        const cleaned = cameraUrl.replace(/[?&]t=\d+/g, "");
        const sep = cleaned.includes("?") ? "&" : "?";
        return `${API_BASE}/api/proxy/image?url=${encodeURIComponent(`${cleaned}${sep}t=${timestamp}`)}`;
    }, [cameraUrl, timestamp]);

    const getProxiedHlsUrl = useCallback(() => {
        return `${API_BASE}/api/proxy/hls?url=${encodeURIComponent(cameraUrl)}`;
    }, [cameraUrl]);

    useEffect(() => {
        if (!cameraId) return;
        getZones(cameraId).then((data) => setZones(data.zones || [])).catch(console.error);
    }, [cameraId]);

    // Fetch current model info when page loads
    useEffect(() => {
        fetch(`${API_BASE}/api/benchmark/models/current`)
            .then(res => res.json())
            .then(data => {
                if (data.name) {
                    setCurrentModel(data.name);
                }
            })
            .catch(err => console.error("Failed to fetch current model:", err));
    }, []);

    useEffect(() => {
        if (!isM3u8) {
            const interval = setInterval(() => setTimestamp(Date.now()), 12000);
            return () => clearInterval(interval);
        }
    }, [isM3u8]);

    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect();
                setContainerSize({ width: rect.width, height: rect.height });
                console.log("Container size:", rect.width, rect.height);
            }
        };
        updateSize();
        window.addEventListener("resize", updateSize);
        return () => window.removeEventListener("resize", updateSize);
    }, []);

    useEffect(() => {
        if (!isM3u8 || !videoRef.current) return;
        if (isDetecting && syncedFrame) return;

        const video = videoRef.current;
        const hlsUrl = getProxiedHlsUrl();
        console.log("Loading HLS:", hlsUrl);

        if (video.canPlayType("application/vnd.apple.mpegurl")) {
            video.src = hlsUrl;
            video.play().catch(() => { });
            return;
        }

        if (Hls.isSupported()) {
            if (hlsRef.current) {
                hlsRef.current.destroy();
            }
            const hls = new Hls({ lowLatencyMode: true, enableWorker: true });
            hls.loadSource(hlsUrl);
            hls.attachMedia(video);
            hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(() => { }));
            hlsRef.current = hls;
        }

        return () => {
            if (hlsRef.current) {
                hlsRef.current.destroy();
                hlsRef.current = null;
            }
        };
    }, [isM3u8, getProxiedHlsUrl, isDetecting, syncedFrame]);

    const captureVideoFrame = useCallback((): string | null => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) {
            console.log("Video not ready:", video?.readyState);
            return null;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return null;

        ctx.drawImage(video, 0, 0);
        console.log("Captured frame:", video.videoWidth, video.videoHeight);
        setFrameSize({ width: video.videoWidth, height: video.videoHeight });
        return canvas.toDataURL("image/jpeg", 0.7);
    }, []);

    const runDetectionForImage = useCallback(async () => {
        if (!cameraUrl || !cameraId) return;

        const cleaned = cameraUrl.replace(/[?&]t=\d+/g, "");
        const sep = cleaned.includes("?") ? "&" : "?";
        const urlWithT = `${cleaned}${sep}t=${Date.now()}`;
        console.log("Detecting image:", urlWithT);

        try {
            const response = await detectVehicles(urlWithT, cameraId, true);
            console.log("Detection response:", response);
            if (response.success && response.result) {
                setResult(response.result);
                setViolations(response.violations || []);
                setFrameSize({ width: response.result.frame_width, height: response.result.frame_height });
                setDebugInfo(`Detected: ${response.result.total_count} vehicles`);
                if (response.model_info) {
                    setCurrentModel(response.model_info.model_name || response.model_info.model_key);
                }
            } else {
                setDebugInfo(`Error: ${response.error || "No result"}`);
            }
        } catch (e: any) {
            console.error("Detection failed:", e);
            setDebugInfo(`Error: ${e.message}`);
        }
    }, [cameraUrl, cameraId]);

    // Capture current frame from video player
    const captureFrameFromVideo = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;

        if (!video || !canvas || video.readyState < 2) {
            console.log("Video not ready for capture");
            return null;
        }

        try {
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            const ctx = canvas.getContext('2d');

            if (!ctx) return null;

            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
            return dataUrl;
        } catch (e) {
            console.error("Failed to capture frame:", e);
            return null;
        }
    }, []);

    const runDetectionForVideo = useCallback(async () => {
        if (!cameraId) return;

        // Capture current frame from video player
        const frameDataUrl = captureFrameFromVideo();
        if (!frameDataUrl) {
            setDebugInfo("Waiting for video to load...");
            return;
        }

        console.log("Detecting from captured video frame");
        setDebugInfo("Detecting from video...");

        try {
            const response = await detectVehicles(frameDataUrl, cameraId, true);
            console.log("Video detection response:", response);
            if (response.success && response.result) {
                setResult(response.result);
                setViolations(response.violations || []);
                setFrameSize({ width: response.result.frame_width, height: response.result.frame_height });
                setDebugInfo(`Detected: ${response.result.total_count} vehicles (${response.result.processing_time_ms.toFixed(0)}ms)`);
                if (response.model_info) {
                    setCurrentModel(response.model_info.model_name || response.model_info.model_key);
                }
            } else {
                setDebugInfo(`Error: ${response.error || "No result"}`);
            }
        } catch (e: any) {
            console.error("Video detection failed:", e);
            setDebugInfo(`Error: ${e.message}`);
        }
    }, [cameraId, captureFrameFromVideo]);

    useEffect(() => {
        if (isDetecting) {
            if (isVideo) {
                // ✅ Use WebSocket for ALL videos (HLS + uploaded MP4)
                const ws = new WebSocket(`${WS_BASE}/api/detection/video-stream/${encodeURIComponent(cameraId)}`);
                wsRef.current = ws;

                ws.onopen = () => {
                    console.log("WebSocket connected for video detection");
                    setDebugInfo("Connecting to video stream...");
                    const activeClasses = Object.entries(classFilter).filter(([_, v]) => v).map(([k]) => k);
                    ws.send(JSON.stringify({
                        video_url: cameraUrl,
                        send_frame: true,
                        class_filter: activeClasses.length < 6 ? activeClasses : null,
                        feature_filter: featureFilter
                    }));
                };

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === "detection_result") {
                            setResult(data.result);
                            setViolations(data.violations || []);
                            setFrameSize({ width: data.result.frame_width, height: data.result.frame_height });
                            setDebugInfo(`${data.result.total_count} vehicles (${data.result.processing_time_ms.toFixed(0)}ms)`);
                            if (data.frame) setSyncedFrame(data.frame);
                            if (data.line_counts) setLineCounts(data.line_counts);
                        } else if (data.type === "connected") {
                            setDebugInfo("Video stream connected!");
                        } else if (data.type === "error") {
                            setDebugInfo(`Error: ${data.error}`);
                        }
                    } catch (e) {
                        console.error("WS message parse error:", e);
                    }
                };

                ws.onerror = () => setDebugInfo("WebSocket error");
                ws.onclose = () => console.log("WebSocket closed");

            } else {
                runDetectionForImage();
                detectIntervalRef.current = window.setInterval(runDetectionForImage, 3000);
            }
        } else {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (detectIntervalRef.current) {
                clearInterval(detectIntervalRef.current);
                detectIntervalRef.current = null;
            }
            setResult(null);
            setSyncedFrame(null);
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            if (detectIntervalRef.current) clearInterval(detectIntervalRef.current);
        };
    }, [isDetecting, isVideo, isM3u8, cameraId, cameraUrl, runDetectionForImage, runDetectionForVideo]);


    const handleVideoLoad = useCallback(() => {
        const video = videoRef.current;
        if (video) {
            console.log("Video loaded:", video.videoWidth, video.videoHeight);
            setFrameSize({ width: video.videoWidth || 1920, height: video.videoHeight || 1080 });
        }
    }, []);

    const handleImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
        const img = e.currentTarget;
        console.log("Image loaded:", img.naturalWidth, img.naturalHeight);
        setFrameSize({ width: img.naturalWidth, height: img.naturalHeight });
    }, []);

    const handleZoneAdd = useCallback(async (zone: ZonePolygon) => {
        if (!cameraId) return;
        const response = await addZone(cameraId, zone);
        if (response.success) setZones(response.zones);
    }, [cameraId]);

    const handleZoneDelete = useCallback(async (zoneId: string) => {
        if (!cameraId) return;
        const response = await deleteZone(cameraId, zoneId);
        if (response.success) setZones(response.zones);
    }, [cameraId]);

    const handleZonesClear = useCallback(async () => {
        if (!cameraId) return;
        await saveZones(cameraId, []);
        setZones([]);
    }, [cameraId]);

    const handleZoneUpdate = useCallback(async (updatedZone: ZonePolygon) => {
        if (!cameraId) return;
        const newZones = zones.map(z => z.id === updatedZone.id ? updatedZone : z);
        await saveZones(cameraId, newZones);
        setZones(newZones);

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: "update_zones" }));
        }
    }, [cameraId, zones]);

    const handleClassFilterToggle = useCallback((className: string) => {
        setClassFilter(prev => {
            const newFilter = { ...prev, [className]: !prev[className] };
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                const activeClasses = Object.entries(newFilter).filter(([_, v]) => v).map(([k]) => k);
                wsRef.current.send(JSON.stringify({
                    type: "update_class_filter",
                    class_filter: activeClasses.length < 6 ? activeClasses : null
                }));
            }
            return newFilter;
        });
    }, []);

    if (!cameraUrl) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold mb-4">No Camera Selected</h1>
                    <Button onClick={() => navigate("/")}>Go Back</Button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background">
            <header className="bg-card border-b border-border p-4">
                <div className="flex items-center gap-4 flex-wrap">
                    <Button variant="outline" size="sm" onClick={() => navigate("/")}>← Back</Button>
                    <h1 className="text-xl font-bold">{cameraName}</h1>
                    <span className="text-xs text-muted-foreground px-2 py-1 bg-muted rounded">
                        {isM3u8 ? "🎥 Live Stream" : "📷 Snapshot"}
                    </span>
                    <span className="text-xs text-green-500">{frameSize.width}x{frameSize.height}</span>
                    {currentModel && (
                        <span className="text-xs px-2 py-1 bg-blue-600/20 text-blue-400 rounded border border-blue-500/30">
                            🤖 {currentModel}
                        </span>
                    )}
                    <div className="flex-1" />
                    <Button
                        variant={isDetecting ? "destructive" : "default"}
                        onClick={() => setIsDetecting(!isDetecting)}
                        disabled={isEditing}
                        className={isDetecting ? "" : "bg-green-600 hover:bg-green-700"}
                    >
                        {isDetecting ? "⏹ Stop" : "▶ Start Detection"}
                    </Button>
                    {isDetecting && (
                        <Button
                            variant="outline"
                            onClick={() => setShowDensitySection(!showDensitySection)}
                            className={showDensitySection
                                ? "bg-purple-600/40 border-purple-500 text-purple-300 hover:bg-purple-600/50"
                                : "bg-purple-600/20 border-purple-500 text-purple-400 hover:bg-purple-600/30"}
                        >
                            📊 Traffic Density
                        </Button>
                    )}
                    <Button variant="outline" onClick={() => setShowDashboard(!showDashboard)}>
                        {showDashboard ? "Hide Stats" : "Stats"}
                    </Button>
                </div>

                {/* Class Filter Toggle Buttons */}
                <div className="flex items-center gap-2 flex-wrap mt-2">
                    <span className="text-xs text-muted-foreground">Filter:</span>
                    {Object.entries(classFilter).map(([cls, enabled]) => {
                        const icons: Record<string, string> = {
                            person: "👤",
                            car: "🚗",
                            motorcycle: "🏍️",
                            bus: "🚌",
                            truck: "🚚",
                            bicycle: "🚲",
                        };
                        const labels: Record<string, string> = {
                            person: "Person",
                            car: "Car",
                            motorcycle: "Motorbike",
                            bus: "Bus",
                            truck: "Truck",
                            bicycle: "Bicycle",
                        };
                        return (
                            <button
                                key={cls}
                                onClick={() => handleClassFilterToggle(cls)}
                                className={`px-3 py-1.5 text-xs rounded-full border transition-all ${enabled
                                    ? "bg-green-600/20 border-green-500 text-green-400"
                                    : "bg-gray-800/50 border-gray-600 text-gray-500 opacity-60"
                                    }`}
                                title={enabled ? `Detecting ${labels[cls]}` : `Not detecting ${labels[cls]}`}
                            >
                                {icons[cls]} {labels[cls]}
                            </button>
                        );
                    })}

                    <span className="text-gray-600 mx-2">|</span>
                    <span className="text-xs text-muted-foreground">Features:</span>
                    {Object.entries(featureFilter).map(([feature, enabled]) => {
                        const icons: Record<string, string> = {
                            traffic_light: "🚦",
                            parking_zone: "🅿️",
                            counting_line: "🔢",
                            stop_line: "🚧",
                        };
                        const labels: Record<string, string> = {
                            traffic_light: "Traffic Light",
                            parking_zone: "Parking Zone",
                            counting_line: "Counting Line",
                            stop_line: "Stop Line",
                        };
                        return (
                            <button
                                key={feature}
                                onClick={() => {
                                    const newFilter = { ...featureFilter, [feature]: !featureFilter[feature] };
                                    setFeatureFilter(newFilter);
                                    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                                        wsRef.current.send(JSON.stringify({
                                            type: "update_feature_filter",
                                            feature_filter: newFilter
                                        }));
                                    }
                                }}
                                className={`px-3 py-1.5 text-xs rounded-full border transition-all ${enabled
                                    ? "bg-blue-600/20 border-blue-500 text-blue-400"
                                    : "bg-gray-800/50 border-gray-600 text-gray-500 opacity-60"
                                    }`}
                                title={enabled ? `Show ${labels[feature]}` : `Hide ${labels[feature]}`}
                            >
                                {icons[feature]} {labels[feature]}
                            </button>
                        );
                    })}
                </div>
            </header>

            <div className="flex p-4 gap-4">
                <div className="flex-1">
                    <div
                        ref={containerRef}
                        className="relative bg-black rounded-lg overflow-hidden"
                        style={{ aspectRatio: "16/9" }}
                    >
                        {isVideo ? (
                            isDetecting && syncedFrame ? (
                                <img
                                    src={syncedFrame}
                                    alt="Detection Stream"
                                    className="absolute inset-0 w-full h-full object-contain"
                                />
                            ) : (
                                <video
                                    ref={videoRef}
                                    className="absolute inset-0 w-full h-full object-contain"
                                    autoPlay
                                    muted
                                    playsInline
                                    loop
                                    crossOrigin="anonymous"
                                    onLoadedMetadata={handleVideoLoad}
                                    src={isM3u8 ? undefined : cameraUrl}
                                />
                            )
                        ) : (
                            <img
                                src={getProxiedImageUrl()}
                                alt={cameraName}
                                className="absolute inset-0 w-full h-full object-contain"
                                onLoad={handleImageLoad}
                                crossOrigin="anonymous"
                                onError={(e) => {
                                    (e.target as HTMLImageElement).src =
                                        "data:image/svg+xml;utf8," +
                                        encodeURIComponent(
                                            `<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'>
                                            <rect width='100%' height='100%' fill='#111827'/>
                                            <text x='50%' y='50%' fill='#e5e7eb' font-size='28' font-family='Arial'
                                            dominant-baseline='middle' text-anchor='middle'>Camera Offline</text>
                                        </svg>`
                                        );
                                }}
                            />
                        )}

                        <canvas ref={canvasRef} style={{ display: "none" }} />

                        {isDetecting && result && !isEditing && !syncedFrame && (
                            <DetectionViewer
                                detections={result.detections}
                                zones={zones}
                                violations={violations}
                                frameWidth={frameSize.width}
                                frameHeight={frameSize.height}
                                containerWidth={containerSize.width}
                                containerHeight={containerSize.height}
                            />
                        )}

                        <PolygonEditor
                            zones={zones}
                            frameWidth={frameSize.width}
                            frameHeight={frameSize.height}
                            containerWidth={containerSize.width}
                            containerHeight={containerSize.height}
                            onZoneAdd={handleZoneAdd}
                            onZoneDelete={handleZoneDelete}
                            onZonesClear={handleZonesClear}
                            isEditing={isEditing}
                            onEditingChange={setIsEditing}
                        />

                        {isDetecting && (
                            <div className="absolute top-2 left-2 bg-black/80 text-white text-xs px-3 py-2 rounded flex items-center gap-2 z-50">
                                <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                                {debugInfo || "Starting..."}
                            </div>
                        )}
                    </div>
                </div>

                {showDashboard && (
                    <div className="w-80 shrink-0">
                        <TrafficDashboard
                            result={result}
                            violations={violations}
                            zones={zones}
                            isConnected={isDetecting}
                            lineCounts={lineCounts}
                            cameraId={cameraId}
                            showDensitySection={showDensitySection}
                        />
                    </div>
                )}
            </div>


        </div>
    );
}
