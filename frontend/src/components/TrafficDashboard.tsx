import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import type { DetectionResult, ParkingViolation, ZonePolygon, LineCounts, TrafficDensityResult, TrafficDensityConfig } from "@/lib/api";
import { startDensityTracking, stopDensityTracking, getDensityStatus } from "@/lib/api";

interface TrafficDashboardProps {
    result: DetectionResult | null;
    violations: ParkingViolation[];
    zones: ZonePolygon[];
    isConnected: boolean;
    lineCounts?: Record<string, LineCounts>;
    cameraId: string;
    showDensitySection?: boolean;
}

const VEHICLE_ICONS: Record<string, string> = {
    car: "🚗",
    motorcycle: "🏍️",
    bus: "🚌",
    truck: "🚛",
    bicycle: "🚲",
    person: "🚶",
};

const VEHICLE_COLORS: Record<string, string> = {
    car: "bg-green-500",
    motorcycle: "bg-yellow-500",
    bus: "bg-orange-500",
    truck: "bg-purple-500",
    bicycle: "bg-cyan-500",
    person: "bg-blue-500",
};

export default function TrafficDashboard({
    result,
    violations,
    zones,
    isConnected,
    lineCounts = {},
    cameraId,
    showDensitySection = false,
}: TrafficDashboardProps) {
    const [history, setHistory] = useState<{ time: string; count: number }[]>([]);

    // Traffic Density State
    const [durationMinutes, setDurationMinutes] = useState(15);
    const [isTracking, setIsTracking] = useState(false);
    const [currentCount, setCurrentCount] = useState(0);
    const [elapsedMinutes, setElapsedMinutes] = useState(0);
    const [currentHour, setCurrentHour] = useState(new Date().getHours());
    const [densityResult, setDensityResult] = useState<TrafficDensityResult | null>(null);
    const [densityError, setDensityError] = useState<string | null>(null);
    const [densityLoading, setDensityLoading] = useState(false);

    useEffect(() => {
        if (result && result.total_count > 0) {
            const now = new Date().toLocaleTimeString("vi-VN", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
            });
            setHistory((prev) => {
                const updated = [...prev, { time: now, count: result.total_count }];
                return updated.slice(-20);
            });
        }
    }, [result]);

    // Polling for density status when tracking
    useEffect(() => {
        let interval: number | null = null;

        if (isTracking && cameraId) {
            interval = window.setInterval(async () => {
                try {
                    const status = await getDensityStatus(cameraId);
                    setCurrentCount(status.current_count);
                    setElapsedMinutes(status.elapsed_minutes);

                    if (status.elapsed_minutes >= durationMinutes) {
                        await handleDensityStop();
                    }
                } catch (e) {
                    console.error("Error polling density status:", e);
                }
            }, 2000);
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [isTracking, cameraId, durationMinutes]);

    const handleDensityStart = async () => {
        if (!cameraId) return;
        setDensityError(null);
        setDensityResult(null);
        setDensityLoading(true);

        try {
            const config: TrafficDensityConfig = {
                duration_minutes: durationMinutes,
            };
            await startDensityTracking(cameraId, config);
            setIsTracking(true);
            setCurrentCount(0);
            setElapsedMinutes(0);
            setCurrentHour(new Date().getHours());
        } catch (e: any) {
            setDensityError(e.message || "Failed to start tracking");
        } finally {
            setDensityLoading(false);
        }
    };

    const handleDensityStop = async () => {
        if (!cameraId) return;
        setDensityLoading(true);
        try {
            const result = await stopDensityTracking(cameraId);
            setDensityResult(result);
            setIsTracking(false);
        } catch (e: any) {
            setDensityError(e.message || "Failed to stop tracking");
        } finally {
            setDensityLoading(false);
        }
    };

    const maxCount = Math.max(...history.map((h) => h.count), 1);

    const DENSITY_COLORS = {
        heavy: "bg-red-500",
        medium: "bg-yellow-500",
        light: "bg-green-500",
    };

    return (
        <div className="bg-card border border-border rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between">
                <h3 className="font-bold text-lg text-foreground">Traffic Analytics</h3>
                <div className="flex items-center gap-2">
                    <div
                        className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
                            }`}
                    />
                    <span className="text-xs text-muted-foreground">
                        {isConnected ? "Live" : "Offline"}
                    </span>
                </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                <div className="bg-primary/10 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-primary">
                        {result?.total_count ?? 0}
                    </div>
                    <div className="text-xs text-muted-foreground">Total Vehicles</div>
                </div>

                <div className="bg-red-500/10 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-red-500">
                        {violations.length}
                    </div>
                    <div className="text-xs text-muted-foreground">Violations</div>
                </div>

                <div className="bg-blue-500/10 rounded-lg p-3 text-center">
                    <div className="text-2xl font-bold text-blue-500">{zones.length}</div>
                    <div className="text-xs text-muted-foreground">Active Zones</div>
                </div>
            </div>

            {result && Object.keys(result.vehicle_count).length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-foreground">By Type</h4>
                    <div className="grid grid-cols-2 gap-2">
                        {Object.entries(result.vehicle_count).map(([type, count]) => (
                            <div
                                key={type}
                                className="flex items-center gap-2 bg-muted/50 rounded p-2"
                            >
                                <span className="text-lg">{VEHICLE_ICONS[type] ?? "🚙"}</span>
                                <div className="flex-1">
                                    <div className="text-sm font-medium capitalize">{type}</div>
                                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                                        <div
                                            className={`h-full ${VEHICLE_COLORS[type] ?? "bg-gray-500"}`}
                                            style={{
                                                width: `${(count / result.total_count) * 100}%`,
                                            }}
                                        />
                                    </div>
                                </div>
                                <span className="text-lg font-bold">{count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {history.length > 1 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-foreground">
                        Vehicle Count Trend
                    </h4>
                    <div className="h-20 flex items-end gap-1">
                        {history.map((h, i) => (
                            <div
                                key={i}
                                className="flex-1 bg-primary/70 rounded-t transition-all duration-300"
                                style={{ height: `${(h.count / maxCount) * 100}%` }}
                                title={`${h.time}: ${h.count}`}
                            />
                        ))}
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{history[0]?.time}</span>
                        <span>{history[history.length - 1]?.time}</span>
                    </div>
                </div>
            )}

            {violations.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-semibold text-red-500">
                        ⚠ Parking Violations
                    </h4>
                    <div className="max-h-32 overflow-y-auto space-y-1">
                        {violations.map((v, i) => (
                            <div
                                key={i}
                                className="flex items-center gap-2 bg-red-500/10 rounded p-2 text-xs"
                            >
                                <span className="text-red-500 font-bold">#{v.track_id}</span>
                                <span className="capitalize">{v.vehicle_class}</span>
                                <span className="text-muted-foreground">in</span>
                                <span className="font-medium">{v.zone_name}</span>
                                <span className="ml-auto text-red-500 font-bold">
                                    {v.duration_seconds.toFixed(0)}s
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Counting Lines Stats */}
            {Object.keys(lineCounts).length > 0 && (
                <div>
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm font-semibold text-foreground">🔢 Vehicle Counting</span>
                    </div>
                    <div className="space-y-2">
                        {Object.values(lineCounts).map((count) => (
                            <div
                                key={count.line_id}
                                className="bg-cyan-500/10 rounded p-3 border border-cyan-500/30"
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className="text-xs font-semibold text-foreground">
                                        {count.line_name}
                                    </span>
                                    <span className="text-lg font-bold text-cyan-400">
                                        {count.total}
                                    </span>
                                </div>
                                <div className="flex gap-4 text-xs text-muted-foreground">
                                    <span>↓ In: <span className="text-green-400 font-semibold">{count.count_in}</span></span>
                                    <span>↑ Out: <span className="text-red-400 font-semibold">{count.count_out}</span></span>
                                </div>
                                {Object.keys(count.by_class).length > 0 && (
                                    <div className="mt-2 flex flex-wrap gap-1">
                                        {Object.entries(count.by_class).map(([cls, counts]) => (
                                            <span
                                                key={cls}
                                                className="text-xs bg-background/50 px-2 py-0.5 rounded"
                                            >
                                                {VEHICLE_ICONS[cls] || "🚗"} {counts.in + counts.out}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {result && (
                <div className="text-xs text-muted-foreground text-right">
                    Processing: {result.processing_time_ms.toFixed(0)}ms
                </div>
            )}

            {/* Traffic Density Section */}
            {showDensitySection && (
                <div className="border-t border-border pt-4 space-y-3">
                    <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold text-foreground">📊 Traffic Density</span>
                        {isTracking && (
                            <span className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                        )}
                    </div>

                    {/* Config Form - hiển thị khi chưa tracking và chưa có result */}
                    {!isTracking && !densityResult && (
                        <div className="space-y-3">
                            <div className="bg-secondary/50 rounded-lg p-2 text-xs">
                                <p className="text-muted-foreground">
                                    So sánh với trung bình khung giờ {new Date().getHours()}h - {new Date().getHours() + 1}h
                                </p>
                            </div>
                            <div className="flex items-center gap-2">
                                <label className="text-xs text-muted-foreground">Duration:</label>
                                <input
                                    type="number"
                                    value={durationMinutes}
                                    onChange={(e) => setDurationMinutes(Number(e.target.value))}
                                    className="w-16 px-2 py-1 bg-secondary border border-border rounded text-sm"
                                    min={1}
                                    max={60}
                                />
                                <span className="text-xs text-muted-foreground">phút</span>
                            </div>
                            <Button
                                className="w-full bg-purple-600 hover:bg-purple-700"
                                size="sm"
                                onClick={handleDensityStart}
                                disabled={densityLoading}
                            >
                                {densityLoading ? "Starting..." : "▶ Start Tracking"}
                            </Button>
                        </div>
                    )}

                    {/* Tracking Progress */}
                    {isTracking && (
                        <div className="space-y-3">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-primary">{currentCount}</div>
                                <div className="text-xs text-muted-foreground">vehicles counted</div>
                            </div>

                            <div className="bg-secondary rounded-lg p-2">
                                <div className="flex justify-between text-xs mb-1">
                                    <span>Progress</span>
                                    <span>{elapsedMinutes.toFixed(1)} / {durationMinutes} min</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-1.5">
                                    <div
                                        className="bg-purple-500 h-1.5 rounded-full transition-all"
                                        style={{ width: `${Math.min((elapsedMinutes / durationMinutes) * 100, 100)}%` }}
                                    />
                                </div>
                            </div>

                            <div className="text-xs text-muted-foreground text-center">
                                Hour: {currentHour}h - {currentHour + 1}h
                            </div>

                            <Button
                                className="w-full"
                                variant="destructive"
                                size="sm"
                                onClick={handleDensityStop}
                                disabled={densityLoading}
                            >
                                {densityLoading ? "Stopping..." : "⏹ Stop & Get Result"}
                            </Button>
                        </div>
                    )}

                    {/* Density Result */}
                    {densityResult && (
                        <div className="space-y-3">
                            <div className="text-center">
                                <div className={`inline-block px-4 py-2 rounded-lg text-white font-bold text-lg ${DENSITY_COLORS[densityResult.density_level]}`}>
                                    {densityResult.density_label}
                                </div>
                                <div className="mt-1 text-2xl font-bold">
                                    {densityResult.flow_rate.toFixed(1)} <span className="text-sm font-normal">xe/giờ</span>
                                </div>
                                <div className="text-xs text-muted-foreground">
                                    {densityResult.density_percentage.toFixed(1)}% so với trung bình
                                </div>
                            </div>

                            <div className="bg-secondary rounded-lg p-3 space-y-1 text-xs">
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Total Vehicles:</span>
                                    <span className="font-bold">{densityResult.total_vehicles}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Hourly Average:</span>
                                    <span>{densityResult.hourly_average.toFixed(1)} xe/giờ</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-muted-foreground">Duration:</span>
                                    <span>{densityResult.duration_minutes} min</span>
                                </div>
                            </div>

                            {Object.keys(densityResult.vehicle_breakdown).length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                    {Object.entries(densityResult.vehicle_breakdown).map(([type, count]) => (
                                        <span key={type} className="text-xs bg-background/50 px-2 py-0.5 rounded capitalize">
                                            {VEHICLE_ICONS[type] || "🚗"} {count}
                                        </span>
                                    ))}
                                </div>
                            )}

                            <div className="flex gap-2">
                                <Button
                                    className="flex-1"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => {
                                        const dataStr = JSON.stringify(densityResult, null, 2);
                                        const blob = new Blob([dataStr], { type: "application/json" });
                                        const url = URL.createObjectURL(blob);
                                        const a = document.createElement("a");
                                        a.href = url;
                                        a.download = `density_${cameraId}_${new Date().toISOString().slice(0, 10)}.json`;
                                        a.click();
                                        URL.revokeObjectURL(url);
                                    }}
                                >
                                    📄 JSON
                                </Button>
                                <Button
                                    className="flex-1"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setDensityResult(null)}
                                >
                                    🔄 Again
                                </Button>
                            </div>
                        </div>
                    )}

                    {densityError && (
                        <div className="p-2 bg-red-500/20 border border-red-500 rounded text-red-400 text-xs">
                            {densityError}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
