import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import type { TrafficDensityResult, TrafficDensityConfig, TrafficDensityStatus } from "@/lib/api";
import {
    startDensityTracking,
    stopDensityTracking,
    getDensityStatus,
} from "@/lib/api";

interface TrafficDensityPanelProps {
    cameraId: string;
    isOpen: boolean;
    onClose: () => void;
}

const DENSITY_COLORS = {
    heavy: "bg-red-500",
    medium: "bg-yellow-500",
    light: "bg-green-500",
};

const DENSITY_LABELS = {
    heavy: "Đông",
    medium: "Trung bình",
    light: "Ít",
};

export default function TrafficDensityPanel({
    cameraId,
    isOpen,
    onClose,
}: TrafficDensityPanelProps) {
    const [durationMinutes, setDurationMinutes] = useState(15);
    const [isTracking, setIsTracking] = useState(false);
    const [currentCount, setCurrentCount] = useState(0);
    const [elapsedMinutes, setElapsedMinutes] = useState(0);
    const [currentHour, setCurrentHour] = useState(new Date().getHours());
    const [result, setResult] = useState<TrafficDensityResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        let interval: number | null = null;

        if (isTracking) {
            interval = window.setInterval(async () => {
                try {
                    const status = await getDensityStatus(cameraId);
                    setCurrentCount(status.current_count);
                    setElapsedMinutes(status.elapsed_minutes);

                    if (status.elapsed_minutes >= durationMinutes) {
                        await handleStop();
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

    const handleStart = async () => {
        setError(null);
        setResult(null);
        setLoading(true);

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
            setError(e.message || "Failed to start tracking");
        } finally {
            setLoading(false);
        }
    };

    const handleStop = async () => {
        setLoading(true);
        try {
            const densityResult = await stopDensityTracking(cameraId);
            setResult(densityResult);
            setIsTracking(false);
        } catch (e: any) {
            setError(e.message || "Failed to stop tracking");
        } finally {
            setLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-card border border-border rounded-lg p-6 w-full max-w-md shadow-xl">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold">📊 Traffic Density</h2>
                    <Button variant="ghost" size="sm" onClick={onClose}>
                        ✕
                    </Button>
                </div>

                {!isTracking && !result && (
                    <div className="space-y-4">
                        <div className="bg-secondary/50 rounded-lg p-3 text-sm">
                            <p className="text-muted-foreground">
                                So sánh lượng xe với trung bình của khung giờ hiện tại ({new Date().getHours()}h - {new Date().getHours() + 1}h)
                            </p>
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">
                                Duration (phút)
                            </label>
                            <input
                                type="number"
                                value={durationMinutes}
                                onChange={(e) => setDurationMinutes(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-secondary border border-border rounded-md"
                                min={1}
                                max={60}
                            />
                        </div>
                        <Button
                            className="w-full bg-green-600 hover:bg-green-700"
                            onClick={handleStart}
                            disabled={loading}
                        >
                            {loading ? "Starting..." : "▶ Start Tracking"}
                        </Button>
                    </div>
                )}

                {isTracking && (
                    <div className="space-y-4">
                        <div className="text-center">
                            <div className="text-4xl font-bold text-primary mb-2">
                                {currentCount}
                            </div>
                            <div className="text-sm text-muted-foreground">
                                vehicles counted
                            </div>
                        </div>

                        <div className="bg-secondary rounded-lg p-3">
                            <div className="flex justify-between text-sm mb-2">
                                <span>Progress</span>
                                <span>
                                    {elapsedMinutes.toFixed(1)} / {durationMinutes} min
                                </span>
                            </div>
                            <div className="w-full bg-gray-700 rounded-full h-2">
                                <div
                                    className="bg-primary h-2 rounded-full transition-all"
                                    style={{
                                        width: `${Math.min((elapsedMinutes / durationMinutes) * 100, 100)}%`,
                                    }}
                                />
                            </div>
                        </div>

                        <div className="text-sm text-muted-foreground text-center">
                            Tracking for hour: {currentHour}h - {currentHour + 1}h
                        </div>

                        <Button
                            className="w-full"
                            variant="destructive"
                            onClick={handleStop}
                            disabled={loading}
                        >
                            {loading ? "Stopping..." : "⏹ Stop & Get Result"}
                        </Button>
                    </div>
                )}

                {result && (
                    <div className="space-y-4">
                        <div className="text-center">
                            <div
                                className={`inline-block px-6 py-3 rounded-lg text-white font-bold text-2xl ${DENSITY_COLORS[result.density_level]}`}
                            >
                                {result.density_label}
                            </div>
                            <div className="mt-2 text-3xl font-bold">
                                {result.flow_rate.toFixed(1)} <span className="text-lg font-normal">xe/giờ</span>
                            </div>
                            <div className="text-sm text-muted-foreground mt-1">
                                {result.density_percentage.toFixed(1)}% so với trung bình
                            </div>
                        </div>

                        <div className="bg-secondary rounded-lg p-4 space-y-2">
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Total Vehicles:</span>
                                <span className="font-bold">{result.total_vehicles}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Flow Rate (q=n/t):</span>
                                <span className="font-bold">{result.flow_rate.toFixed(1)} xe/giờ</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Hourly Average:</span>
                                <span>{result.hourly_average.toFixed(1)} xe/giờ</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Hour:</span>
                                <span>{result.hour}h - {result.hour + 1}h</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted-foreground">Duration:</span>
                                <span>{result.duration_minutes} min</span>
                            </div>
                        </div>

                        {Object.keys(result.vehicle_breakdown).length > 0 && (
                            <div className="bg-secondary rounded-lg p-4">
                                <h3 className="font-medium mb-2">Vehicle Breakdown</h3>
                                <div className="space-y-1">
                                    {Object.entries(result.vehicle_breakdown).map(([type, count]) => (
                                        <div key={type} className="flex justify-between text-sm">
                                            <span className="capitalize">{type}</span>
                                            <span className="font-medium">{count}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Export Buttons */}
                        <div className="flex gap-2">
                            <Button
                                className="flex-1"
                                variant="outline"
                                onClick={() => {
                                    const dataStr = JSON.stringify(result, null, 2);
                                    const blob = new Blob([dataStr], { type: "application/json" });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;
                                    a.download = `density_${cameraId}_${new Date().toISOString().slice(0, 10)}.json`;
                                    a.click();
                                    URL.revokeObjectURL(url);
                                }}
                            >
                                📄 Export JSON
                            </Button>
                            <Button
                                className="flex-1"
                                variant="outline"
                                onClick={() => {
                                    const rows = [
                                        ["Camera ID", result.camera_id],
                                        ["Start Time", result.start_time],
                                        ["End Time", result.end_time],
                                        ["Hour", `${result.hour}h - ${result.hour + 1}h`],
                                        ["Duration (min)", result.duration_minutes.toString()],
                                        ["Total Vehicles", result.total_vehicles.toString()],
                                        ["Flow Rate (q=n/t)", `${result.flow_rate.toFixed(2)} xe/giờ`],
                                        ["Hourly Average", `${result.hourly_average.toFixed(2)} xe/giờ`],
                                        ["Density %", result.density_percentage.toFixed(2)],
                                        ["Density Level", result.density_level],
                                        ["Density Label", result.density_label],
                                        [""],
                                        ["Vehicle Type", "Count"],
                                        ...Object.entries(result.vehicle_breakdown).map(([type, count]) => [type, count.toString()])
                                    ];
                                    const csv = rows.map(r => r.join(",")).join("\n");
                                    const blob = new Blob([csv], { type: "text/csv" });
                                    const url = URL.createObjectURL(blob);
                                    const a = document.createElement("a");
                                    a.href = url;
                                    a.download = `density_${cameraId}_${new Date().toISOString().slice(0, 10)}.csv`;
                                    a.click();
                                    URL.revokeObjectURL(url);
                                }}
                            >
                                📊 Export CSV
                            </Button>
                        </div>

                        <Button
                            className="w-full"
                            variant="outline"
                            onClick={() => setResult(null)}
                        >
                            🔄 Track Again
                        </Button>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-3 bg-red-500/20 border border-red-500 rounded-lg text-red-400 text-sm">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
}
