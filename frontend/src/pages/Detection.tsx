import { useState } from "react";
import { useLocation } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { ShieldCheck, ShieldAlert, Car, Bus, Truck, Bike, Play, Square, MapPin } from "lucide-react";
import { ZoneEditor } from "../components/ZoneEditor";

const API_URL = "http://localhost:8000/api";

export default function Detection() {
    const queryClient = useQueryClient();
    const location = useLocation();
    const cameraUrl = location.state?.cameraUrl; // Get URL from state

    const [isZoneEditorOpen, setIsZoneEditorOpen] = useState(false);
    const [parkingZone, setParkingZone] = useState<number[][] | null>(null);

    const { data: status } = useQuery({
        queryKey: ['detectionStatus'],
        queryFn: async () => {
            const res = await fetch(`${API_URL}/detection/status`);
            return res.json();
        },
        refetchInterval: 1000, // Faster polling for smoother updates
    });

    const startMutation = useMutation({
        mutationFn: async () => { // eslint-disable-line
            const body = {
                source: cameraUrl || 'video.mp4',
                parking_zone: parkingZone // Pass drawn zone
            };

            const res = await fetch(`${API_URL}/detection/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            return res.json();
        },
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['detectionStatus'] })
    });

    const stopMutation = useMutation({
        mutationFn: async () => {
            const res = await fetch(`${API_URL}/detection/stop`, { method: 'POST' });
            return res.json();
        },
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['detectionStatus'] })
    });

    // Real data from backend
    const counts = status?.counts || { car: 0, motorcycle: 0, bus: 0, truck: 0 };

    const chartData = [
        { name: 'Car', count: counts.car || 0 },
        { name: 'Moto', count: counts.motorcycle || 0 },
        { name: 'Bus', count: counts.bus || 0 },
        { name: 'Truck', count: counts.truck || 0 },
    ];

    // Snapshot URL for Zone Editor
    // Use the backend snapshot endpoint to ensure we get a valid image even for m3u8 streams
    const snapshotUrl = `${API_URL}/detection/snapshot?source=${encodeURIComponent(cameraUrl || "https://giaothong.hochiminhcity.gov.vn/render/ImageHandler.ashx?id=58df7c46bd22020011882d8c")}`;


    return (
        <div className="min-h-screen bg-background p-8">
            <div className="max-w-7xl mx-auto space-y-8">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-4xl font-bold tracking-tight text-foreground">Vehicle Detection Survey</h1>
                        <p className="text-muted-foreground mt-2">Real-time monitoring and classification system</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className={`px-4 py-2 rounded-full flex items-center gap-2 ${status?.running ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                            {status?.running ? <ShieldCheck className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
                            <span className="font-medium">{status?.running ? 'System Active' : 'System Inactive'}</span>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="col-span-2 bg-card rounded-xl border p-6 shadow-sm">
                        <h3 className="text-lg font-semibold mb-4">Live Feed</h3>
                        <div className="aspect-video bg-black/5 rounded-lg flex items-center justify-center border-2 border-dashed border-border relative overflow-hidden group">
                            {status?.running ? (
                                <img
                                    src={`${API_URL}/detection/video_feed`}
                                    alt="Live Detection Feed"
                                    className="w-full h-full object-contain"
                                />
                            ) : (
                                <div className="text-muted-foreground flex flex-col items-center gap-2">
                                    <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                                        <Play className="w-6 h-6 text-primary" />
                                    </div>
                                    <span>Click Start to Begin Monitoring</span>
                                </div>
                            )}
                        </div>
                        <div className="flex gap-4 mt-6">
                            <button
                                onClick={() => setIsZoneEditorOpen(true)}
                                className="bg-yellow-500 text-white hover:bg-yellow-600 h-10 px-4 py-2 rounded-md inline-flex items-center justify-center whitespace-nowrap text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                            >
                                <MapPin className="mr-2 h-4 w-4" />
                                {parkingZone ? "Edit Parking Zone" : "Set Parking Zone"}
                            </button>

                            <button
                                onClick={() => startMutation.mutate()}
                                disabled={status?.running}
                                className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2 rounded-md inline-flex items-center justify-center whitespace-nowrap text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                            >
                                <Play className="mr-2 h-4 w-4" /> Start Detection
                            </button>
                            <button
                                onClick={() => stopMutation.mutate()}
                                disabled={!status?.running}
                                className="flex-1 bg-destructive text-destructive-foreground hover:bg-destructive/90 h-10 px-4 py-2 rounded-md inline-flex items-center justify-center whitespace-nowrap text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                            >
                                <Square className="mr-2 h-4 w-4" /> Stop Detection
                            </button>
                        </div>
                    </div>

                    <div className="space-y-6">
                        <div className="bg-card rounded-xl border p-6 shadow-sm">
                            <h3 className="text-lg font-semibold mb-4">Statistics</h3>
                            <div className="h-[200px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={chartData}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                                        <XAxis dataKey="name" fontSize={12} tickLine={false} axisLine={false} />
                                        <YAxis fontSize={12} tickLine={false} axisLine={false} />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: 'hsl(var(--card))', borderRadius: '8px', border: '1px solid hsl(var(--border))' }}
                                            itemStyle={{ color: 'hsl(var(--foreground))' }}
                                        />
                                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-card p-4 rounded-xl border flex flex-col items-center justify-center text-center">
                                <Car className="w-8 h-8 text-blue-500 mb-2" />
                                <span className="text-2xl font-bold">{chartData[0].count}</span>
                                <span className="text-xs text-muted-foreground">Cars</span>
                            </div>
                            <div className="bg-card p-4 rounded-xl border flex flex-col items-center justify-center text-center">
                                <Bike className="w-8 h-8 text-orange-500 mb-2" />
                                <span className="text-2xl font-bold">{chartData[1].count}</span>
                                <span className="text-xs text-muted-foreground">Motorcycles</span>
                            </div>
                            <div className="bg-card p-4 rounded-xl border flex flex-col items-center justify-center text-center">
                                <Bus className="w-8 h-8 text-green-500 mb-2" />
                                <span className="text-2xl font-bold">{chartData[2].count}</span>
                                <span className="text-xs text-muted-foreground">Buses</span>
                            </div>
                            <div className="bg-card p-4 rounded-xl border flex flex-col items-center justify-center text-center">
                                <Truck className="w-8 h-8 text-purple-500 mb-2" />
                                <span className="text-2xl font-bold">{chartData[3].count}</span>
                                <span className="text-xs text-muted-foreground">Trucks</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <ZoneEditor
                open={isZoneEditorOpen}
                onOpenChange={setIsZoneEditorOpen}
                imageUrl={snapshotUrl}
                onSave={setParkingZone}
            />
        </div>
    );
}
