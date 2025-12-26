import { useNavigate } from "react-router-dom";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Camera, Car, TrafficCone, Ban, ScanLine, MapPin, Activity } from "lucide-react";
import type { Camera as CameraType } from "@/lib/api";

interface CameraActionDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    camera: CameraType;
}

export default function CameraActionDialog({
    open,
    onOpenChange,
    camera,
}: CameraActionDialogProps) {
    const navigate = useNavigate();

    const handleAction = (action: string) => {
        if (action === "counting") {
            // In a real app, we might pass the camera ID to the detection page
            // e.g., navigate(`/detection?camera=${camera.id}`)
            // For now, just navigate to the page as requested
            navigate("/detection", { state: { cameraUrl: camera.url } });
            onOpenChange(false);
        }
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[800px] bg-background border-border p-0 overflow-hidden gap-0">

                {/* Header Section */}
                <div className="bg-muted/50 p-6 border-b border-border">
                    <DialogHeader>
                        <DialogTitle className="flex items-start justify-between">
                            <div className="space-y-1.5">
                                <div className="flex items-center gap-2 text-xl">
                                    <Camera className="w-5 h-5 text-primary" />
                                    {camera.name}
                                </div>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <MapPin className="w-4 h-4" />
                                    {camera.location || "Unknown Location"}
                                </div>
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 text-green-500 text-xs font-medium border border-green-500/20">
                                <Activity className="w-3 h-3" />
                                Signal Active
                            </div>
                        </DialogTitle>
                    </DialogHeader>
                </div>

                {/* Action Grid */}
                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-4">

                    {/* Active Module: Vehicle Counting */}
                    <button
                        onClick={() => handleAction("counting")}
                        className="flex flex-col items-start p-4 rounded-xl border-2 border-primary/20 bg-primary/5 hover:bg-primary/10 hover:border-primary transition-all group text-left"
                    >
                        <div className="p-3 rounded-lg bg-primary/10 text-primary mb-3 group-hover:scale-110 transition-transform">
                            <Car className="w-6 h-6" />
                        </div>
                        <h3 className="font-semibold text-foreground mb-1">Vehicle Counting</h3>
                        <p className="text-sm text-muted-foreground">
                            Real-time classification and counting of cars, trucks, and motorcycles.
                        </p>
                    </button>

                    {/* Planned Module: Traffic Light */}
                    <button
                        disabled
                        className="flex flex-col items-start p-4 rounded-xl border border-border bg-card/50 opacity-60 cursor-not-allowed hover:bg-muted/50 text-left"
                    >
                        <div className="p-3 rounded-lg bg-muted text-muted-foreground mb-3">
                            <TrafficCone className="w-6 h-6" />
                        </div>
                        <h3 className="font-semibold text-foreground mb-1">Traffic Violation</h3>
                        <p className="text-sm text-muted-foreground">
                            Detect red light runners and lane violations. (Coming Soon)
                        </p>
                    </button>

                    {/* Planned Module: Illegal Parking */}
                    <button
                        disabled
                        className="flex flex-col items-start p-4 rounded-xl border border-border bg-card/50 opacity-60 cursor-not-allowed hover:bg-muted/50 text-left"
                    >
                        <div className="p-3 rounded-lg bg-muted text-muted-foreground mb-3">
                            <Ban className="w-6 h-6" />
                        </div>
                        <h3 className="font-semibold text-foreground mb-1">Illegal Parking</h3>
                        <p className="text-sm text-muted-foreground">
                            Monitor no-stopping zones and bus stops. (Coming Soon)
                        </p>
                    </button>

                    {/* Planned Module: LPR */}
                    <button
                        disabled
                        className="flex flex-col items-start p-4 rounded-xl border border-border bg-card/50 opacity-60 cursor-not-allowed hover:bg-muted/50 text-left"
                    >
                        <div className="p-3 rounded-lg bg-muted text-muted-foreground mb-3">
                            <ScanLine className="w-6 h-6" />
                        </div>
                        <h3 className="font-semibold text-foreground mb-1">License Plate (LPR)</h3>
                        <p className="text-sm text-muted-foreground">
                            Automatic number plate recognition for security. (Coming Soon)
                        </p>
                    </button>

                </div>

                <div className="p-6 pt-0 flex justify-end">
                    <Button variant="outline" onClick={() => onOpenChange(false)}>Close Control Panel</Button>
                </div>

            </DialogContent>
        </Dialog>
    );
}
