import { useState, useRef, useEffect, MouseEvent } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface ZoneEditorProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    imageUrl: string;
    onSave: (points: number[][]) => void;
}

export function ZoneEditor({ open, onOpenChange, imageUrl, onSave }: ZoneEditorProps) {
    const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
    const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imageRef = useRef<HTMLImageElement>(null);

    // Reset points when dialog opens/closes or image changes
    useEffect(() => {
        if (!open) {
            setPoints([]);
        }
    }, [open]);

    const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
        const img = e.currentTarget;
        setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    };

    const handleCanvasClick = (e: MouseEvent<HTMLCanvasElement>) => {
        if (!canvasRef.current || !imageRef.current) return;

        const rect = canvasRef.current.getBoundingClientRect();
        const scaleX = canvasRef.current.width / rect.width;
        const scaleY = canvasRef.current.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        setPoints([...points, { x, y }]);
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (!canvas || !ctx || !imageSize) return;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Polygon
        if (points.length > 0) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(points[i].x, points[i].y);
            }
            if (points.length > 2) {
                ctx.closePath();
            }

            ctx.strokeStyle = 'red';
            ctx.lineWidth = 3;
            ctx.stroke();

            ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
            if (points.length > 2) ctx.fill();

            // Draw vertices
            ctx.fillStyle = 'yellow';
            points.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
                ctx.fill();
            });
        }

    }, [points, imageSize]);

    const handleSave = () => {
        if (!imageRef.current || !canvasRef.current) return;

        // Convert scale if needed (Canvas resolution match native image?)
        // The canvas width/height are set to native image size below.
        // So 'points' are already in native coordinates.

        const zonePoints = points.map(p => [Math.round(p.x), Math.round(p.y)]);
        onSave(zonePoints);
        onOpenChange(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-[90vw] max-h-[90vh] flex flex-col">
                <DialogHeader>
                    <DialogTitle>Define Parking Zone</DialogTitle>
                </DialogHeader>

                <div className="relative flex-1 min-h-0 overflow-hidden border rounded bg-black flex items-center justify-center">
                    {/* Container for Image and Overlay */}
                    <div className="relative inline-block">
                        <img
                            ref={imageRef}
                            src={imageUrl}
                            alt="Camera Snapshot"
                            className="max-h-[70vh] object-contain"
                            onLoad={handleImageLoad}
                        />
                        {imageSize && (
                            <canvas
                                ref={canvasRef}
                                width={imageSize.width}
                                height={imageSize.height}
                                className="absolute top-0 left-0 w-full h-full cursor-crosshair"
                                onClick={handleCanvasClick}
                            />
                        )}
                    </div>
                </div>

                <DialogFooter className="gap-2">
                    <div className="flex-1 text-sm text-muted-foreground self-center">
                        Click corner points to define the area. {points.length} points added.
                    </div>
                    <Button variant="outline" onClick={() => setPoints([])}>Reset</Button>
                    <Button onClick={handleSave} disabled={points.length < 3}>Save Zone</Button>
                </DialogFooter>
            </DialogContent>
        </Dialog>
    );
}
