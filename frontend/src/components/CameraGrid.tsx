// src/components/CameraGrid.tsx
import type { Camera } from "@/lib/api";
import CameraFeed from "./CameraFeed";

interface CameraGridProps {
  cameras: Camera[];
}

export default function CameraGrid({ cameras }: CameraGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
      {cameras.map((camera) => (
        <CameraFeed key={camera.id} camera={camera} />
      ))}
    </div>
  );
}
