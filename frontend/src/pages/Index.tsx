// src/pages/Index.tsx (hoặc src/pages/index.tsx)
import { useEffect, useState } from "react";
import CameraGrid from "@/components/CameraGrid";
import SideBar from "@/components/SideBar";
import Header from "@/components/HeaderHome";
import CameraSelectDialog from "@/components/CameraSelectDialog";
import type { Camera } from "@/lib/api";

const STORAGE_KEY = "selectedCameras";

const Index = () => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedCameras, setSelectedCameras] = useState<Camera[]>([]);
  const [isLoaded, setIsLoaded] = useState(false); // Track if initial load is done

  // Load selected cameras từ localStorage (nếu có)
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      console.log("Loading from localStorage:", raw);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          console.log("Loaded cameras:", parsed);
          setSelectedCameras(parsed);
        }
      }
    } catch (e) {
      console.error("Failed to load from localStorage:", e);
    } finally {
      setIsLoaded(true); // Mark as loaded regardless of success/failure
    }
  }, []);

  // Save selected cameras (only after initial load)
  useEffect(() => {
    if (!isLoaded) return; // Don't save until we've loaded from localStorage

    try {
      console.log("Saving to localStorage:", selectedCameras);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(selectedCameras));
    } catch (e) {
      console.error("Failed to save to localStorage:", e);
    }
  }, [selectedCameras, isLoaded]);

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar cố định bên trái */}
      <SideBar />

      {/* Nội dung chính, đẩy sang phải 220px đúng bằng width sidebar */}
      <main className="ml-[220px] p-0">
        <Header onAddCamera={() => setDialogOpen(true)} />
        <CameraGrid cameras={selectedCameras} />
      </main>

      <CameraSelectDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        selectedCameras={selectedCameras}
        onCamerasChange={setSelectedCameras}
      />
    </div>
  );
};

export default Index;
