// src/components/CameraSelectDialog.tsx
import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Camera as CameraIcon, Trash2, MapPin, Plus, RefreshCw } from "lucide-react";
import { fetchCameras, type Camera } from "@/lib/api";

interface CameraSelectDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedCameras: Camera[];
  onCamerasChange: (cameras: Camera[]) => void;
}

type LoadState =
  | { status: "idle" | "loading" }
  | { status: "error"; message: string }
  | { status: "success"; items: Camera[] };

export default function CameraSelectDialog({
  open,
  onOpenChange,
  selectedCameras,
  onCamerasChange,
}: CameraSelectDialogProps) {
  const [load, setLoad] = useState<LoadState>({ status: "idle" });

  // tempSelected là danh sách id được tick trong tab "Chọn Camera"
  const [tempSelected, setTempSelected] = useState<string[]>(() =>
    selectedCameras.map((c) => c.id)
  );

  // Khi mở dialog, sync tick hiện tại theo selectedCameras
  useEffect(() => {
    if (!open) return;
    setTempSelected(selectedCameras.map((c) => c.id));
  }, [open, selectedCameras]);

  async function reload() {
    try {
      setLoad({ status: "loading" });
      const data = await fetchCameras();
      setLoad({ status: "success", items: data.items || [] });
    } catch (e: any) {
      setLoad({ status: "error", message: e?.message || "Không tải được danh sách camera" });
    }
  }

  // Load cameras khi mở dialog
  useEffect(() => {
    if (!open) return;
    void reload();
  }, [open]);

  const allCameras = useMemo(() => {
    if (load.status === "success") return load.items;
    return [];
  }, [load]);

  const handleToggleCamera = (cameraId: string) => {
    setTempSelected((prev) =>
      prev.includes(cameraId) ? prev.filter((id) => id !== cameraId) : [...prev, cameraId]
    );
  };

  const handleSave = () => {
    const newCameras = allCameras.filter((cam) => tempSelected.includes(cam.id));
    onCamerasChange(newCameras);
    onOpenChange(false);
  };

  const handleRemoveCamera = (cameraId: string) => {
    const newCameras = selectedCameras.filter((cam) => cam.id !== cameraId);
    onCamerasChange(newCameras);
    setTempSelected((prev) => prev.filter((id) => id !== cameraId));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[900px] bg-card border-border">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <CameraIcon className="w-5 h-5 text-primary" />
            Quản lý Camera
          </DialogTitle>
        </DialogHeader>

        <Tabs defaultValue="select" className="w-full">
          <TabsList className="grid w-full grid-cols-2 bg-secondary">
            <TabsTrigger
              value="select"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <Plus className="w-4 h-4 mr-2" />
              Chọn Camera
            </TabsTrigger>
            <TabsTrigger
              value="selected"
              className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              <CameraIcon className="w-4 h-4 mr-2" />
              Đã chọn ({selectedCameras.length})
            </TabsTrigger>
          </TabsList>

          {/* Tab 1: Chọn camera */}
          <TabsContent value="select" className="mt-4">
            <div className="flex items-center justify-between gap-2 mb-3">
              <div className="text-xs text-muted-foreground">
                {load.status === "loading"
                  ? "Đang tải danh sách camera..."
                  : load.status === "error"
                  ? "Tải danh sách camera lỗi"
                  : `Tổng: ${allCameras.length} camera`}
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={reload}
                className="border-border"
                disabled={load.status === "loading"}
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Tải lại
              </Button>
            </div>

            {load.status === "error" ? (
              <div className="rounded-lg border border-border bg-secondary/50 p-4 text-sm text-destructive">
                {load.message}
              </div>
            ) : (
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {allCameras.map((camera) => (
                  <div
                    key={camera.id}
                    className={`flex items-center gap-4 p-3 rounded-lg border transition-all cursor-pointer ${
                      tempSelected.includes(camera.id)
                        ? "border-primary bg-primary/10"
                        : "border-border bg-secondary/50 hover:bg-secondary"
                    }`}
                    onClick={() => handleToggleCamera(camera.id)}
                  >
                    <Checkbox
                      checked={tempSelected.includes(camera.id)}
                      onCheckedChange={() => handleToggleCamera(camera.id)}
                      className="border-muted-foreground data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground truncate">{camera.name}</p>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground truncate">
                        <MapPin className="w-3 h-3" />
                        {camera.location || "Không có vị trí"}
                      </div>
                    </div>
                    <span className="text-xs text-muted-foreground font-mono">{camera.id}</span>
                  </div>
                ))}

                {load.status === "loading" && (
                  <div className="text-sm text-muted-foreground py-6 text-center">Loading...</div>
                )}

                {load.status === "success" && allCameras.length === 0 && (
                  <div className="text-sm text-muted-foreground py-6 text-center">
                    Không có camera trong database.
                  </div>
                )}
              </div>
            )}

            <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-border">
              <Button variant="outline" onClick={() => onOpenChange(false)} className="border-border">
                Hủy
              </Button>
              <Button onClick={handleSave} className="bg-primary hover:bg-primary/90">
                Lưu thay đổi
              </Button>
            </div>
          </TabsContent>

          {/* Tab 2: Camera đã chọn */}
          <TabsContent value="selected" className="mt-4">
            {selectedCameras.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <CameraIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Chưa có camera nào được chọn</p>
                <p className="text-sm">Chuyển sang tab &quot;Chọn Camera&quot; để thêm</p>
              </div>
            ) : (
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {selectedCameras.map((camera) => (
                  <div
                    key={camera.id}
                    className="flex items-center gap-4 p-3 rounded-lg border border-border bg-secondary/50 group"
                  >
                    <div className="w-16 h-10 rounded bg-muted overflow-hidden">
                      {camera.url ? (
                        <img
                          src={camera.url}
                          alt={camera.name}
                          className="w-full h-full object-cover"
                        />
                      ) : null}
                    </div>

                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-foreground truncate">{camera.name}</p>
                      <div className="flex items-center gap-1 text-xs text-muted-foreground truncate">
                        <MapPin className="w-3 h-3" />
                        {camera.location || "Không có vị trí"}
                      </div>
                    </div>

                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleRemoveCamera(camera.id)}
                      className="text-destructive hover:text-destructive hover:bg-destructive/10 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
