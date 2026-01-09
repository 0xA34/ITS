import { useState } from "react";
import HeaderSearch from "@/components/HeaderSearch";
import SearchComposer from "@/components/SearchComposer";
import TypewriterTitle from "@/components/TitleSearch";
import { useNavigate } from "react-router-dom";

type Detection = {
  label: string;
  conf: number;
  bbox: [number, number, number, number];
  track_id?: number;
};

type Annotated = { jpegBase64: string; width: number; height: number };

type SyncResult = {
  mode: "sync";
  status: "success" | "empty";
  source?: unknown;
  detections: Detection[];
  summary: Record<string, number>;
  annotated?: Annotated;
};

type UiState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "sync"; data: SyncResult }
  | { phase: "error"; message: string };

const API_BASE = import.meta.env.VITE_API_BASE?.toString() ?? "http://localhost:8000";

// ====== API RIÊNG: Upload image/video ======
async function uploadImage(file: File): Promise<{ url: string }> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload-image`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Upload image failed (${res.status})`);
  }

  return (await res.json()) as { url: string };
}

async function uploadVideo(file: File): Promise<{ url: string; id?: string; name?: string }> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${API_BASE}/api/upload-video`, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Upload video failed (${res.status})`);
  }

  return (await res.json()) as { url: string; id?: string; name?: string };
}

// ====== API: Detect image by URL (backend chạy YOLO) ======
async function detectImageByUrl(imageUrl: string): Promise<SyncResult> {
  const res = await fetch(`${API_BASE}/api/detect-image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageUrl }),
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(msg || `Detect image failed (${res.status})`);
  }

  return (await res.json()) as SyncResult;
}

function toMessage(x: unknown): string {
  if (typeof x === "string") return x;
  if (typeof x === "number") return String(x);
  if (x instanceof Error) return x.message;
  try {
    return JSON.stringify(x);
  } catch {
    return String(x);
  }
}

// ====== Helpers phân loại link ======
const IMAGE_HINTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"];
const VIDEO_HINTS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m3u8"];

function extractFirstUrl(text: string): string | null {
  const m = text.match(/(https?:\/\/\S+)/i);
  if (!m) return null;
  return m[1].replace(/[),.;'"]+$/g, "");
}

function isImageUrl(url: string) {
  const u = url.toLowerCase();
  return IMAGE_HINTS.some((h) => u.includes(h));
}

function isVideoUrl(url: string) {
  const u = url.toLowerCase();
  return VIDEO_HINTS.some((h) => u.includes(h));
}

const Pages = () => {
  const navigate = useNavigate();
  const [ui, setUi] = useState<UiState>({ phase: "idle" });

  // SearchComposer expects onSend({text, files})
  const handleSend = async (payload: { text: string; files: File[] }) => {
    const { text, files } = payload;

    try {
      setUi({ phase: "loading" });

      // ===== 1) Nếu user upload file =====
      if (files?.length > 0) {
        const f = files[0];

        // Ảnh => upload-image => detect-image => show ngay
        if (f.type.startsWith("image/")) {
          const up = await uploadImage(f);
          const sync = await detectImageByUrl(up.url);
          setUi({ phase: "sync", data: sync });
          return;
        }

        // Video => upload-video => redirect sang Detection.tsx
        if (f.type.startsWith("video/")) {
          const up = await uploadVideo(f);
          const id = encodeURIComponent(String(up.id ?? "upload-video"));
          const url = encodeURIComponent(String(up.url));
          const name = encodeURIComponent(String(up.name ?? f.name ?? "Video"));
          setUi({ phase: "idle" });
          navigate(`/detection?id=${id}&url=${url}&name=${name}`);
          return;
        }

        setUi({ phase: "error", message: "File không hỗ trợ. Chỉ nhận image/video." });
        return;
      }

      // ===== 2) Không có file => xử lý theo link trong text =====
      const raw = text?.trim() ?? "";
      const url = extractFirstUrl(raw) ?? raw;

      if (!url) {
        setUi({ phase: "error", message: "Hãy dán link hoặc upload ảnh/video." });
        return;
      }

      // Link ảnh => detect ngay
      if (isImageUrl(url)) {
        const sync = await detectImageByUrl(url);
        setUi({ phase: "sync", data: sync });
        return;
      }

      // Link video/stream => sang Detection
      if (isVideoUrl(url)) {
        const id = encodeURIComponent("search-video");
        const encUrl = encodeURIComponent(url);
        const name = encodeURIComponent("Video");
        setUi({ phase: "idle" });
        navigate(`/detection?id=${id}&url=${encUrl}&name=${name}`);
        return;
      }

      // Không đoán được loại link => coi như stream/camera => sang Detection
      {
        const id = encodeURIComponent("search-stream");
        const encUrl = encodeURIComponent(url);
        const name = encodeURIComponent("Stream");
        setUi({ phase: "idle" });
        navigate(`/detection?id=${id}&url=${encUrl}&name=${name}`);
      }
    } catch (e) {
      setUi({ phase: "error", message: toMessage(e) });
    }
  };

  const renderResult = () => {
    const cardCls =
      "w-full max-w-3xl mt-6 rounded-2xl border border-border bg-card/80 backdrop-blur shadow-sm p-4";

    if (ui.phase === "idle") return null;

    if (ui.phase === "loading") {
      return (
        <div className={cardCls}>
          <div className="text-sm font-medium">Đang xử lý…</div>
          <div className="text-xs text-muted-foreground mt-1">
            Ảnh sẽ trả kết quả ngay • Video/Stream sẽ chuyển sang trang Detection.
          </div>
        </div>
      );
    }

    if (ui.phase === "error") {
      return (
        <div className={cardCls + " border-destructive/40 bg-destructive/10"}>
          <div className="text-sm font-semibold text-destructive">Lỗi</div>
          <div className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap">{ui.message}</div>
        </div>
      );
    }

    // sync
    const dets = ui.data.detections ?? [];
    const summary = ui.data.summary ?? {};
    const imgB64 = ui.data.annotated?.jpegBase64 ?? "";

    return (
      <div className={cardCls}>
        <div className="flex items-center justify-between gap-3">
          <div className="text-sm font-semibold">Kết quả YOLO</div>
          <div className="text-xs text-muted-foreground">Mode: sync • {ui.data.status}</div>
        </div>

        <div className="mt-3 text-sm">
          Tổng đối tượng: <b>{dets.length}</b>
        </div>

        <div className="mt-2 flex flex-wrap gap-2">
          {Object.keys(summary).length === 0 ? (
            <span className="text-xs text-muted-foreground">Không có thống kê</span>
          ) : (
            Object.entries(summary).map(([k, v]) => (
              <span
                key={k}
                className="text-xs px-2 py-1 rounded-md border border-border bg-background/60"
              >
                {k}: <b>{v}</b>
              </span>
            ))
          )}
        </div>

        {imgB64 ? (
          <img
            className="mt-4 w-full rounded-xl border border-border"
            src={`data:image/jpeg;base64,${imgB64}`}
            alt="YOLO annotated"
          />
        ) : null}

        {dets.length === 0 && (
          <div className="mt-3 text-xs text-muted-foreground">Không phát hiện đối tượng.</div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <HeaderSearch />

      {/* keep original layout */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 py-10">
        <div className="w-full max-w-3xl text-center">
          <TypewriterTitle
            lines={[
              "VEHICLE DETECTION & MONITORING",
              "Tìm kiếm nhanh và dễ dàng sử dụng",
              "Dán link hoặc thêm các video + hình ảnh để xác định phương tiện",
            ]}
            className="text-center"
            lineClassNames={[
              "text-[35px] sm:text-4xl md:text-5xl font-extrabold tracking-tight whitespace-nowrap",
              "mt-3 text-[15px] text-muted-foreground",
              "mt-1 text-[15px] text-muted-foreground",
            ]}
          />
        </div>

        <div className="w-full max-w-3xl mt-8">
          <SearchComposer onSend={handleSend} />
        </div>

        {renderResult()}
      </main>
    </div>
  );
};

export default Pages;
