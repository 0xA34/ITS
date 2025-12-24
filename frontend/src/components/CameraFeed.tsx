// src/components/CameraFeed.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import Hls from "hls.js";
import type { Camera } from "@/lib/api";

type CameraFeedProps = {
  camera: Camera;
  refreshMs?: number; // refresh ảnh để tránh cache (chỉ áp dụng cho image)
};

function toCacheBustedUrl(url: string) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${Date.now()}`;
}

/**
 * Một số link camera có dạng ...?videoUrl=<m3u8-url>&...
 * Với trường hợp này, FE thường nên dùng URL đã được backend chuẩn hoá.
 * Tuy nhiên để an toàn, ta vẫn "bóc" m3u8 từ param videoUrl nếu có.
 */
function extractHlsUrl(inputUrl: string): string | null {
  try {
    const u = new URL(inputUrl);
    const videoUrl = u.searchParams.get("videoUrl");
    if (videoUrl && (videoUrl.includes(".m3u8") || videoUrl.includes("m3u8"))) {
      return videoUrl;
    }
  } catch {
    // ignore
  }

  if (inputUrl.includes(".m3u8") || inputUrl.includes("m3u8")) return inputUrl;
  return null;
}

export default function CameraFeed({ camera, refreshMs = 12000 }: CameraFeedProps) {
  const { id, name, location, url } = camera;

  // Clock
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(t);
  }, []);
  //
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const hlsUrl = useMemo(() => (url ? extractHlsUrl(url) : null), [url]);
  const isHls = !!hlsUrl;

  // Refresh image src (cache-busting), chỉ cho image
  const [imgSrc, setImgSrc] = useState<string>(() => (url ? toCacheBustedUrl(url) : ""));
  useEffect(() => {
    if (!url) return;
    if (isHls) return; // video không refresh kiểu này

    let timer: number | undefined;

    const refresh = () => {
      const next = toCacheBustedUrl(url);
      const img = new Image();
      img.onload = () => setImgSrc(next);
      img.onerror = () => {
        // vẫn set để user thấy thử lại
        setImgSrc(next);
      };
      img.src = next;
    };

    refresh();
    timer = window.setInterval(refresh, refreshMs);

    return () => {
      if (timer) window.clearInterval(timer);
    };
  }, [url, isHls, refreshMs]);

  // Attach HLS playback
  useEffect(() => {
    if (!isHls || !hlsUrl || !videoRef.current) return;

    const video = videoRef.current;

    // Safari/iOS có thể play native
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = hlsUrl;
      return;
    }

    if (Hls.isSupported()) {
      const hls = new Hls({ lowLatencyMode: true });
      hls.loadSource(hlsUrl);
      hls.attachMedia(video);

      return () => {
        hls.destroy();
      };
    }
  }, [isHls, hlsUrl]);

  return (
    <div className="relative bg-camera-bg rounded-lg overflow-hidden border border-border transition-all duration-300 hover:border-primary/50">
      {/* Camera View */}
      <div className="aspect-video bg-gradient-to-br from-secondary to-camera-bg flex items-center justify-center relative">
        {!url ? (
          <div className="text-sm text-muted-foreground">Chưa có URL camera</div>
        ) : isHls ? (
          <video
            ref={videoRef}
            className="h-full w-full object-cover"
            controls
            muted
            playsInline
          />
        ) : (
          <img src={imgSrc} alt={name} className="h-full w-full object-cover" />
        )}
      </div>

      {/* Info */}
      <div className="bg-card p-3 border-t border-border">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <h3 className="font-semibold text-sm text-foreground truncate">{name}</h3>
            <p className="text-xs text-muted-foreground truncate">
              {location ? location : `ID: ${id}`}
            </p>
          </div>

          <div className="text-right shrink-0">
            <p className="text-xs text-primary font-mono">{now.toLocaleTimeString("vi-VN")}</p>
            <p className="text-xs text-muted-foreground">
              {now.toLocaleDateString("vi-VN")}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
