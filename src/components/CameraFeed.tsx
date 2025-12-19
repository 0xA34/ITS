// src\components\CameraFeed.tsx
import { useEffect, useState, useRef, useMemo } from "react";
import Hls from "hls.js" // hls 

interface CameraFeedProps {
  cameraId: string;
  cameraName: string;
  url: string;
   
}

// url_m3u8.replace(/[&?]videoUrl=.*/, "")
// xử lí url + ảnh + video
function xuli_m3u8(url_m3u8: string): string{
  try {
    // url
    const u = new URL(url_m3u8);
    if (u.href.includes("videoUrl")){
      console.log(url_m3u8.replace(/[&?]videoUrl=.*/, ""));
      return url_m3u8.replace(/[&?]videoUrl=.*/, "")
    } else {
      return null;
    }
  } catch {
    // path video + anh
    console.log('2'); 
  }
}


const CameraFeed = ({ cameraId, cameraName, url }: CameraFeedProps) => {
  const x = false
  
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const m3u8Url = useMemo(() => xuli_m3u8(url), [url]);

  const isHls = !!m3u8Url;
  console.log(isHls)
  useEffect(() => {
    if (!isHls || !m3u8Url || !videoRef.current) return;

    const video = videoRef.current;

    // Safari / iOS có thể play native
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = m3u8Url;
      return;
    }

    // Chrome/Edge/Firefox: dùng hls.js
    if (Hls.isSupported()) {
      const hls = new Hls({
        lowLatencyMode: true,
      });
      hls.loadSource(m3u8Url);
      hls.attachMedia(video);

      return () => {
        hls.destroy();
      };
    }
  }, [isHls, m3u8Url]);

  // Thời gian
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000); // cập nhật mỗi giây
    return () => clearInterval(id);
  }, []);

  // update hình ảnh sau 12s dcmmm m Lâm ơi 
  const [src, setSrc] = useState(`${url}&t=${Date.now()}`);
  const timerRef = useRef<number | null>(null);
  useEffect(()=>{
    const refresh = () => {
      const next = `${url}&t=${Date.now()}`;
      const img = new Image();
      img.onload = () => setSrc(next);
      img.src = next;
    };

    refresh();
    timerRef.current = window.setInterval(refresh, 12000);

    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  return (
    <div
      className={`relative bg-camera-bg rounded-lg overflow-hidden border border-border transition-all duration-300 hover:border-primary/50 ${
        x ? "fixed inset-4 z-50" : ""
      }`}
    >
      {/* Camera View */}
      <div className="aspect-video bg-gradient-to-br from-secondary to-camera-bg flex items-center justify-center relative group">
        {!isHls ? (<img src={url} className="h-full w-full object-cover" />): (<iframe src={url} className="h-full w-full object-cover"/>)}
      </div>

      {/* thông tin IdCamera + cameraName */}
      <div className="bg-card p-3 border-t border-border">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-sm text-foreground">{cameraName}</h3>
            <p className="text-xs text-muted-foreground">ID: {cameraId}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-primary font-mono">
              {now.toLocaleTimeString("vi-VN")}
            </p>
            <p className="text-xs text-muted-foreground">
              {new Date().toLocaleDateString('vi-VN')}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraFeed;
