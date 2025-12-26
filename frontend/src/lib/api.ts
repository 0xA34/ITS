export interface Camera {
    id: string;
    name: string;
    location?: string;
    url?: string;
}

export interface CamerasResponse {
    items: Camera[];
    total: number;
}

const API_Base = "http://localhost:8000/api";

export async function fetchCameras(skip = 0, limit = 50): Promise<CamerasResponse> {
    const res = await fetch(`${API_Base}/cameras?skip=${skip}&limit=${limit}`);
    if (!res.ok) {
        throw new Error("Failed to fetch cameras");
    }
    return res.json();
}
