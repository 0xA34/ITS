# Vehicle Detection & Monitoring

A real-time traffic monitoring and analysis system that uses YOLO-based object detection to track vehicles, detect violations, and measure traffic density from camera feeds.

## Features

- **Real-time Vehicle Detection** - Detect and track vehicles using YOLO v8/v12 models with IOU-based tracking
- **Zone-based Monitoring** - Define custom zones for parking, traffic lights, stop lines, and counting
- **Violation Detection** - Automatic detection of parking violations and red light violations
- **Line Counting** - Count vehicles crossing defined lines with directional tracking
- **Traffic Density Analysis** - Measure traffic flow with historical hourly comparisons
- **Model Benchmarking** - Compare and evaluate different YOLO models

## Tech Stack

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS
- shadcn/ui

### Backend
- FastAPI (Python)
- Ultralytics YOLO
- OpenCV
- SimpleTracker (IOU-based object tracking)

## Getting Started

### Prerequisites

- Node.js 18+ or Bun
- Python 3.11.9
- CUDA-compatible GPU (recommended for real-time detection)

### Frontend Setup

```sh
cd frontend
npm install
npm run dev
```

Or with Bun:

```sh
cd frontend
bun install
bun run dev
```

The frontend will be available at `http://localhost:8080`

### Backend Setup

```sh
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Project Structure

```
ITS/
├── frontend/           # React frontend application
│   ├── src/
│   │   ├── components/ # UI components
│   │   ├── pages/      # Page components
│   │   └── lib/        # API client and utilities
│   └── ...
├── backend/            # FastAPI backend
│   └── app/
│       ├── routes/     # API endpoints
│       ├── services/   # Business logic
│       ├── models/     # Data models
│       └── data/       # Camera and zone configurations
```

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for the interactive Swagger documentation.

## Configuration

- Camera list: `backend/app/data/ITS.link.json`
- Zone configurations: stored per camera in `backend/app/data/zones/`
- YOLO models: place `.pt` files in `backend/app/preTrainedModels/`
- Violation Storage: stored in `backend/app/data/violations/`

## License

This project is for educational and research purposes.
