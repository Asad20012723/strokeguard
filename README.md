# Stroke Monitoring System

An AI-powered stroke risk assessment application that combines facial expression analysis with health metrics to predict stroke risk.

## Features

- **Multimodal Analysis**: Combines facial image analysis with health data for comprehensive risk assessment
- **Webcam Support**: Capture facial expressions directly from your webcam
- **Image Upload**: Alternative option to upload existing photos
- **Risk Visualization**: Clear visual representation of risk score with contributing factors
- **Health Recommendations**: Personalized recommendations based on assessment results

## Architecture

```
stroke-monitoring-system/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI application
│   │   ├── config.py    # Environment configuration
│   │   ├── api/routes/  # API endpoints
│   │   ├── models/      # Pydantic schemas & ML models
│   │   └── services/    # Business logic
│   ├── models/          # PyTorch model weights
│   └── Dockerfile
├── frontend/            # Next.js frontend
│   ├── src/
│   │   ├── app/        # Next.js pages
│   │   ├── components/ # React components
│   │   └── lib/        # Utilities
│   └── Dockerfile
└── docker-compose.yml
```

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)

## Quick Start

### Using Docker (Recommended)

1. Clone the repository and navigate to the project directory

2. Start all services:
   ```bash
   docker-compose up --build
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Local Development

#### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment file:
   ```bash
   cp .env.example .env
   ```

5. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

#### Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Copy environment file:
   ```bash
   cp .env.local.example .env.local
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

## Deployment

### Backend (Railway/Render)

1. Create a new project on Railway or Render
2. Connect your GitHub repository
3. Set the root directory to `backend`
4. Configure environment variables:
   - `CORS_ORIGINS`: Your frontend Vercel URL
   - `MODEL_DEVICE`: `cpu`
5. Deploy

### Frontend (Vercel)

1. Import your GitHub repository to Vercel
2. Set the root directory to `frontend`
3. Configure environment variables:
   - `NEXT_PUBLIC_API_URL`: Your backend URL
4. Deploy

## API Endpoints

### Health Check
```
GET /health
```

### Predict Stroke Risk
```
POST /api/v1/predict
Content-Type: application/json

{
  "health_data": {
    "age": 55,
    "gender": "male",
    "systolic": 130,
    "diastolic": 85,
    "glucose": 110,
    "bmi": 26.5,
    "cholesterol": 210,
    "smoking": 0
  },
  "images": {
    "kiss": "base64_encoded_image",
    "normal": "base64_encoded_image",
    "spread": "base64_encoded_image",
    "open": "base64_encoded_image"
  }
}
```

### Response
```json
{
  "risk_score": 0.35,
  "risk_level": "moderate",
  "confidence": 0.78,
  "contributing_factors": [
    {
      "factor": "High Blood Pressure",
      "value": 145,
      "threshold": 140,
      "severity": "moderate"
    }
  ],
  "recommendations": [
    "Monitor blood pressure regularly",
    "Schedule a check-up with your doctor"
  ],
  "processing_time_ms": 1250.5
}
```

## Tech Stack

### Backend
- FastAPI
- PyTorch
- OpenCV
- Pydantic

### Frontend
- Next.js 14
- React
- Tailwind CSS
- TypeScript

## Disclaimer

This application is for educational and informational purposes only. It is not intended to provide medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## License

MIT License
