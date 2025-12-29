# Production Medical AI Inference System  
## Spine DICOM Anomaly Detection (AutoEncoder)

A **production-ready medical AI inference service** for detecting anomalies in spinal MRI DICOM images using a convolutional autoencoder.

The system is designed as a **backend inference API**, not a research notebook, and can be integrated into downstream applications or clinical research pipelines.

> ⚠️ This project is intended as a **decision-support system**, not a diagnostic tool.

---

## 🧠 Problem

Medical imaging systems often lack labeled anomaly data, making supervised classification difficult.  
This project addresses that challenge by using **reconstruction-based anomaly detection** on spinal MRI DICOM images.

The goal is to provide:
- Reliable inference
- Stable deployment
- Clear API contracts
- Production-oriented system design

---

## 💡 Solution Overview

- A **convolutional autoencoder** trained on normal spinal MRI images
- An **RCA (Reconstruction-based Cleaning Autoencoder)** training loop to iteratively remove noisy samples
- A **FastAPI inference service** for real-time anomaly detection
- Fully **Dockerized** with **CI/CD automation**
- Optional **Streamlit UI** as a demo layer on top of the API

---

## 🏗 System Architecture

```

Client (Streamlit / API Consumer)
↓
FastAPI Inference Service
↓
DICOM Preprocessing
↓
AutoEncoder Reconstruction
↓
Reconstruction Error Scoring
↓
JSON Response + Logging

```

---

## 📡 API Contract

### **POST /predict**

Runs anomaly inference on a single DICOM image.

**Request**
```

POST /predict?threshold=<float>
Content-Type: multipart/form-data

````

| Field | Type | Description |
|------|------|-------------|
| file | DICOM file | Spinal MRI image |

**Response**
```json
{
  "reconstruction_mse": 0.023,
  "is_anomaly": true,
  "inference_time_ms": 142
}
````

---

## 🧪 Model & Training Strategy

### AutoEncoder

* CNN-based encoder–decoder
* Learns reconstruction of normal spinal MRI images
* Anomalies detected via high reconstruction error

### RCA (Reconstruction-based Cleaning Autoencoder)

Training is performed in multiple iterations:

1. Train autoencoder on current dataset
2. Compute reconstruction errors
3. Remove top percentile of high-error samples
4. Retrain on cleaned dataset

This improves robustness by ensuring only **clean normal images** define the anomaly threshold.

---

## 📦 Installation & Running

### Docker (Recommended)

```bash
git clone https://github.com/AyaMYousef/Dicom_AutoEncoder_Spine.git
cd Dicom_AutoEncoder_Spine

docker build -t dicom-autoencoder:latest .
docker run -p 8000:8000 dicom-autoencoder:latest
```

FastAPI will be available at:

```
http://localhost:8000
```

---

## 🚀 CI/CD & Deployment

This repository includes a **GitHub Actions pipeline** that:

1. Builds the Docker image
2. Runs automated checks
3. Publishes the image to **GitHub Container Registry (GHCR)**

Example pull:

```bash
docker pull ghcr.io/<username>/dicom-autoencoder:latest
```

The service can be deployed to:

* Railway
* Render
* AWS ECS / EC2
* Any Docker-compatible platform

---

## 📊 Performance (Baseline)
**How the baseline was calculated:**  
The average inference latency was computed from multiple test runs of the model on sample DICOM and image files using a CPU. Each run measured the time taken for the model to reconstruct the input and compute the anomaly score (`inference_time_ms`). The minimum and maximum latencies observed across these runs are also reported to indicate variation. Model size limit is based on the saved model file and preprocessing constraints.

| Metric                   | Value            |
|--------------------------|----------------|
| Avg inference latency     | ~103 ms (CPU)  |
| Min inference latency     | 59 ms           |
| Max inference latency     | 300 ms          |
| Model size               | ~3.9 MB         |
| Supported format         | DICOM (.dcm)    |

---

## 🖥 Streamlit Demo (Optional)

A Streamlit interface is provided as a **demo UI** on top of the FastAPI backend.

> The **FastAPI service is the core production component**.
> Streamlit is used only for visualization and interaction.

### Run Streamlit

```bash
streamlit run streamlit_app.py
```

Ensure FastAPI is running at:

```
http://127.0.0.1:8000
```

---

## 🔍 Logging & Observability

* Structured request and error logging
* Inference latency measured per request
* Designed for easy integration with monitoring tools

---

## 🛠 Future Improvements

* Grafana monitoring
* Batch inference support

---
