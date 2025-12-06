# DICOM AutoEncoder - Spine Anomaly Detection

## Description

This project uses a convolutional autoencoder to identify abnormal patterns in spinal DICOM images. The model reconstructs input images and computes a reconstruction error; images with high error are flagged as anomalies.

It provides a **FastAPI** endpoint for inference and is fully containerized with **Docker**, enabling easy deployment.

---

## Features

* Train and save AutoEncoder for spinal DICOM images
* Detect anomalies using reconstruction error
* **RCA-based training** to iteratively clean normal images
* FastAPI API for inference
* Dockerized for seamless deployment
* CI/CD integration via GitHub Actions

---

## Installation

### Clone repository

```bash
git clone https://github.com/AyaMYousef/Dicom_AutoEncoder_Spine.git
cd Dicom_AutoEncoder_Spine
```

### Using Docker

1. Build Docker image:

```bash
docker build -t dicom-autoencoder:latest .
```

2. Run the container:

```bash
docker run -p 8000:8000 dicom-autoencoder:latest
```

The FastAPI server will be available at `http://localhost:8000`.

---

## Training with RCA (Reconstruction-based Cleaning Autoencoder)

The autoencoder is trained using an **iterative RCA loop** to remove images with the highest reconstruction error and retrain clean normal images for training:

```python
X_current = X.copy()  # full dataset of normal images

for loop in range(3):  # 3 RCA iterations
    print(f"\n===== RCA Loop {loop+1} =====")

    # Train AutoEncoder on CURRENT dataset
    autoencoder.fit(
        X_current, X_current,
        epochs=30,
        batch_size=16,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    # Compute reconstruction errors
    recon = autoencoder.predict(X_current)
    errors = np.mean((X_current - recon)**2, axis=(1,2,3))

    # Remove worst 10% images
    threshold = np.percentile(errors, 90)
    keep_idx = np.where(errors < threshold)[0]
    X_current = X_current[keep_idx]

    print("Remaining images:", X_current.shape)
```

After RCA, the **final anomaly threshold** is computed on the fully cleaned training set, which is then used in inference to classify anomalies.

---

## API

**POST** `/predict?threshold=<value>`

* **Request**: send a DICOM image as input
* **Response**: JSON containing:

  * `reconstruction_mse`: float
  * `is_anomaly`: boolean

Example:

```json
{
  "reconstruction_mse": 0.023,
  "is_anomaly": true
}
```

---

## GitHub Actions

* The project includes a CI/CD workflow to:

  1. Build Docker image
  2. Run tests (if configured)
  3. Publish image to GitHub Container Registry (GHCR)

---

## Requirements

* Python 3.11
* TensorFlow 2.18.0
* FastAPI, Uvicorn
* OpenCV, Pydicom, NumPy, Pillow
* Docker

---

## Docker Image

* Built and published via GitHub Actions
* Pull image:

```bash
docker pull ghcr.io/<your-username>/dicom-autoencoder:latest
```

---

## 📊 Streamlit Interface

You can interact with the model through a simple web interface built with **Streamlit**.

### 1️⃣ Installation

Install Streamlit in your Python environment:

```bash
pip install streamlit
```

### 2️⃣ Running the Interface

1. Start your FastAPI backend (if not already running):

```bash
uvicorn app:app --reload
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

> If `streamlit` command is not found, use:

```bash
python -m streamlit run streamlit_app.py
```

3. Open your browser at [http://localhost:8501](http://localhost:8501).

---

### 3️⃣ Usage

* Upload an image ( DICOM) using the file uploader.
* Streamlit sends the file to the FastAPI backend for prediction.
* The model’s prediction will be displayed on the interface.

---

### 4️⃣ Notes

* Ensure FastAPI is running on `http://127.0.0.1:8000` or update the Streamlit code with your backend URL.
* The RCA loop ensures that only **clean normal images** are used to compute the final anomaly threshold, improving the model’s sensitivity to true anomalies.

---
