# Real-Time TensorRT C++ Hand Detection & Tracking Engine

A low-latency, high-performance C++ implementation for real-time hand landmark estimation using **NVIDIA TensorRT**, **CUDA**, and **OpenCV**.

## Features
- **NVIDIA TensorRT Acceleration:** Low-latency deep learning inference directly on the GPU.
- **Aspect-Correct Preprocessing:** Custom letterboxing routine preserving original frame proportions.
- **Planar (NCHW) Data Preparation:** High-speed CUDA memory transfer (`cudaMemcpy`).
- **Temporal Filtering:** Low-pass exponential smoothing to eliminate landmark jitter.

## Technical Details
- **Input Size:** 224x224 (Aspect-correct letterboxed canvas)
- **Keypoints Tracked:** 21 3D hand landmark coordinates
- **Dependencies:** C++17, CUDA Toolkit, NVIDIA TensorRT, OpenCV 4.x

## Engine Setup
Place your compiled TensorRT model engine (`hand_landmark_full.engine`) in your working directory and ensure `enginePath` in `main.cpp` points to its local path.
