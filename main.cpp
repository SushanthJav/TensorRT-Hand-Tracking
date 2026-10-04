#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

// Full technical connection map
const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},       // Thumb
    {0, 5}, {5, 6}, {6, 7}, {7, 8},       // Index
    {9, 10}, {10, 11}, {11, 12},          // Middle
    {13, 14}, {14, 15}, {15, 16},         // Ring
    {17, 18}, {18, 19}, {19, 20},         // Pinky
    {5, 9}, {9, 13}, {13, 17}, {0, 17},   // Palm
    {2, 5}, {5, 9}, {9, 13}, {13, 17}     // Cross-knuckle support for angles
};

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// Helper to smooth out jitters (Temporal Filtering)
struct SmoothPoint {
    float x = 0, y = 0;
    void update(float nx, float ny, float alpha) {
        x = x + alpha * (nx - x);
        y = y + alpha * (ny - y);
    }
};

int main() {
    const std::string enginePath = "C:\\Users\\default.LAPTOP-P0KIEEBH\\hand_landmark_full.engine";
    const int INPUT_SIZE = 224;
    const float SCORE_THRESHOLD = 0.45f; 
    const float SMOOTHING_FACTOR = 0.65f; // Adjust between 0.1 (very smooth/laggy) and 1.0 (raw/jittery)

    std::ifstream file(enginePath, std::ios::binary);
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    float *d_input, *d_landmarks, *d_score, *d_extra2, *d_extra3;
    cudaMalloc(&d_input, 1 * 3 * INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_landmarks, 63 * sizeof(float));
    cudaMalloc(&d_score, 1 * sizeof(float));
    cudaMalloc(&d_extra2, 63 * sizeof(float));
    cudaMalloc(&d_extra3, 1 * sizeof(float));

    context->setTensorAddress("input_1", d_input);
    context->setTensorAddress("Identity", d_landmarks);
    context->setTensorAddress("Identity_1", d_score);
    context->setTensorAddress("Identity_2", d_extra2);
    context->setTensorAddress("Identity_3", d_extra3);

    cv::VideoCapture cap(0);
    cv::Mat frame;
    std::vector<SmoothPoint> smoothedPts(21);

    while (cap.read(frame)) {
        int screenW = frame.cols;
        int screenH = frame.rows;

        // --- ASPECT-CORRECT PREPROCESSING ---
        // Calculate scale such that the image fits inside 224x224 without stretching
        float scale = std::min((float)INPUT_SIZE / screenW, (float)INPUT_SIZE / screenH);
        int newW = (int)(screenW * scale);
        int newH = (int)(screenH * scale);
        
        // Use INTER_AREA for better quality at a distance
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);

        // Letterbox setup
        cv::Mat canvas = cv::Mat::zeros(cv::Size(INPUT_SIZE, INPUT_SIZE), CV_8UC3);
        int offsetX = (INPUT_SIZE - newW) / 2;
        int offsetY = (INPUT_SIZE - newH) / 2;
        resized.copyTo(canvas(cv::Rect(offsetX, offsetY, newW, newH)));

        // Inference Preparation
        cv::Mat blob;
        cv::cvtColor(canvas, blob, cv::COLOR_BGR2RGB);
        blob.convertTo(blob, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(blob, channels);
        for (int i = 0; i < 3; ++i) {
            cudaMemcpy(d_input + i * INPUT_SIZE * INPUT_SIZE, channels[i].data, 
                       INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        }

        context->enqueueV3(0);

        float score = 0;
        cudaMemcpy(&score, d_score, sizeof(float), cudaMemcpyDeviceToHost);

        if (score > SCORE_THRESHOLD) {
            float rawLnd[63];
            cudaMemcpy(rawLnd, d_landmarks, 63 * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<cv::Point> finalPts(21);
            for (int i = 0; i < 21; i++) {
                // Map from 224 square back to original frame dimensions
                float rawX = (rawLnd[i * 3] - offsetX) / scale;
                float rawY = (rawLnd[i * 3 + 1] - offsetY) / scale;

                // Apply smoothing to prevent jitter at extreme angles
                smoothedPts[i].update(rawX, rawY, SMOOTHING_FACTOR);
                finalPts[i] = cv::Point((int)smoothedPts[i].x, (int)smoothedPts[i].y);
            }

            // Draw Lines (Shadow effect for realism)
            for (const auto& c : HAND_CONNECTIONS) {
                cv::line(frame, finalPts[c.first], finalPts[c.second], cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
                cv::line(frame, finalPts[c.first], finalPts[c.second], cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }

            // Draw Nodes
            for (int i = 0; i < 21; i++) {
                cv::circle(frame, finalPts[i], 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
                cv::circle(frame, finalPts[i], 6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }

        cv::imshow("RTX 4070 - Realistic Technical Hand", frame);
        if (cv::waitKey(1) == 27) break;
    }

    cudaFree(d_input); cudaFree(d_landmarks); cudaFree(d_score); 
    cudaFree(d_extra2); cudaFree(d_extra3);
    delete context; delete engine; delete runtime;
    return 0;
}
