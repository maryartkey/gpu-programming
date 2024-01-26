#include <iostream>
#include <curand_kernel.h>
#include <png.h>
#include <cmath>
#include <cstring>
#include <chrono>

#define CHANNELS 3
#define BLOCK_SIZE 16
#define POINT_RADIUS 10
#define N 131072
#define K 5

void readPNG(const char* filename, int& height, int& width)
{
    FILE* fp = fopen(filename, "rb");

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    png_infop info_ptr = png_create_info_struct(png_ptr);

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    height = png_get_image_height(png_ptr, info_ptr);
    width = png_get_image_width(png_ptr, info_ptr);
    *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; ++i)
    {
        (*row_pointers)[i] = (png_byte*)malloc(sizeof(png_byte) * width * CHANNELS);
    }
    png_read_image(png_ptr, (*row_pointers));
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    fclose(fp);
}

void writePNG(const char* filename)
{
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, (*row_pointers));
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    int Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    float sobel_x = 0.0f;
    float sobel_y = 0.0f;
    int pixelSize = 1; // Assuming a grayscale image

    for (int i = -1; i <= 1; i++) 
    {
        for (int j = -1; j <= 1; j++) 
        {
            int pixel_x = min(max(x + i, 0), width - 1);
            int pixel_y = min(max(y + j, 0), height - 1);

            unsigned char pixel = input[pixel_y * width + pixel_x];
            sobel_x += pixel * Gx[i + 1][j + 1];
            sobel_y += pixel * Gy[i + 1][j + 1];
        }
    }

    int magnitude = sqrt(sobel_x * sobel_x + sobel_y * sobel_y);
    magnitude = min(max(magnitude, 0), 255);

    output[y * width + x] = static_cast<unsigned char>(magnitude);
}

void applySobelFilter(const std::vector<unsigned char>& inputImage, std::vector<unsigned char>& outputImage, int width, int height) 
{
    unsigned char* d_input, * d_output;
    int imageSize = width * height;

    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, imageSize);

    cudaMemcpy(d_input, inputImage.data(), imageSize, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    sobelKernel << <dimGrid, dimBlock >> > (d_input, d_output, width, height);

    cudaMemcpy(outputImage.data(), d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

__global__ void generateRandomSamples(int* x, int* y, int width, int height, int N, int K) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        curandState state;
        curand_init(1234, idx, 0, &state);

        for (int i = 0; i < K; i++) 
        {
            x[idx * K + i] = curand(&state) % width;
            y[idx * K + i] = curand(&state) % height;
        }
    }
}

__device__ void leastSquaresCircle(const int* points, int numPoints, int width, int& centerX, int& centerY, int& radius) 
{    
    // (x-a)^2 + (y-b)^2 = r^2
    a = 0;
    b = 0;
    radius = 0;

    for (int iter = 0; iter < 3; ++iter) 
    {
        float sumX = 0.0, sumY = 0.0, sumDist = 0.0;
        for (int i = 0; i < numPoints; ++i) 
        {
            float dx = (points[i] % width) - a;
            float dy = (points[i] / width) - b;
            float dist = sqrt(dx * dx + dy * dy);

            sumX += dx * (dist - radius) / dist;
            sumY += dy * (dist - radius) / dist;
            sumDist += (dist - radius) * (dist - radius);
        }

        a += sumX / numPoints;
        b += sumY / numPoints;
        radius = sqrt(sumDist / numPoints);
    }
}


void drawCircle(unsigned char* image, int width, int height, int index, int radius) {
    int x = index % width;
    int y = index / width;

    if !(x < 0 || x >= width || y < 0 || y >= height) 
    {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int dx = j - x;
                int dy = i - y;
                int distanceSquared = dx * dx + dy * dy;

                if (((radius - 2) * (radius - 2) <= distanceSquared) && (distanceSquared <= radius * radius)) 
                {
                    int new_index = (i * width + j) * 3;
                    image[new_index] = 0;
                    image[new_index + 1] = 0;
                    image[new_index + 2] = 0;
                }
            }
        }
    }    
}


__global__ void searchCircle(const int* whitePoints, int* array, int* pointsArray, int numWhitePoints, int width, int height, float* results) 
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < N) 
    {
        curandState state;
        curand_init(threadId, 0, 0, &state);

        int selectedPoints[K];
        for (int& selectedPoint : selectedPoints) {
            int randomIndex = curand(&state) % numWhitePoints;
            selectedPoint = whitePoints[randomIndex];
        }

        int a, b, radius;
        leastSquaresCircle(selectedPoints, K, width, centerX, centerY, radius);

        // Creating shortlist
        float error = 0.0;
        for (int selectedPoint : selectedPoints) {
            int dx = (selectedPoint % width) - centerX;
            int dy = (selectedPoint / width) - centerY;
            float distance = std::sqrt((float)(dx * dx + dy * dy));
            error += abs(distance - radius);
        }

        results[threadId] = error;

        array[threadId * 2] = centerY * width + centerX;
        array[threadId * 2 + 1] = radius;

        for (int i = 0; i < K; ++i) {
            pointsArray[threadId * K + i] = selectedPoints[i];
        }
    }
}

int* applySearchCircle(const int* whitePoints, int numWhitePoints, int width, int height, int& centerIndex, int& radius) 
{
    int* d_whitePoints;
    cudaMalloc((void**)&d_whitePoints, sizeof(int) * numWhitePoints);
    cudaMemcpy(d_whitePoints, whitePoints, sizeof(int) * numWhitePoints, cudaMemcpyHostToDevice);

    float* d_results;
    cudaMalloc((void**)&d_results, sizeof(float) * N);

    int* d_array;
    cudaMalloc((void**)&d_array, sizeof(int) * N * 2);

    int* d_pointsArray;
    cudaMalloc((void**)&d_pointsArray, sizeof(int) * N * K);


    dim3 GS(N / 256);
    dim3 BS(256);

    processPoints << <GS, BS >> > (d_whitePoints, d_array, d_pointsArray, numWhitePoints, width, height, d_results);

    cudaDeviceSynchronize();

    auto* h_results = new float[N];
    cudaMemcpy(h_results, d_results, sizeof(float) * N, cudaMemcpyDeviceToHost);

    int bestResultIndex = 0;
    float bestResult = h_results[0];
    for (int i = 1; i < N; ++i) {
        if (h_results[i] < bestResult) {
            bestResultIndex = i;
            bestResult = h_results[i];
        }
    }

    int* h_array = new int[2];
    cudaMemcpy(h_array, d_array + bestResultIndex * 2, sizeof(int) * 2, cudaMemcpyDeviceToHost);

    centerIndex = h_array[0];
    radius = h_array[1];

    int* h_pointsArray = new int[K];
    cudaMemcpy(h_pointsArray, d_pointsArray + bestResultIndex * K, sizeof(int) * K, cudaMemcpyDeviceToHost);

    cudaFree(d_whitePoints);

    delete[] h_results;
    cudaFree(d_results);

    delete[] h_array;
    cudaFree(d_array);

    return h_pointsArray;
}

int main() {
    const char* inputFileName = "sign.png";
    int width, height;
    png_bytep* input_row_pointers;

    read_png(filename_in, height, width);

    size_t imageSizeIn = width * height * CHANNELS * sizeof(unsigned char);
    size_t imageSizeOut = width * height * sizeof(unsigned char);

    auto* inputImage = new unsigned char[inputImageSize];
    auto* outputImage = new unsigned char[outputImageSize];

    for (int i = 0; i < height; ++i) {
        memcpy(&inputImage[i * width * CHANNELS * sizeof(unsigned char)],
            input_row_pointers[i],
            width * CHANNELS * sizeof(unsigned char));
    }

    int numWhitePoints = 0;
    int* whitePoints = applyBlur(inputImage, outputImage, imageSizeIn, imageSizeOut, width, height, numWhitePoints);

    int centerIndex, radius;
    int* points = applySearchCircle(whitePoints, numWhitePoints, width, height, centerIndex, radius);

    drawCircle(inputImage, width, height, centerIndex, radius);

    for (int i = 0; i < K; i++) {
        drawPoint(inputImage, width, height, points[i]);
    }

    for (int i = 0; i < height; ++i) {
        memcpy(input_row_pointers[i],
            &inputImage[i * width * CHANNELS * sizeof(unsigned char)],
            width * CHANNELS * sizeof(unsigned char));
    }

    auto* output_row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; ++i) {
        output_row_pointers[i] = (png_byte*)malloc(sizeof(png_byte) * width * CHANNELS);
    }

    for (int i = 0; i < height; ++i) {
        memcpy(output_row_pointers[i],
            &outputImage[i * width * sizeof(unsigned char)],
            width * sizeof(unsigned char));
    }

    write_png(filename_out);

    delete[] inputImage;
    delete[] outputImage;

    for (int i = 0; i < height; ++i) {
        delete[] output_row_pointers[i];
    }
    delete[] output_row_pointers;

    for (int i = 0; i < height; ++i) {
        delete[] input_row_pointers[i];
    }
    delete[] input_row_pointers;

    return 0;
}
