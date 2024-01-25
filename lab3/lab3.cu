#include <cuda_runtime.h>
#include <iostream>
#include <png.h>
#include <string>
#define CHANNELS 3

const char *filename_in = "cat.png";
const char *filename_out = "cat_blurred.png";

png_infop info_ptr;
png_bytepp row_pointers;

void readPNG(const char *filename, int &height, int &width)
{
    FILE *fp = fopen(filename, "rb");

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    
    png_infop info_ptr = png_create_info_struct(png_ptr);  
    
    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);
    
    height = png_get_image_height(png_ptr, info_ptr);
    width = png_get_image_width(png_ptr, info_ptr);	
    *row_pointers = (png_bytep *) malloc(sizeof(png_bytep) * height);
        for (int i = 0; i < height; ++i) 
        {
            (*row_pointers)[i] = (png_byte *) malloc(sizeof(png_byte) * width * CHANNELS);
        }
    png_read_image(png_ptr, (*row_pointers));
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);
}

void writePNG(const char *filename)
{
    FILE *fp = fopen(filename, "wb");
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

__global__ void blurKernelGlobal(unsigned char* in, unsigned char* out, int width, int height) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) 
    {
        int blurSize = 5; // You can adjust the blur size
        int pixVal = 0;
        int pixels = 0;

        // Average the surrounding pixels
        for (int blurRow = -blurSize; blurRow <= blurSize; blurRow++) 
        {
            for (int blurCol = -blurSize; blurCol <= blurSize; blurCol++) 
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Check for boundary conditions
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
                {
                    pixVal += in[curRow * width + curCol];
                    pixels++;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

__global__ void blurKernelShared(unsigned char* in, unsigned char* out, int width, int height) 
{
    __shared__ unsigned char* image[];
    size_t imageSize = width * height * CHANNELS * sizeof(unsigned char);
    for (int i = threadIdx.x; i < width; i += blockDim.x){
        image[i] = in[i];
    }    
    __syncthreads();
    
    int blurSize = 5; // You can adjust the blur size
    int col = blockIdx.x * blockDim.x + threadIdx.x - (2 * blockIdx.x + 1) * blurSize;
    int row = blockIdx.y * blockDim.y + threadIdx.y - (2 * blockIdx.y + 1) * blurSize;

    int idx = (row * width + col) * CHANNELS;
    if (col < width && row < height) 
    {        
        int pixVal = 0;
        int pixels = 0;

        // Average the surrounding pixels
        for (int blurRow = -blurSize; blurRow <= blurSize; blurRow++) 
        {
            for (int blurCol = -blurSize; blurCol <= blurSize; blurCol++) 
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Check for boundary conditions
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
                {
                    pixVal += image[curRow * width + curCol];
                    pixels++;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

texture<unsigned char, 1, cudaReadModeElementType> texRef;
__global__ void blurKernelTexture(unsigned char* out, int width, int height) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) 
    {
        int blurSize = 5; // You can adjust the blur size
        int pixVal = 0;
        int pixels = 0;

        // Average the surrounding pixels
        for (int blurRow = -blurSize; blurRow <= blurSize; blurRow++) 
        {
            for (int blurCol = -blurSize; blurCol <= blurSize; blurCol++) 
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                // Check for boundary conditions
                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) 
                {
                    pixVal += in[curRow * width + curCol];
                    pixels++;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}

void applyBlur(const unsigned char *inputImage, unsigned char *outputImage, size_t imageSize, int width, int height, const char* memoryType) 
{    
    int numofGPUs;
    cudaGetDeviceCount(&numofGPUs);
    if (nGPUs >= 1) 
    {
        omp_set_num_threads(nGPUs);
    }
    #pragma omp parallel  
    {
  
    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void ) &d_inputImage, imageSize);
    cudaMalloc((void **) &d_outputImage, imageSize);
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int blurSize = 1;
    
    dim3 GridSize(width / blockSize, height / blockSize, 1);
    dim3 BlockSize(blockSize, blockSize, 1);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    if (type == "global") 
    {
        blurKernelGlobal<<<GridSize,BlockSize>>>(d_inputImage, d_outputImage, width, height);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "Time:" << duration << "us." << std::endl;
    } 
    else if (type == "shared") 
    {
        blurKernelShared<<<GridSize,BlockSize>>>(d_inputImage, d_outputImage, width, height);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "Time:" << duration << "us." << std::endl;
    } 
    else if (type == "texture") 
    {
        blurKernelTexture<<<GridSize, BlockSize>>>(d_outputImage, width, height);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        std::cout << "Time:" << duration << "us." << std::endl;
    }   
    
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::cout << "Time:" << duration << "us." << std::endl;

    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
}

int main()
{    
    int height;
    int width;
    read_png(filename_in, height, width);

    size_t imageSize = width * height * CHANNELS * sizeof(unsigned char);
    auto *inputImage = new unsigned char[imageSize];
    auto *outputImage = new unsigned char[imageSize]; 

    auto start_time = std::chrono::high_resolution_clock::now();
    // Copy data from host to device
    cudaMemcpy(d_inImage, h_inImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    applyBlur(inputImage, outputImage, imageSize, width, height, "global");
    applyBlur(inputImage, outputImage, imageSize, width, height, "shared");
    applyBlur(inputImage, outputImage, imageSize, width, height, "texture");
        
    write_png(filename_out);

    // Free device memory
    cudaFree(d_inImage);
    cudaFree(d_outImage);

    // Free host memory
    delete[] h_inImage;
    delete[] h_outImage;

    return 0;
}
