#include <cuda_runtime.h>
#include <iostream>
#include <png.h>
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
        int blurSize = 1; // You can adjust the blur size
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

int main()
{    
    int height;
    int width;
    read_png(filename_in, height, width);

    unsigned char* h_inImage = new unsigned char[width * height];
    unsigned char* h_outImage = new unsigned char[width * height];

    // Allocate memory on the device
    unsigned char* d_inImage;
    unsigned char* d_outImage;
    cudaMalloc(&d_inImage, width * height * sizeof(unsigned char));
    cudaMalloc(&d_outImage, width * height * sizeof(unsigned char));
    auto start_time = std::chrono::high_resolution_clock::now();
    // Copy data from host to device
    cudaMemcpy(d_inImage, h_inImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);    

    // Launch the kernel
    blurKernel<<<dimGrid, dimBlock>>>(d_inImage, d_outImage, width, height);

    // Copy result back to host
    cudaMemcpy(h_outImage, d_outImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    write_png(filename_out);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Время:" << duration << "мкс." << std::endl;

    // Free device memory
    cudaFree(d_inImage);
    cudaFree(d_outImage);

    // Free host memory
    delete[] h_inImage;
    delete[] h_outImage;

    return 0;
}