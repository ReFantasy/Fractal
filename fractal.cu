#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "opencv2/opencv.hpp"

struct Vector2f
{
    float x;
    float y;

    __host__ __device__
        float norm()
    {
        return sqrt(x * x + y * y);
    }
};

__host__ __device__
Vector2f operator+(Vector2f a, Vector2f b)
{
    Vector2f tmp;
    tmp.x = a.x + b.x;
    tmp.y = a.y + b.y;
    return tmp;
}

__host__ __device__
Vector2f complex_sqr(Vector2f z)
{
    return Vector2f{ z.x * z.x - z.y * z.y, z.x * z.y * 2 };
}

__global__
void TestEigen(int n, float t, float *pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
   
    /*if (threadIdx.x < 10)
    {
        pixels[j * 2 * n + i] = 1;
    }
    else
    {
        pixels[j * 2 * n + i] = 0;
    }*/

    Vector2f c = Vector2f{ -0.8,cos(t) * 0.2 };
    Vector2f z = Vector2f{ (double)i / n - 1,(double)j / n - 0.5 };
    z.x = z.x * 2;
    z.y = z.y * 2;

    int iterator = 0;
    while ((z.norm() < 20) && (iterator < 50))
    {
        z = complex_sqr(z) + c;
        iterator++;
    }
    //printf("%d\n", iterator);
    pixels[j * 2 * n + i] = 1-iterator*0.02;
}



int main(int argc, char*argv[])
{
   
    int n = 320;
    float t = 10000;

    float* pixels;
    cudaMallocManaged((void**)&pixels, 2 * n * n * sizeof(float));

    dim3 gridSize(20, 10);
    dim3 blockSize(32,32);
    
    cv::Mat img(n, 2 * n, CV_32FC1);

    // ��ʱ��ʼ
    cudaEvent_t start_gpu = 0, stop_gpu = 0;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    for (int i = 0; i < 1000000; i++)
    {
        
        cudaEventRecord(start_gpu);

        TestEigen << <gridSize, blockSize >> > (n, i*0.03, pixels);
        cudaDeviceSynchronize();

        cudaEventRecord(stop_gpu);
        cudaEventSynchronize(stop_gpu);
        float time_matrix_add_gpu = 0;
        cudaEventElapsedTime(&time_matrix_add_gpu, start_gpu, stop_gpu);
        //std::cout << "consumed time: " << time_matrix_add_gpu << " ms" << std::endl;
        std::cout << 1000.0 / time_matrix_add_gpu << " fps" << std::endl;

        memcpy(img.data, pixels, 2 * n * n * sizeof(float));

        cv::imshow("res", img);
        cv::waitKey(1);

        
    }

    

    

   
    

	return 0;
}