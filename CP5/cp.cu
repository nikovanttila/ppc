/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
using namespace std;
#include <stdio.h>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void mykernel(float *result, const float *data, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny)
        return;
    if (i < j) {
        result[i + j * ny] = 0;
        return;
    }
    //printf("Iteration j: %d, i: %d\n", j, i);
    float product = 0;
    for (int k = 0; k < nx; ++k) {
        float x = data[k + i * nx];
        float y = data[k + j * nx];
        //float x = data[k + j * ny];
        //float y = data[i + k * nx];
        float z = x * y;
        product = product + z;
        //product += data[k + i * nx] * data[k + j * nx];
        //printf("k: %d \n", k);
    }
    result[i + j * ny] = product;
    //printf("product: %d \n", product);
    //printf("result: %d \n", result[i + j * ny]);
}

void correlate(int ny, int nx, const float *data, float *result) {

    // create and initialize input vector
    //float normalized[nx * ny];
    std::vector<float> normalized(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = data[i + j * nx];
        }
    }

    // normalize input vector with each row having arithmetic mean of 0
    for (int j = 0; j < ny; j++) {
        float sum_row = 0;
        // calculate the sum of current row
        for (int i = 0; i < nx; i++) {
            sum_row += normalized[i + j * nx];
        }
        // calculate the mean of current row 
        float mean_row = sum_row / nx;
        // substract the mean of row from row columns
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] -= mean_row;
        }
    }

    // normalize the input vector with each row the sum of the squares of the elements being 1
    for (int j = 0; j < ny; j++) {
        float sum_square = 0;
        // calculate the sum of squares in the row
        for (int i = 0; i < nx; i++) {
            sum_square += normalized[i + j * nx] * normalized[i + j * nx];
        }
        // calculate the factor that is used for multiplying each element
        float factor = sqrt(1 / sum_square);
        // mulitply each element with the factor
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = normalized[i + j * nx] * factor;
        }
    }

    // calculate the (upper triangle of the) matrix product result = normalized * normalized^T
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, normalized.data(), nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    //cout  << "\n" << "dimBlock.y: " << dimBlock.y << ", dimBlock.x: " << dimBlock.x;
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    //cout  << "\n" << "dimGrid.y: " << dimGrid.y << ", dimGrid.x: " << dimGrid.x;
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}
