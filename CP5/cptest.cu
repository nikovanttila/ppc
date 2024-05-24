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

__global__ void dGPUtesting(float* data, int nx, int ny, int nn, bool matrix=false) {
    printf("dGPU:\nlength: %d\n", (int) sizeof(data));
    if (matrix) {
        printf("data:\n");
        for (int i = 0; i < nn; ++i) {
            bool space = false;
            for (int j = 0; j < nn; ++j) {
                if (data[nn * i + j] >= 0 && i < nx && j < ny) {
                    printf("  %f  ", data[nn * i + j]);
                    space = true;
                } else if (data[nn * i + j] < 0 && i < nx && j < ny) {
                    printf(" %f  ", data[nn * i + j]);
                    space = true;
                }
            }
            if (space) printf("\n");
        }
        float* transpose = data + nn * nn;
        printf("transpose:\n");
        for (int i = 0; i < nn; ++i) {
            bool space = false;
            for (int j = 0; j < nn; ++j) {
                if (transpose[nn * i + j] >= 0 && i < nx && j < ny) {
                    printf("  %f  ", transpose[nn * i + j]);
                    space = true;
                } else if (transpose[nn * i + j] < 0 && i < nx && j < ny) {
                    printf(" %f  ", transpose[nn * i + j]);
                    space = true;
                }
            }
            if (space) printf("\n");
        }
    } else {
        printf("data:\n");
        for (int i = 0; i < nn; ++i) {
            for (int j = 0; j < nn; ++j) {
                if (data[nn * i + j] >= 0 && i < nx && j < ny) {
                    printf("j: %d, i: %d,    data[nn * i + j]:  %f\n", j, i, data[nn * i + j]);
                } else if (data[nn * i + j] < 0 && i < nx && j < ny) {
                    printf("j: %d, i: %d,    data[nn * i + j]: %f\n", j, i, data[nn * i + j]);
                }
            }
        }
        float* transpose = data + nn * nn;
        printf("transpose:\n");
        for (int i = 0; i < nn; ++i) {
            for (int j = 0; j < nn; ++j) {
                if (transpose[nn * i + j] >= 0 && i < nx && j < ny) {
                    printf("j: %d, i: %d,    transpose[nn * i + j]:  %f\n", j, i, transpose[nn * i + j]);
                } else if (transpose[nn * i + j] < 0 && i < nx && j < ny) {
                    printf("j: %d, i: %d,    transpose[nn * i + j]: %f\n", j, i, transpose[nn * i + j]);
                }
            }
        }
    }
}

__global__ void rGPUtesting(float* data, int nx, int ny, bool matrix=false) {
    printf("rGPU:\nlength: %d\n", (int) sizeof(data));
    if (matrix) {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (data[ny * i + j] >= 0) {
                    printf("  %f  ", data[ny * i + j]);
                } else if (data[ny * i + j] < 0) {
                    printf(" %f  ", data[ny * i + j]);
                }
            }
            printf("\n");
        }
    } else {
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (data[ny * i + j] >= 0) {
                    printf("j: %d, i: %d,    data[ny * i + j]:  %f\n", j, i, data[ny * i + j]);
                } else if (data[ny * i + j] < 0) {
                    printf("j: %d, i: %d,    data[ny * i + j]: %f\n", j, i, data[ny * i + j]);
                }
            }
        }
    }

}

__global__ void myppkernel(const float* result, float* data, int nx, int ny, int nn) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    float* transpose = data + nn * nn;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < ny && j < ny) ? result[ny * i + j] : 0; /*ny * i + j*/
        //float v = result[ny * i + j];
        data[nn * i + j] = v;
        transpose[nn * j + i] = v;
    }
}

__global__ void mykernel(float* result, const float* data, int nx, int ny, int nn) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    const float* t = data + nn * nn;

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = 0;
        }
    }
    for (int k = 0; k < ny; ++k) { /*k < nx*/
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t[nn * k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = data[nn * k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] += x[ib] * y[jb];
            }
        }
    }
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) {
                result[i + j * ny] = v[ib][jb];
            }
        }
    }
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

    std::vector<float> padded(ny * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < ny; i++) {
            result[i + j * nx] = 0;
            float v = (i < nx) ? normalized[i + j * nx] : 0;
            padded[i + j * nx] = v;
        }
    }

    // calculate the (upper triangle of the) matrix product result = normalized * normalized^T
    int nn = roundup(ny, 64);

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(rGPU, padded.data(), ny * ny * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        myppkernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny, nn);
        CHECK(cudaGetLastError());
    }

    // Run dGPU test
    {
        dim3 dimBlock(1, 1);
        dim3 dimGrid(1, 1);
        dGPUtesting<<<dimGrid, dimBlock>>>(dGPU, nx, ny, nn, true);
        CHECK(cudaGetLastError());
    }
    // Run rGPU test
    {
        dim3 dimBlock(1, 1);
        dim3 dimGrid(1, 1);
        rGPUtesting<<<dimGrid, dimBlock>>>(rGPU, nx, ny, true);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, nx, ny, nn);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}
