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
#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

static inline void check(cl_int err, const char* context) {
    if (err != CL_SUCCESS) {
        std::cerr << "OpenCL error: " << context << ": " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline void check_build(cl_program program, cl_device_id device, cl_int err) {
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        std::cerr << "OpenCL build failed:" << std::endl;
        size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        char* log = new char[len];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, NULL);
        std::cout << log << std::endl;
        delete[] log;
        std::exit(EXIT_FAILURE);
    } else if (err != CL_SUCCESS) {
        std::cerr << "OpenCL build failed: " << err << std::endl;
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

const char* kernel_source =
R""(
__kernel void mykernel(__global float* r, __global const float* d, int nx, int ny) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i >= nx || j >= ny)
        return;
    float v = HUGE_VALF;
    float product = 0;
    for (int k = 0; k < nx; ++k) {
        product += data[k + j * nx] * data[k + i * nx];
    }
    result[i + j * ny] = product;
}
)"";

void correlate(int ny, int nx, const float *data, float *result) {

    // create and initialize input vector
    std::vector<float> normalized(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = data[i + j * nx];
        }
    }

    // normalize input vector with each row having arithmetic mean of 0
    for (int j = 0; j < ny; j++) {
        // calculate the sum of current row
        float sum_row = 0;
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
        // calculate the sum of squares in the row
        float sum_square = 0;
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

    // Setup device
    cl_int err;
    cl_platform_id platform;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    cl_device_id device;
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check(err, "clCreateContext");
#ifdef CL_VERSION_2_0
    cl_command_queue queue
        = clCreateCommandQueueWithProperties(context, device, NULL, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    check(err, "clCreateCommandQueue");

    // Compile kernel
    cl_program program
        = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    check(err, "clCreateProgramWithSource");
    check_build(program, device, clBuildProgram(program, 1, &device, NULL, NULL, NULL));
    cl_kernel kernel = clCreateKernel(program, "mykernel", &err);
    check(err, "clCreateKernel");

    // Allocate memory & copy data to GPU
    cl_mem dGPU = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                 nx * ny * sizeof(float), (void*)data, &err);
    check(err, "clCreateBuffer");
    cl_mem rGPU = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 nx * ny * sizeof(float), NULL, &err);
    check(err, "clCreateBuffer");

    // Run kernel
    size_t wlsize[2] = {16, 16};
    size_t wgsize[2] = {size_t(roundup(nx, wlsize[0])), size_t(roundup(ny, wlsize[1]))};
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&rGPU));
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dGPU));
    CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void*)&nx));
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 2, NULL, wgsize, wlsize, 0, NULL, NULL));
    CHECK(clFinish(queue));

    // Copy data back to CPU & release memory
    CHECK(clEnqueueReadBuffer(queue, rGPU, true, 0,
                              nx * ny * sizeof(float), result, 0, NULL, NULL));
    CHECK(clReleaseMemObject(rGPU));
    CHECK(clReleaseMemObject(dGPU));

    // Release everything else
    CHECK(clReleaseKernel(kernel));
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));
}
