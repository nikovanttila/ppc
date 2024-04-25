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
#include <iostream>
using namespace std;

void correlate(int ny, int nx, const float *data, float *result) {

    constexpr int nb = 4;
    int na = (nx + nb - 1) / nb;
    int nab = na*nb;

    // create and initialize input vector
    std::vector<double> input(nx * ny);
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            input[i + j * nx] = (double) data[i + j * nx];
        }
    }


    // normalize input vector, arithmetic mean of 0
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        double sum_row = 0;
        double mean = 0;
        for (int i = 0; i < nx; i++) {
            sum_row += input[i + j * nx];
        }
        mean = sum_row / nx;
        for (int i = 0; i < nx; i++) {
            input[i + j * nx] -=  mean;
        }
        sum_row = 0;
    }

    // normalize input vector, sum of the squares of the elements is 1
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        double sum_square = 0;
        double factor = 0;
        for (int i = 0; i < nx; i++) {
            sum_square += input[i + j * nx] * input[i + j * nx];
        }
        factor = sqrt(1 / sum_square);
        for (int i = 0; i < nx; i++) {
            input[i + j * nx] = input[i + j * nx] * factor;
        }
    }

    // create and initialize normalized padded vector
    std::vector<double> normalized(ny * nab, 0);
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[nab * j + i] = input[i + j * nx];
        }
    }

    // dot produzt between rows
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = j; i < ny; i++) {
            double vv[nb];
            for (int kb = 0; kb < nb; ++kb) {
                vv[kb] = 0;
            }
            for (int ka = 0; ka < na; ++ka) {
                for (int kb = 0; kb < nb; ++kb) {
                    double x = normalized[nab * j + ka * nb + kb];
                    double y = normalized[nab * i + ka * nb + kb];
                    double product = x * y;
                    vv[kb] += product;
                }
            }
            double v = 0;
            for (int kb = 0; kb < nb; ++kb) {
                v += vv[kb];
            }
            result[i + j * ny] = v;
        }
    }
}