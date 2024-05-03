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
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr double4_t f8zero {
    0, 0, 0, 0
};

static inline double hsum4(double4_t vv) {
    double v = 0;
    for (int i = 0; i < 4; ++i) {
        v += vv[i];
    }
    return v;
}

static inline double4_t sum4(double4_t a, double4_t b) {
    return a + b;
}

void correlate(int ny, int nx, const float *data, float *result) {

    // initialize values for padded matrix
    constexpr int nb = 4;
    // vectors per input row
    int na = (nx + nb - 1) / nb;

    // create and initialize input vector
    std::vector<double> normalized(nx * ny);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = (double) data[i + j * nx];
        }
    }

    // normalize input vector with each row having arithmetic mean of 0
    double sum_row = 0;
    double mean_row = 0;
    for (int j = 0; j < ny; j++) {
        // calculate the sum of current row
        for (int i = 0; i < nx; i++) {
            sum_row += normalized[i + j * nx];
        }
        // calculate the mean of current row 
        mean_row = sum_row / nx;
        // substract the mean of row from row columns
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] -= mean_row;
        }
        // reset sum value
        sum_row = 0;
    }

    // normalize the input vector with each row the sum of the squares of the elements being 1
    double sum_square = 0;
    double factor = 0;
    for (int j = 0; j < ny; j++) {
        // calculate the sum of squares in the row
        for (int i = 0; i < nx; i++) {
            sum_square += normalized[i + j * nx] * normalized[i + j * nx];
        }
        // calculate the factor that is used for multiplying each element
        factor = sqrt(1 / sum_square);
        // mulitply each element with the factor
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = normalized[i + j * nx] * factor;
        }
        // reset sum value
        sum_square = 0;
    }

    // create and initialize normalized padded vector
    std::vector<double4_t> padded(ny * na);
    for (int j = 0; j < ny; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int i = ka * nb + kb;
                padded[na * j + ka][kb] = i < nx ? normalized[i + j * nx] : 0;
            }
        }
    }

    // calculate the (upper triangle of the) matrix product result = padded * padded^T
    for (int j = 0; j < ny; j++) {
        // calculate the dot product between current rows j and i
        for (int i = j; i < ny; i++) {
            // initialize variable that stores the sum of results below
            //double vv = 0;
            double4_t vv = f8zero;
            // calculate the dot product between rows for (nab / nb) * 4 elements
            for (int ka = 0; ka < na; ++ka) {
                double4_t x = padded[na * i + ka];
                double4_t y = padded[na * j + ka];
                double4_t product = x * y;
                //vv += sum(product);
                vv = sum4(vv, product);
            }
            // insert the final value to the result
            result[i + j * ny] = (float) hsum4(vv);
        }
    }
}
