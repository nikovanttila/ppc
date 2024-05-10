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
typedef double double8_t __attribute__ ((vector_size (8 * sizeof(double))));

constexpr double8_t d8zero {
    0, 0, 0, 0, 0, 0, 0, 0
};

static inline double hsum8(double8_t vv) {
    double v = 0;
    for (int i = 0; i < 8; ++i) {
        v += vv[i];
    }
    return v;
}

static inline double8_t sum8(double8_t a, double8_t b) {
    return a + b;
}

void correlate(int ny, int nx, const float *data, float *result) {

    // initialize values for padded matrix
    constexpr int nb = 8;
    // vectors per input row
    int na = (nx + nb - 1) / nb;

    // block size
    constexpr int nd = 3;
    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;

    // create and initialize input vector
    std::vector<double> normalized(nx * ny);
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = (double) data[i + j * nx];
        }
    }

    // normalize input vector with each row having arithmetic mean of 0
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        double sum_row = 0;
        // calculate the sum of current row
        for (int i = 0; i < nx; i++) {
            sum_row += normalized[i + j * nx];
        }
        // calculate the mean of current row 
        double mean_row = sum_row / nx;
        // substract the mean of row from row columns
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] -= mean_row;
        }
    }

    // normalize the input vector with each row the sum of the squares of the elements being 1
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        double sum_square = 0;
        // calculate the sum of squares in the row
        for (int i = 0; i < nx; i++) {
            sum_square += normalized[i + j * nx] * normalized[i + j * nx];
        }
        // calculate the factor that is used for multiplying each element
        double factor = sqrt(1 / sum_square);
        // mulitply each element with the factor
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = normalized[i + j * nx] * factor;
        }
    }

    // create and initialize normalized padded vector
    std::vector<double8_t> padded(ncd * na);
    #pragma omp parallel for
    for (int j = 0; j < ny; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                int i = ka * nb + kb;
                padded[na * j + ka][kb] = i < nx ? normalized[i + j * nx] : 0;
            }
        }
    }
    for (int j = ny; j < ncd; ++j) {
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; ++kb) {
                padded[na * j + ka][kb] = 0;
            }
        }
    }

    // calculate the (upper triangle of the) matrix product result = padded * padded^T
    #pragma omp parallel for schedule(static,1)
    for (int jc = 0; jc < nc; ++jc) {
        // calculate the dot product between current rows j and i
        for (int ic = jc; ic < nc; ++ic) {
            // initialize variable that stores the sum of results below
            double8_t vv[nd][nd];
            for (int jd = 0; jd < nd; ++jd) {
                for (int id = 0; id < nd; ++id) {
                    vv[id][jd] = d8zero;
                }
            }
            // calculate the dot product between rows for (nab / nb) * 8 elements
            for (int ka = 0; ka < na; ++ka) {
                double8_t y0 = padded[na*(jc * nd + 0) + ka];
                double8_t y1 = padded[na*(jc * nd + 1) + ka];
                double8_t y2 = padded[na*(jc * nd + 2) + ka];
                double8_t x0 = padded[na*(ic * nd + 0) + ka];
                double8_t x1 = padded[na*(ic * nd + 1) + ka];
                double8_t x2 = padded[na*(ic * nd + 2) + ka];
                vv[0][0] = sum8(vv[0][0], x0 * y0);
                vv[0][1] = sum8(vv[0][1], x0 * y1);
                vv[0][2] = sum8(vv[0][2], x0 * y2);
                vv[1][0] = sum8(vv[1][0], x1 * y0);
                vv[1][1] = sum8(vv[1][1], x1 * y1);
                vv[1][2] = sum8(vv[1][2], x1 * y2);
                vv[2][0] = sum8(vv[2][0], x2 * y0);
                vv[2][1] = sum8(vv[2][1], x2 * y1);
                vv[2][2] = sum8(vv[2][2], x2 * y2);
            }
            // insert the final value to the result
            for (int jd = 0; jd < nd; ++jd) {
                for (int id = 0; id < nd; ++id) {
                    int i = ic * nd + id;
                    int j = jc * nd + jd;
                    if (i < ny && j < ny) {
                        result[i + j * ny] = (float) hsum8(vv[id][jd]);
                    }
                }
            }
        }
    }
}
