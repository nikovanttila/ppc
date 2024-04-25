/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
using namespace std;
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

  // go through all y pixels
  #pragma omp parallel for
  for (int y = 0; y < ny; y++) {
    std::vector<float> window((2 * hx + 1) * (2 * hy + 1));
    int i_min, j_min, i_max, j_max, length, k, z;
    float value_1, value_2;

    // initialize limits of sliding window - dimension y
    if (y - hy < 0) j_min = 0; else j_min = y - hy;
    if (y + hy >= ny) j_max = ny; else j_max = y + hy + 1;

      // go through all y pixels
      for (int x = 0; x < nx; x++) {

        // initialize limits of sliding window - dimension x
        if (x - hx < 0) i_min = 0; else i_min = x - hx;
        if (x + hx >= nx) i_max = nx; else i_max = x + hx + 1;
        length = (i_max - i_min) * (j_max - j_min);

        // go through pixels in sliding window
        z = 0;
        for (int j = j_min; j < j_max; j++) {
          for (int i = i_min; i < i_max; i++) {
            window[z] = in[i + j * nx];
            z += 1;
          }
        }

        // find the median from the sliding window
        k = length / 2;
        if (length % 2 == 0) { // when size is even number
          nth_element(window.begin(), window.begin() + k - 1, window.begin() + length);
          value_1 =  window[k - 1];
          nth_element(window.begin(), window.begin() + k, window.begin() + length);
          value_2 =  window[k];
          out[x + y * nx] = (value_1 + value_2) / 2;
        } else { // when size is odd number
          nth_element(window.begin(), window.begin() + k, window.begin() + length);
          out[x + y * nx] = window[k];
        }
      }
  }

}