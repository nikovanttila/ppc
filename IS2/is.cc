struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

#include <vector>
#include <math.h>

#include <iostream>
using namespace std;

Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};

    // outer loop of that tries different heights of rectangle
    for (int j = ny; j > 0; --j) {
        cout << "\nCurrent j: " << j << "\n";
        // outer loop of that tries different wirdths of rectangle
        cout << "Current i: ";
        for (int i = nx; i > 0; --i) {
            cout << i << " ";

            // outer loop of that tries different heights of rectangle
            for (int j = ny; j > 0; --j) {
                cout << "\nCurrent j: " << j << "\n";
                // outer loop of that tries different wirdths of rectangle
                cout << "Current i: ";
                for (int i = nx; i > 0; --i) {
                    cout << i << " ";

                }
            }

        }
    }

    return result;
}
