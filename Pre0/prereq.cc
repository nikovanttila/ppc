struct Result {
    float avg[3];
};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- horizontal position: 0 <= x0 < x1 <= nx
- vertical position: 0 <= y0 < y1 <= ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
- output: avg[c]
*/
Result calculate(int ny, int nx, const float *data, int y0, int x0, int y1, int x1) {
    Result result{{0.0f, 0.0f, 0.0f}};
    double r = 0;
    double g = 0;
    double b = 0;

    for (int x = x0; x < x1; x++) {
        for (int y = y0; y < y1; y++) {
            r += (double) data[0 + 3 * x + 3 * nx * y];
            g += (double) data[1 + 3 * x + 3 * nx * y];
            b += (double) data[2 + 3 * x + 3 * nx * y];
        }
    }

    double length = (x1 - x0) * (y1 - y0);
    double avg_r = r / length;
    double avg_g = g / length;
    double avg_b = b / length;

    result.avg[0] = (float) avg_r;
    result.avg[1] = (float) avg_g;
    result.avg[2] = (float) avg_b;

    return result;
}
