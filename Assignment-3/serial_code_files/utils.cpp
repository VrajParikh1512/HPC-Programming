void interpolation(double *mesh_value, Points *points)
{
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_area = inv_dx * inv_dy;   // 1 / (dx * dy)

    for (int p = 0; p < NUM_Points; p++)
    {
        double x = points[p].x;
        double y = points[p].y;

        // Cell index (fast)
        int i = (int)(x * inv_dx);
        int j = (int)(y * inv_dy);

        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        // Lower-left node
        double Xi = i * dx;
        double Yj = j * dy;

        // Local distances
        double lx = x - Xi;
        double ly = y - Yj;

        double dx_minus_lx = dx - lx;
        double dy_minus_ly = dy - ly;

        // Normalized bilinear weights
        double w00 = dx_minus_lx * dy_minus_ly * inv_area;
        double w10 = lx          * dy_minus_ly * inv_area;
        double w01 = dx_minus_lx * ly          * inv_area;
        double w11 = lx          * ly          * inv_area;

        int base = j * GRID_X + i;

        mesh_value[base]                 += w00;
        mesh_value[base + 1]             += w10;
        mesh_value[base + GRID_X]        += w01;
        mesh_value[base + GRID_X + 1]    += w11;
    }
}