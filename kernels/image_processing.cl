#define clamp_int(val, a, b) clamp((int)(val), a, b)

__kernel void adjust_brightness(global uchar4* data, float factor) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);

    int id = (y * width + x) * 3; // 3 is for 3 channels of RGB
    uchar4 pixel = data[id];

    float4 f_pixel = convert_float4(pixel);
    f_pixel.xyz *= factor;

    data[id] = convert_uchar4_sat(f_pixel);

}

kernel void grayscale(global uchar4* data) {
    int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    uchar4 p = data[id];
    uchar avg = (p.x + p.y + p.z) / 3;
    data[id] = (uchar4)(avg, avg, avg, 255);
}

kernel void nuclear_red(global uchar4* data) {
    int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    data[id] = (uchar4)(255, 0, 0, 255);
}