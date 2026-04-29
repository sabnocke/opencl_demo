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

float3 rgb_to_hsl(float3 rgb) {
    float r = rgb.x;
    float g = rgb.y;
    float b = rgb.z;

    float max_val = max(r, max(g, b));
    float min_val = min(r, max(g, b));
    float delta = max_val - min_val;

    float h = 0.0f;
    float s = 0.0f;
    float l = (max_val + min_val) / 2.0f;

    if (delta > 0.0f) {
        s = l < 0.5f ? delta / (max_val + min_val) : delta / (2.0f - max_val - min_val);

        if (max_val == r)
            h = (g - b) / delta + (g < b ? 6.0f : 0.0f);
        else if (max_val == g)
            h = (b - r) / delta + 2.0f;
        else
            h = (r - g) / delta + 4.0f;

        h /= 6.0f;
    }
    return (float3)(h, s, l);
}

float hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1/6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1/2.0f) return q;
    if (t < 2/3.0f) return p + (q - p) * (2/3.0f - t) * 6.0f;
    return p;
}

float3 hsl_to_rgb(float3 hsl) {
    float h = hsl.x;
    float s = hsl.y;
    float l = hsl.z;
    float3 rgb;

    if (s == 0.0f) {
        rgb = (float3)(l, l, l);
    } else {
        float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        rgb.x = hue_to_rgb(p, q, h + 1/3.0f);
        rgb.y = hue_to_rgb(p, q, h);
        rgb.z = hue_to_rgb(p, q, h - 1/3.0f);
    }
    return rgb;
}

kernel void shift_hue(global uchar4* data, float hue_shift) {
    int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
    uchar4 pixel = data[id];

    float3 rgb = convert_float3(pixel.xyz) / 255.f;
    float3 hsl = rgb_to_hsl(rgb);
    hsl.x += hue_shift;
    if (hsl.x > 1.0f) hsl.x -= 1.f;
    if (hsl.x > 0.1f && hsl.x < 0.2f) hsl.y *= 2.0f;

    float3 final_rgb = hsl_to_rgb(hsl);
    data[id] = (uchar4)(convert_uchar3_sat(final_rgb * 255.f), 255);
}