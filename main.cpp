#include <CL/opencl.hpp>
import GpuCompute;
#include <print>
#include <format>
#include <stb_image.h>
#include <stb_image_write.h>

using std::println;

int main() {
    int w, h, c;
    unsigned char* pixels = stbi_load("./input.png", &w, &h, &c, 4);
    if (!pixels) return -1;


    auto gpu = Gpu::GpuState("kernels/image_processing.cl");
    // println("source:\n{}", gpu.source);

    size_t img_size = w * h * 4;
    println("Image size: {}", img_size);
    cl::Buffer buffer(gpu.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, img_size, pixels);

    // cl::Kernel kernel(gpu.program, "grayscale");
    // cl::Kernel red_kernel(gpu.program, "nuclear_red");
    cl::Kernel hue_kernel(gpu.program, "shift_hue");
    /*red_kernel.setArg(0, buffer);
    kernel.setArg(0, buffer);*/

    hue_kernel.setArg(0, buffer);
    hue_kernel.setArg(1, 0.2f);
    println("Kernel name: {}", hue_kernel.getInfo<CL_KERNEL_FUNCTION_NAME>());
    println("Width: {}, Height: {}", w, h);
    if (const auto result = gpu.queue.enqueueNDRangeKernel(
        hue_kernel,
        cl::NullRange,
        cl::NDRange(w * h),
        cl::NullRange
        ); result != CL_SUCCESS) {
        std::println("Failed to enqueue kernel!");
        return result;
    }
    gpu.queue.enqueueReadBuffer(buffer, CL_TRUE, 0, img_size, pixels);
    stbi_write_png("output.png", w, h, 4, pixels, w * 4);

    std::println("Image processed successfully!");
    stbi_image_free(pixels);

    return 0;
}
