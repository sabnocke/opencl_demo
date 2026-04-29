//
// Created by ReWyn on 29.04.2026.
//

module;
#define CL_HPP_TARGET_OPENCL_VERSION 300 // Targets OpenCL 1.2
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

export module GpuCompute;

export namespace Gpu {
    using std::string;
    using std::vector;
    using namespace cl;

    class DeviceInfo {
    public:
        string name;
        int compute_units;
    };

    class GpuState {
    public:
        Context context;
        CommandQueue queue;
        Program program;
        string source;

        explicit GpuState(const string& kernelPath) {
            vector<Platform> platforms;
            Platform::get(&platforms);
            vector<Device> devices;
            platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

            context = devices[0];
            queue = {context, devices[0]};
            source = load_source(kernelPath);
            program = {context, source};

            program.build("-cl-std=CL1.2");
        }

        static string load_source(const string& path) {
            const std::ifstream file(path);
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
    };



    DeviceInfo getPrimaryDeviceInfo() {
        vector<Platform> platforms;
        Platform::get(&platforms);

        if (platforms.empty()) return {"No Platform", 0};

        vector<Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        if (devices.empty()) return {"No Device", 0};

        return {
            devices[0].getInfo<CL_DEVICE_NAME>(),
            static_cast<int>(devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>())
        };
    }
}