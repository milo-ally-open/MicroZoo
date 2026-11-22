#pragma once

#include <NvInfer.h>

#include <cstdio>

namespace demo {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TensorRT] %s\n", message);
        }
    }
};

}  // namespace demo
