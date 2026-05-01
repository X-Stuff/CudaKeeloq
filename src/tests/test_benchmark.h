#pragma once

#include "common.h"

#include "host/command_line_args.h"

#include <vector>

struct BruteforceConfig;

namespace benchmark
{
    bool run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig, uint16_t CudaBlocks, uint16_t CudaThreads);

    void run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig);

    void real();

    void all(const CommandLineArgs& args);
}