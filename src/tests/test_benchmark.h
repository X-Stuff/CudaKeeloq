#pragma once

#include "common.h"

#include "host/command_line_args.h"

#include <vector>

struct BruteforceConfig;

namespace benchmark
{
    void run(const CommandLineArgs& args, const BruteforceConfig& benchmarkConfig, const std::vector<uint16_t>& CudaBlocks, const std::vector<uint16_t>& CudaThreads);

    void all(const CommandLineArgs& args);
}