#pragma once

#include "common.h"

#include "keeloq_types.cuh"

void Call_Kernel_RunFiltersTests(BruteforceFilters::Test::Inputs* tests, uint8_t num);

void Call_CUDA_keeloq_generate_filtered(uint32_t blocks, uint32_t threads, KernelInput::TCudaPtr input, KernelResult::TCudaPtr results);
