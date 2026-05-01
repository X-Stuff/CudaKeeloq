#pragma once

#include "common.h"

namespace tests
{

namespace generators
{
bool pattern();

/**
 *  Check if CUDA generator for seed bruteforce generates correct decryptors on GPU
 */
bool seed();

/**
 *  Run all test for generators
 */
bool all();

}

}