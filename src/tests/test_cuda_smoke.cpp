#include "doctest/doctest.h"

#include <cuda_runtime_api.h>

#include "common.h"
#include "device/cuda_common.h"

namespace tests
{
    // Defined in test_kernel.cu
    __host__ bool cuda_check_working();
    __host__ bool check_utils();
}

TEST_CASE("cuda: toolchain and driver produce expected kernel result")
{
    CHECK(tests::cuda_check_working());
}

TEST_CASE("cuda: misc:: byte/bit helpers match between host and device")
{
    CHECK(tests::check_utils());
}
