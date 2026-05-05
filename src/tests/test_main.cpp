// doctest entry point for the CudaKeeloqTests binary.
//
// We use DOCTEST_CONFIG_IMPLEMENT (not _WITH_MAIN) so we control main():
// CUDA runtime must be initialised AFTER process start — doing it from a
// global/static initializer works on some platforms but breaks on WSL/Linux
// with cudaErrorInvalidResourceHandle because the driver isn't fully up
// before main() runs.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdio>
#include <cuda_runtime_api.h>

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"


int main(int argc, char** argv)
{
    if (!keeloq::kernels::cuda_is_working())
    {
        std::fprintf(stderr, "Fatal: this device cannot compute keeloq. Aborting tests.\n");
        return 2;
    }

    if (KeeloqLearning::DecryptedResults::cuda_init() != cudaSuccess)
    {
        std::fprintf(stderr, "Fatal: failed to init DecryptedResults cache on device.\n");
        return 2;
    }

    // Clear any lingering error state from the init probe so the first
    // TEST_CASE sees a clean cudaGetLastError().
    cudaGetLastError();

    doctest::Context ctx;

    // Print each test case as it runs (with timing) so the user can tell the
    // binary is making progress — doctest's default output is silent until
    // something fails, which looks like a hang on the long CUDA cases.
    // User-supplied flags override these via applyCommandLine().
    //ctx.setOption("duration", true);

    ctx.applyCommandLine(argc, argv);
    const int res = ctx.run();

    cudaDeviceReset();

    if (ctx.shouldExit())
    {
        return res;
    }
    return res;
}
