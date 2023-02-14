#include "keeloq_kernel.h"

#if _DEBUG
    #define STRICT_ANALYSIS 1
#else
    #define STRICT_ANALYSIS 0
#endif

#include "keeloq_kernel.inl"

__global__ void Kernel_keeloq_test(KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();

    if (ctx.thread_id == 0) {

        uint32_t pln_test = 0x11223344;
        uint64_t key_test = 0xDEADBEEF00226688;
        uint32_t enc_test = keeloq_common_encrypt(pln_test, key_test);
        if (pln_test != keeloq_common_decrypt(enc_test, key_test))
        {
            ret->error = 1;
        }
        else
        {
            ret->value = 1;
        }
    }
}

__global__ void Kernel_keeloq_main(KeeloqKernelInput::TCudaPtr CUDA_inputs, KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();

    auto& encoded_data = *CUDA_inputs->encdata;
    auto& decryptors = *CUDA_inputs->decryptors;
    auto& results = *CUDA_inputs->results;

    uint8_t num_errors = keeloq_decryption_run(ctx, *CUDA_inputs);
    uint8_t num_matches = keeloq_analyze_results(ctx, results, decryptors.num, encoded_data.num);

    atomicAdd(&ret->error,  num_errors);
    atomicAdd(&ret->value, num_matches);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock)
{
    KernelResult kernel_results;

    Kernel_keeloq_main<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());

    mainInputs.read();
    kernel_results.read();

    return kernel_results;
}

__host__ bool keeloq::kernels::cuda_is_working()
{
    KernelResult kernel_results;
    Kernel_keeloq_test<<<1, 1 >>>(kernel_results.ptr());
    kernel_results.read();

    return kernel_results.error == 0 && kernel_results.value != 0;
}