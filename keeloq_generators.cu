#include "keeloq_generators.cuh"

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

}

__global__ void CUDA_keeloq_generate_smart(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();
}
