#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_kernel.inl"

#include "kernels/kernel_input_base.h"

/**
 *  Definition of IndicesCache in CUDA
 */
namespace KeeloqLearning
{
__constant__ CudaFixedArray<uint8_t, IndicesCacheSize> IndicesCache = DecryptedResults::ResultIndicesCache::values;
}

// Constant per-run input data (captured encoded)
__constant__ CudaFixedArray<EncParcel, 3> InputsCache;