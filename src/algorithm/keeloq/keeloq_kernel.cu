#include "keeloq_kernel.h"

#include "keeloq_kernel.inl"
#include "keeloq_kernel_input.h"

/**
 *  Definition of IndicesCache in CUDA
 */
namespace KeeloqLearning
{
__constant__ CudaFixedArray<uint8_t, IndicesCacheSize> IndicesCache = DecryptedResults::ResultIndicesCache::values;
}

// Constant per-run input data (captured encoded)
__constant__ CudaFixedArray<EncParcel, 3> InputsCache;