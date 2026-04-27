#include "keeloq_kernel.h"

#include "keeloq_kernel.inl"

/**
 *  Definition of IndicesCache in CUDA
 */
namespace KeeloqLearning
{
__constant__ CudaFixedArray<uint8_t, IndicesCacheSize> IndicesCache = DecryptedResults::ResultIndicesCache::values;
}