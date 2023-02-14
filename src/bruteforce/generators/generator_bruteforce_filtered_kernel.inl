#include "device/cuda_context.h"

#include <cuda_runtime.h>
#include <device_atomic_functions.h>

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_type.h"
#include "bruteforce/bruteforce_filters.h"

template<typename TPtr>
__device__ inline uint64_t RequestNewBlock(TPtr* from, uint32_t size)
{
	return atomicAdd((unsigned long long int*)from, size);
}

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceFiltered, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	assert(input->config.type == BruteforceType::Filtered);

	assert(input->config.start.man > 0x100000000000 && "Starting key should be big enough to start bruteforcing. Consider pattern brute.");

	assert(input->config.filters.include != BruteforceFilters::Flags::None && "Include filter None is invalid - will lead to infinite loop!");
	assert(input->config.filters.exclude != BruteforceFilters::Flags::All && "Exclude filter All is invalid - will lead to infinite loop!");

	CudaContext ctx = CudaContext::Get();

	CudaArray<Decryptor>& decryptors = *input->decryptors;
	size_t num_decryptors = decryptors.num;

	BruteforceFilters& filters = input->config.filters;


	// if we need to generate 24 keys, and we have 64 threads, 40 last should do nothing
	uint8_t at_least_one = num_decryptors >= ctx.thread_id;

	// if we have to generate 75 keys with 64 threads, 11 threads should do +1 key generation
	size_t non_aligned = num_decryptors % ctx.thread_max;
	uint8_t additional_this_thread = non_aligned > 0 && non_aligned > ctx.thread_id;

	// decremental value how many keys should be generated by this thread
	uint32_t num_to_generate = static_cast<uint32_t>(at_least_one * (num_decryptors / ctx.thread_max + additional_this_thread));

	uint32_t block_size = 0;

	// Block (from first to num_to_generate) of keys to check
	uint64_t man_block_begin = 0;
	uint64_t man_block_end = 0;

	CUDA_FOR_THREAD_ID(ctx, write_index, num_decryptors)
	{
		bool written = false;

		do
		{
			if (block_size == 0)
			{
				// We have to acquire next block not more than we have to generate left
				// It's need to prevent situations like:
				//  this thread need to generate 2 keys
				//  but it has requested 100 to check
				//  first to keys are valid and next 98 nobody will check, and thread cannot "return" it to unchecked pool
				//  since check indication is just an atomic add operation
				block_size = num_to_generate;

				// get raw manufacturer key start index (number) which will be incremented and checked
				// for each thread. atomic instruction guaranties no overlapping key checks between threads
				// For example:
				//   each thread should generate 16 keys
				//   `next.man` is 123 right now, thread 1 do atomic add and get
				//   start from 123, do 16 checks
				//   `next.man` now 139, but thread 1 runs checks from 123
				//   ...
				//   thread 1 found 4 keys, so it should generate 12 more
				//   `next.man` is 6383 on next iteration (other threads do jobs as well)
				//   thread 1 adds 12 to `next.man` and get previous value
				//   so now it starts check 12 keys from 6383 to 6395
				//
				man_block_begin = RequestNewBlock(&input->config.last.man, block_size);

				// TODO: Check how overflow behaves
				man_block_end = man_block_begin + block_size;
			}

			for (uint64_t key = man_block_begin; key < man_block_end; ++key)
			{
				if (filters.Pass(key))
				{
					--num_to_generate;

					// next write should start testing keys from next key from current
					man_block_begin = key + 1;

					// break while loop
					written = true;

					Decryptor& decryptor = decryptors[write_index];
					decryptor.man = key;
					decryptor.seed = num_to_generate; // right now we don't do it

					break;
				}
			}

			if (!written)
			{
				// request new block
				block_size = 0;
			}

		} while (!written);
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceFiltered);