#include "keeloq_generators.cuh"

namespace SmartFilter
{
	constexpr uint8_t KeySizeBytes = sizeof(uint64_t);

	template<uint8_t value_min, uint8_t value_max>
	__host__ __device__ inline bool all_min_max(uint64_t key)
	{
		// for logical AND start should be with true
		bool result = true;
		uint8_t* bPtrKey    = (uint8_t*)&key;

		#pragma unroll
		for (uint8_t i = 0; i < KeySizeBytes; ++i)
		{
			// TODO: Some vector instruction here
			result &= bPtrKey[i] >= value_min && bPtrKey[i] <= value_max;
		}

		return result;
	}

	// special case cause faster
	__host__ __device__ inline bool all_any_ascii(uint64_t key)
	{
		constexpr uint8_t value_min = '!';
		constexpr uint8_t value_max = '~';

		return all_min_max<value_min, value_max>(key);
	}

	__host__ __device__ inline bool all_ascii_num(uint64_t key)
	{
		constexpr uint8_t value_min = '0';
		constexpr uint8_t value_max = '9';

		return all_min_max<value_min, value_max>(key);
	}

	__host__ __device__ inline bool all_ascii_alpha(uint64_t key)
	{
		// for logical AND start should be with true
		bool result = true;
		uint8_t* bPtrKey    = (uint8_t*)&key;

		#pragma unroll
		for (uint8_t i = 0; i < KeySizeBytes; ++i)
		{
			result &= (bPtrKey[i] >= 'a' && bPtrKey[i] <= 'z') || (bPtrKey[i] >= 'A' && bPtrKey[i] <= 'Z');
		}

		return result;
	}

	__host__ __device__ inline bool all_ascii_symbol(uint64_t key)
	{
		// for logical AND start should be with true
		bool result = true;
		uint8_t* bPtrKey = (uint8_t*)&key;

		#pragma unroll
		for (uint8_t i = 0; i < KeySizeBytes; ++i)
		{
			result &=
				(bPtrKey[i] >= '!' && bPtrKey[i] <= '/') ||
				(bPtrKey[i] >= ':' && bPtrKey[i] <= '@') ||
				(bPtrKey[i] >= '[' && bPtrKey[i] <= '`') ||
				(bPtrKey[i] >= '{' && bPtrKey[i] <= '~');
		}

		return result;
	}


	template<uint8_t bit, uint8_t MaxCount = 6>
	__host__ __device__ inline bool has_consecutive_bits(uint64_t key)
	{
		uint8_t result = false;
		uint64_t mask = (1 << MaxCount) - 1;

		key = bit ? key : ~key;

		#pragma unroll
		for (uint8_t i = 0; i < sizeof(uint64_t) * 8; ++i)
		{
			// inverse - filter pass if no consecutive bits
			result |= (key & mask) == mask;
			key = key >> 1;
		}

		return result;
	}
}
namespace
{
__host__ __device__ inline bool operator&(SmartFilterFlags a, SmartFilterFlags b)
{
	return static_cast<bool>(static_cast<uint64_t>(a) & static_cast<uint64_t>(b));
}
__host__ __device__ inline bool all_flags(SmartFilterFlags test, SmartFilterFlags check)
{
	return (uint64_t)check == (static_cast<uint64_t>(test) & static_cast<uint64_t>(check));
}
__host__ __device__ inline bool any_flag(SmartFilterFlags test, SmartFilterFlags check)
{
	return (test & check);
}
}

__host__ __device__ bool check_filters(uint64_t key, SmartFilterFlags filter)
{
	bool key_has_any = false;

	// fastest should go first
	if (!key_has_any && all_flags(filter, SmartFilterFlags::AsciiAnySymbol))
	{
		key_has_any |= SmartFilter::all_any_ascii(key);
	}

	if (!key_has_any && any_flag(filter, SmartFilterFlags::AsciiNumbers))
	{
		key_has_any |= SmartFilter::all_ascii_num(key);
	}

	if (!key_has_any && any_flag(filter, SmartFilterFlags::AsciiAlpha))
	{
		key_has_any |= SmartFilter::all_ascii_alpha(key);
	}

	if (!key_has_any && any_flag(filter, SmartFilterFlags::AsciiSpecial))
	{
		key_has_any |= SmartFilter::all_ascii_symbol(key);
	}

	//
	if (!key_has_any && any_flag(filter, SmartFilterFlags::Max6OnesInARow))
	{
		key_has_any |= SmartFilter::has_consecutive_bits<1>(key);
	}

	if (!key_has_any && any_flag(filter, SmartFilterFlags::Max6ZerosInARow))
	{
		key_has_any |= SmartFilter::has_consecutive_bits<0>(key);
	}

	if (key_has_any && any_flag(filter, SmartFilterFlags::BytesIncremental))
	{
		assert(false && "Not implemented");
	}

	if (key_has_any && any_flag(filter, SmartFilterFlags::Same4Bytes))
	{
		assert(false && "Not implemented");
	}

	return key_has_any;
}

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == GeneratorType::Brute);

	Decryptor& start = input->generator.start;
	Decryptor& next = input->generator.next;

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		Decryptor& decryptor = decryptors[decryptor_index];

		decryptor.man = start.man + decryptor_index;
	}

	// every thread will do this. need to measure perf
	next.man = decryptors[decryptors.num - 1].man;
}

__global__ void CUDA_keeloq_generate_smart(KernelInput::TCudaPtr input, KernelResult::TCudaPtr results)
{
	assert(input->generator.type == GeneratorType::Smart);

	CUDACtx ctx = GET_CUDA_CONTEXT();

	size_t num_decryptors = input->decryptors->num;
	assert(num_decryptors % ctx.thread_max == 0 && "Number of decryptors is not aligned with number of threads!");

	// decremental value how many keys should be generated by this thread
	uint32_t num_to_generate = num_decryptors / ctx.thread_max + 1;

	// constant value - shows how many keys will be generated. used for correct write memomry access. |-------x*********x-------|
	const uint32_t thread_generate = num_to_generate;

	Decryptor& next = input->generator.next;
	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	while (num_to_generate > 0)
	{
		// get raw manufactory key start index (number) which will be incremented and checked
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
		uint64_t man_begin = atomicAdd(&next.man, num_to_generate);

		for (uint32_t iteration = 0; iteration < num_to_generate; ++iteration)
		{
			uint64_t key = man_begin + iteration; // even if overflow - ok, but put in for loop - infinite hang

			if (key % 2 == 0) // test if pass all smart filters
			{
				// we fill memory region from last to 0
				--num_to_generate;

				uint64_t write_index = ctx.thread_id * thread_generate + num_to_generate;

				Decryptor& decryptor = decryptors[write_index];
				decryptor.man = key;
				decryptor.seed = 0; // right now we don't do it
			}
		}
	}
}

__global__ void CUDA_generators_test(KernelResult::TCudaPtr ki)
{
	ki->value = check_filters(0x11003344aabbccee,
		SmartFilterFlags::Max6ZerosInARow);
}