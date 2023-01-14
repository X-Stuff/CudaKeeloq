#include "keeloq_generators.cuh"

namespace SmartFilter
{
	constexpr uint8_t KeySizeBytes = sizeof(uint64_t);

	constexpr uint8_t KeySizeBits = sizeof(uint64_t)  * 8;

	template<uint8_t value_min, uint8_t value_max>
	__host__ __device__ inline bool all_min_max(uint64_t key)
	{
		// for logical AND start should be with true
		bool result = true;
		uint8_t* bPtrKey    = (uint8_t*)&key;

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
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

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
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

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
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

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (uint8_t i = 0; i < KeySizeBits; ++i)
		{
			// inverse - filter pass if no consecutive bits
			result |= (key & mask) == mask;
			key = key >> 1;
		}

		return result;
	}

	template<uint8_t MaxCount = 4>
	__host__ __device__ inline bool has_consecutive_bytes(uint64_t key)
	{
		// for logical OR start should be with false
		bool result = false;

		uint8_t index = 0;
		uint8_t* bPtrKey = (uint8_t*)&key;

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (uint8_t i = 1; i < KeySizeBytes; ++i)
		{
			bool equal = bPtrKey[i] == bPtrKey[index];
			index = equal * index + (1 - equal) * i;

			result |= (i - index) >= (MaxCount - 1);
		}

		return result;
	}

	template<uint8_t MaxCount = 6>
	__host__ __device__ inline bool has_incremental_pattern(uint64_t key)
	{
		// for logical OR start should be with false
		bool result = false;

		uint8_t index = 0;
		uint8_t* bPtrKey = (uint8_t*)&key;

#ifdef __CUDA_ARCH__
		#pragma unroll
#endif
		for (uint8_t i = 1; i < KeySizeBytes; ++i)
		{
			uint8_t deltaIndex = (i - index);
#ifdef __CUDA_ARCH__
			uint8_t asbDeltaValue = __sad(bPtrKey[i], bPtrKey[index], 0);
#else
			uint8_t asbDeltaValue = abs(bPtrKey[i] - bPtrKey[index]);
#endif

			bool match = asbDeltaValue == (0x11 * deltaIndex);

			index = match * index + (1 - match) * i;

			result |= deltaIndex >= (MaxCount - 1);
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
	if (!key_has_any && all_flags(filter, SmartFilterFlags::AsciiAny))
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

	if (!key_has_any && any_flag(filter, SmartFilterFlags::BytesRepeat4))
	{
		key_has_any |= SmartFilter::has_consecutive_bytes(key);
	}

	if (!key_has_any && any_flag(filter, SmartFilterFlags::BytesIncremental))
	{
		key_has_any |= SmartFilter::has_incremental_pattern(key);
	}

	return key_has_any;
}

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == BruteforceConfig::Type::Simple);

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

__global__ void CUDA_keeloq_generate_filtered(KernelInput::TCudaPtr input, KernelResult::TCudaPtr results)
{
	assert(input->generator.type == BruteforceConfig::Type::Filtered);

	assert(input->generator.start.man > 0x100000000000 && "Starting key should be big enough to start bruteforcing. Consider pattern brute.");

	assert(input->generator.filters.include != SmartFilterFlags::None && "Include filter None is invalid - will lead to infinite loop!");
	assert(input->generator.filters.exclude != SmartFilterFlags::All && "Exclude filter All is invalid - will lead to infinite loop!");

	CUDACtx ctx = GET_CUDA_CONTEXT();

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;
	size_t num_decryptors = decryptors.num;


	// if we need to generate 24 keys, and we have 64 threads, 40 last should do nothinf
	uint8_t at_least_one = num_decryptors >= ctx.thread_id;

	// if we have to genrate 75 keys with 64 threads, 11 threads should do +1 key generation
	uint8_t additional_this_thread = (num_decryptors % ctx.thread_max) > ctx.thread_id;

	// decremental value how many keys should be generated by this thread
	uint32_t num_to_generate = at_least_one * (num_decryptors / ctx.thread_max + additional_this_thread);

	// constant value - shows how many keys will be generated. used for correct write memomry access. |-------x*********x-------|
	const uint32_t thread_generate = num_to_generate;

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
		Decryptor& next = input->generator.next;
		uint64_t man_begin = atomicAdd(&next.man, num_to_generate);

		for (uint32_t iteration = 0; iteration < num_to_generate; ++iteration)
		{
			uint64_t key = man_begin + iteration; // even if overflow - ok, but put in for loop - infinite hang

			bool canUse = true;

			if (input->generator.filters.include != SmartFilterFlags::All &&
				input->generator.filters.include != SmartFilterFlags::None)
			{
				// Include keys match patterns
				canUse &= check_filters(key, input->generator.filters.include);
			}


			if (input->generator.filters.exclude != SmartFilterFlags::None &&
				input->generator.filters.exclude != SmartFilterFlags::All)
			{
				// Exlude keys  which match patterns
				canUse &= !check_filters(key, input->generator.filters.exclude);
			}

			if (canUse)
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

__global__ void CUDA_keeloq_generate_alphabet(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == BruteforceConfig::Type::Alphabet);

	BruteforceConfig::Alphabet& alphabet = input->generator.alphabet;
	assert(alphabet.num > 0);

	Decryptor& start = input->generator.start;
	Decryptor& next = input->generator.next;

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	// Imagine alphabet as rotating rings with letters on it
	// we have 8-bytes key so there will be 8 rings
	// indexes are per-byte and shows how much ring is rotated
	// and what 'letter' it should have.
	// Or also it can be cosidered as 8-digit N-based number
	uint64_t start_indexer = 0;

	uint8_t* start_key = (uint8_t*)& start.man;
	uint8_t* pIndexer = (uint8_t*)&start_indexer;

	// Get start indexer (rings) values from last start man key (reverse lookup)
	#pragma unroll
	for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
	{
		// Valid or 0 (first letter in alphabet)
		pIndexer[i] = alphabet.is_valid_index(start_key[i]) * alphabet.lookup(start_key[i]);
	}

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		// Set curr indexer to initial value first
		uint8_t curr_indexer[8];
		*(uint64_t*)curr_indexer = start_indexer;

		// Get new indexer value (rotate rings) depending on what decryptor we now producing (basically add 10-base number to N-base number)
		alphabet.add(curr_indexer, decryptor_index);

		// produce key by indexes from curr_indexer
	}
}

__global__ void CUDA_generators_test(FiltersTestinput* tests, uint8_t num)
{
	for (int i = 0; i < num; ++i)
	{
		bool value = check_filters(tests[i].value, tests[i].flags);
		assert(value == tests[i].result);

		tests[i].value = 0;// just to check if kernel worked well
	}
}