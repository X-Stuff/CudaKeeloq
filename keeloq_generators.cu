#include "keeloq_generators.cuh"

#include <algorithm>


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
__host__ __device__ inline SmartFilterFlags operator|(SmartFilterFlags a, SmartFilterFlags b)
{
	return static_cast<SmartFilterFlags>(static_cast<uint64_t>(a) | static_cast<uint64_t>(b));
}
__host__ __device__ inline bool all_flags(SmartFilterFlags test, SmartFilterFlags check)
{
	return (uint64_t)check == (static_cast<uint64_t>(test) & static_cast<uint64_t>(check));
}
__host__ __device__ inline bool any_flag(SmartFilterFlags test, SmartFilterFlags check)
{
	return (test & check);
}

__device__ inline uint64_t generator_request_block(uint64_t* from, uint16_t size)
{
	return atomicAdd(from, size);
}

__device__ inline bool check_can_use(uint64_t value, SmartFilterFlags include, SmartFilterFlags exclude)
{
	bool canUse = true;

	if (include != SmartFilterFlags::All && include != SmartFilterFlags::None)
	{
		// Include keys match patterns
		canUse &= check_filters(value, include);
	}

	if (exclude != SmartFilterFlags::None && exclude != SmartFilterFlags::All)
	{
		// Exlude keys  which match patterns
		canUse &= !check_filters(value, exclude);
	}

	return canUse;
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

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		Decryptor& decryptor = decryptors[decryptor_index];

		decryptor.man = start.man + decryptor_index;
	}
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

	BruteforceConfig::Filters& filters = input->generator.filters;


	// if we need to generate 24 keys, and we have 64 threads, 40 last should do nothing
	uint8_t at_least_one = num_decryptors >= ctx.thread_id;

	// if we have to generate 75 keys with 64 threads, 11 threads should do +1 key generation
	size_t non_aligned = num_decryptors % ctx.thread_max;
	uint8_t additional_this_thread = non_aligned > 0 && non_aligned > ctx.thread_id;

	// decremental value how many keys should be generated by this thread
	uint32_t num_to_generate = at_least_one * (num_decryptors / ctx.thread_max + additional_this_thread);

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
				man_block_begin = generator_request_block(&input->generator.next.man, block_size);

				// TODO: Check how overflow behaves
				man_block_end = man_block_begin + block_size;
			}

			for (uint64_t key = man_block_begin; key < man_block_end; ++key)
			{
				if (check_can_use(key, filters.include, filters.exclude))
				{
					--num_to_generate;

					// next write should start testing keys from next key from current
					man_block_begin = key + 1;

					// break while loop
					written = true;

					Decryptor& decryptor = decryptors[write_index];
					decryptor.man = key;
					decryptor.seed = 0; // right now we don't do it

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

__global__ void CUDA_keeloq_generate_alphabet(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == BruteforceConfig::Type::Alphabet);

	BruteforceConfig::Alphabet& alphabet = input->generator.alphabet;
	assert(alphabet.num > 0);

	Decryptor& start = input->generator.start;

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	// Imagine alphabet as rotating rings with letters on it
	// we have 8-bytes key so there will be 8 rings
	// indexes are per-byte and shows how much ring is rotated
	// and what 'letter' it should have.
	// Or also it can be cosidered as 8-digit N-based number
	uint64_t start_indexer = alphabet.lookup(start.man);

	// decomposed uint64 indexer for inner loop
	uint8_t curr_indexer[8];

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		// Set curr indexer to initial value first

		*(uint64_t*)curr_indexer = start_indexer;

		// Get new indexer value (rotate rings) depending on what decryptor we now producing (basically add 10-base number to N-base number)
		alphabet.add(curr_indexer, decryptor_index);

		Decryptor& current = decryptors[decryptor_index];
		uint8_t* pCurrentKey = (uint8_t*)& current.man;

		// produce key by indexes from curr_indexer
		#pragma unroll
		for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
		{
			pCurrentKey[i] = alphabet[curr_indexer[i]];
		}
	}

}

__global__ void CUDA_generators_filters_test(FiltersTestinput* tests, uint8_t num)
{
	for (int i = 0; i < num; ++i)
	{
		bool value = check_filters(tests[i].value, tests[i].flags);
		assert(value == tests[i].result);

		tests[i].value = value == tests[i].result;
	}
}


int CUDA_generator_wrapper(KernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock)
{
	KernelResult generator_results;

	switch (mainInputs.generator.type)
	{
	case BruteforceConfig::Type::Simple:
		CUDA_keeloq_generate_brute<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), generator_results.ptr());
		break;
	case BruteforceConfig::Type::Filtered:
		mainInputs.generator.next = mainInputs.generator.start;
		CUDA_keeloq_generate_filtered<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), generator_results.ptr());
		break;
	case BruteforceConfig::Type::Alphabet:
		CUDA_keeloq_generate_alphabet<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), generator_results.ptr());
		break;
	case BruteforceConfig::Type::Pattern:
		assert(false && "Not implemented");
		return 1;
		break;

	case BruteforceConfig::Type::Dictionary:
	default:
		return 0;
	}

	mainInputs.read();          // it will not cause underneath arrays copy
	generator_results.read();


	// last generated decryptor - is first on next batch
	//  Warning: In case of non-aligned calculations "real" last decryptor may be somewhere in the middle of array
	mainInputs.generator.next = mainInputs.decryptors->host_last().man;

	return generator_results.error;
}


namespace
{
inline int CUDA_test_generator_filters()
{
	FiltersTestinput test_cases[] = {
		{ 0x1111334404bbccee, SmartFilterFlags::Max6ZerosInARow, true },
		{ 0x11113344aabbccee, SmartFilterFlags::Max6ZerosInARow, false },
		{ 0x11113344aabbccee, SmartFilterFlags::Max6OnesInARow, false },
		{ 0x11113344aFFbccee, SmartFilterFlags::Max6OnesInARow, true },

		{ 0x3132333435363738, SmartFilterFlags::AsciiNumbers, true },
		{ 0x3132333435363738, SmartFilterFlags::AsciiAlphaNum, true },
		{ 0x3132333435363738, SmartFilterFlags::AsciiAny, true },
		{ 0x2931323334353637, SmartFilterFlags::AsciiNumbers, false },

		{ 0x4142434465666768, SmartFilterFlags::AsciiAlpha,     true },
		{ 0x4142434465666768, SmartFilterFlags::AsciiAlphaNum,  true },
		{ 0x4142434465666768, SmartFilterFlags::AsciiAny, true },
		{ 0x3142434465666768, SmartFilterFlags::AsciiAlpha,     false },
		{ 0x2142434465666768, SmartFilterFlags::AsciiAlphaNum,  false },
		{ 0x1142434465666768, SmartFilterFlags::AsciiAny, false },

		{ 0x214023245e28297e, SmartFilterFlags::AsciiSpecial, true },
		{ 0x214023245e28297e, SmartFilterFlags::AsciiAny, true },
		{ 0x114023245e28297e, SmartFilterFlags::AsciiSpecial, false },

		{ 0x0022222222556677, SmartFilterFlags::BytesRepeat4, true },
		{ 0x0022222222226677, SmartFilterFlags::BytesRepeat4, true },
		{ 0x00Abcdef11111111, SmartFilterFlags::BytesRepeat4, true },
		{ 0x0011222222556677, SmartFilterFlags::BytesRepeat4, false },
		{ 0x0011223344556677, SmartFilterFlags::BytesRepeat4, false },

		{ 0x112233445566aa00, SmartFilterFlags::BytesIncremental, true },
		{ 0xFFEEDDCCBBAA1234, SmartFilterFlags::BytesIncremental, true },
		{ 0x1122334455778899, SmartFilterFlags::BytesIncremental, false },

		{ 0xCEB6AE48B5C63ED2, SmartFilterFlags::BytesIncremental | SmartFilterFlags::BytesRepeat4, false },
		{ 0xceb6ae48b5c03aba, SmartFilterFlags::BytesIncremental | SmartFilterFlags::BytesRepeat4, false },
	};

	constexpr uint8_t NumTests = sizeof(test_cases) / sizeof(FiltersTestinput);
	bool result_success = true;

	// CPU tests
	for (int i = 0; i < NumTests; ++i)
	{
		bool value = check_filters(test_cases[i].value, test_cases[i].flags);
		result_success &= value == test_cases[i].result;

		assert(result_success);
	}

	// GPU tests
	DOUBLE_ARRAY<FiltersTestinput> test_inputs(test_cases, NumTests);
	CUDA_generators_filters_test<<<1,1>>>(test_inputs.CUDA_mem, NumTests);
	test_inputs.read_GPU(); // for asserts
	for (int i = 0; i < NumTests; ++i)
	{
		result_success &= (bool)test_inputs.HOST_mem[i].value;
		assert(result_success);
	}

	// Filtered generator test itself
	constexpr auto NumBlocks = 32;// 1; //
	constexpr auto NumThreads = 512;// 2;// 512;

	constexpr auto NumToGenerate = 0xFFFFF;

	auto testConfig = BruteforceConfig::GetBruteforce(0xCEB6AE48B5C00000, NumToGenerate,
		BruteforceConfig::Filters {
			SmartFilterFlags::All,     // SmartFilterFlags::AsciiAny;       //
			SmartFilterFlags::BytesIncremental | SmartFilterFlags::BytesRepeat4,    // SmartFilterFlags::BytesRepeat4;   //
		});

	std::vector<Decryptor> decryptors(NumToGenerate);
	memset(decryptors.data(), 0, decryptors.size() * sizeof(Decryptor));
	KernelInput generatorInputs(nullptr, CUDA_Array<Decryptor>::allocate(decryptors), nullptr, testConfig);
	KernelResult result;

	auto error = CUDA_generator_wrapper(generatorInputs, NumBlocks,NumThreads);
	result_success &= error == 0;

	generatorInputs.decryptors->copy(decryptors);

	bool found = false;
	for (int i = 0; !found && i < decryptors.size(); ++i)
	{
		// looking for exact code - check nothing missed
		found |= decryptors[i].man == 0xCEB6AE48B5C63ED2;
	}

	assert(found);
	result_success &= found;

	result.read();
	return result_success;
}

inline int CUDA_test_generator_alphabet()
	{
		// Filtered generator test itself
		constexpr auto NumBlocks = 16;
		constexpr auto NumThreads = 32;


		auto testConfig = BruteforceConfig::GetAlphabet(0x6262626262626262, "abcd"_b, 0xFFFFFFFF);

		std::vector<Decryptor> decryptors(NumBlocks * NumThreads);
		KernelInput generatorInputs(nullptr, CUDA_Array<Decryptor>::allocate(decryptors), nullptr, testConfig);
		KernelResult result;

		CUDA_keeloq_generate_alphabet<<<NumBlocks,NumThreads>>>(generatorInputs.ptr(), result.ptr());

		generatorInputs.read();
		generatorInputs.decryptors->copy(decryptors);

		result.read();

		return 0;
	}
}

void generators::tests::run()
{
	CUDA_test_generator_filters();

	CUDA_test_generator_alphabet();
}