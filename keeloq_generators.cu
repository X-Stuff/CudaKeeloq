#define CU_FILE

#include "keeloq_generators.cuh"

#include "kernels.cuh"

#include "host/types/bruteforce_filters.h"

#include <algorithm>


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
		Call_CUDA_keeloq_generate_filtered(ThreadBlocks, ThreadsInBlock, mainInputs.ptr(), generator_results.ptr());
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
	BruteforceFilters::Test::Inputs test_cases[] = {
		{ 0x1111334404bbccee, BruteforceFilters::Flags::Max6ZerosInARow, true },
		{ 0x11113344aabbccee, BruteforceFilters::Flags::Max6ZerosInARow, false },
		{ 0x11113344aabbccee, BruteforceFilters::Flags::Max6OnesInARow, false },
		{ 0x11113344aFFbccee, BruteforceFilters::Flags::Max6OnesInARow, true },

		{ 0x3132333435363738, BruteforceFilters::Flags::AsciiNumbers, true },
		{ 0x3132333435363738, BruteforceFilters::Flags::AsciiAlphaNum, true },
		{ 0x3132333435363738, BruteforceFilters::Flags::AsciiAny, true },
		{ 0x2931323334353637, BruteforceFilters::Flags::AsciiNumbers, false },

		{ 0x4142434465666768, BruteforceFilters::Flags::AsciiAlpha,     true },
		{ 0x4142434465666768, BruteforceFilters::Flags::AsciiAlphaNum,  true },
		{ 0x4142434465666768, BruteforceFilters::Flags::AsciiAny, true },
		{ 0x3142434465666768, BruteforceFilters::Flags::AsciiAlpha,     false },
		{ 0x2142434465666768, BruteforceFilters::Flags::AsciiAlphaNum,  false },
		{ 0x1142434465666768, BruteforceFilters::Flags::AsciiAny, false },

		{ 0x214023245e28297e, BruteforceFilters::Flags::AsciiSpecial, true },
		{ 0x214023245e28297e, BruteforceFilters::Flags::AsciiAny, true },
		{ 0x114023245e28297e, BruteforceFilters::Flags::AsciiSpecial, false },

		{ 0x0022222222556677, BruteforceFilters::Flags::BytesRepeat4, true },
		{ 0x0022222222226677, BruteforceFilters::Flags::BytesRepeat4, true },
		{ 0x00Abcdef11111111, BruteforceFilters::Flags::BytesRepeat4, true },
		{ 0x0011222222556677, BruteforceFilters::Flags::BytesRepeat4, false },
		{ 0x0011223344556677, BruteforceFilters::Flags::BytesRepeat4, false },

		{ 0x112233445566aa00, BruteforceFilters::Flags::BytesIncremental, true },
		{ 0xFFEEDDCCBBAA1234, BruteforceFilters::Flags::BytesIncremental, true },
		{ 0x1122334455778899, BruteforceFilters::Flags::BytesIncremental, false },

		{ 0xCEB6AE48B5C63ED2, BruteforceFilters::Flags::BytesIncremental | BruteforceFilters::Flags::BytesRepeat4, false },
		{ 0xceb6ae48b5c03aba, BruteforceFilters::Flags::BytesIncremental | BruteforceFilters::Flags::BytesRepeat4, false },
	};

	constexpr uint8_t NumTests = sizeof(test_cases) / sizeof(BruteforceFilters::Test::Inputs);
	bool result_success = true;

	// CPU tests
	for (int i = 0; i < NumTests; ++i)
	{
		bool value = BruteforceFilters::check_filters(test_cases[i].value, test_cases[i].flags);
		result_success &= value == test_cases[i].result;

		assert(result_success);
	}

	// GPU tests
	DOUBLE_ARRAY<BruteforceFilters::Test::Inputs> test_inputs(test_cases, NumTests);
	Call_Kernel_RunFiltersTests(test_inputs.CUDA_mem, NumTests);
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
		BruteforceFilters {
			BruteforceFilters::Flags::All,     // SmartFilterFlags::AsciiAny;       //
			BruteforceFilters::Flags::BytesIncremental | BruteforceFilters::Flags::BytesRepeat4,    // SmartFilterFlags::BytesRepeat4;   //
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
