#include "common.h"

#include "stdio.h"

#include "host/console.h"

#include "algorithm/keeloq/keeloq_kernel.h"

#include "bruteforce/bruteforce_round.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "tests/test_all.h"


void bruteforce(const CommandLineArgs& args)
{
	if (args.selected_learning.size() == 0)
	{
		printf("Bruteforcing without specific learning type (slower)"
			"(1 KKey/s == %u Kkc (keeloq calcs) per second)\n", KeeloqLearningType::LAST);
	}

	for (const auto& config : args.brute_configs)
	{
		BruteforceRound attackRound(args.inputs, config, args.selected_learning, args.cuda_blocks, args.cuda_threads, args.cuda_loops);

		printf("\nallocating...");
		attackRound.Init();

		printf("\rRunning...\t\t\t\n%s\n", attackRound.ToString().c_str());

		bool match = false;

		size_t batchesInRound = attackRound.NumBatches();
		size_t keysInBatch = attackRound.KeysCheckedInBatch();

		auto roundStartTime = std::chrono::system_clock::now();

		for (size_t batch = 0; !match && batch < batchesInRound; ++batch)
		{
			auto batchStartTime = std::chrono::high_resolution_clock::now();

			KeeloqKernelInput& kernelInput = attackRound.Inputs();

			if (attackRound.Type() == BruteforceType::Dictionary)
			{
				// Write next batch of keys from dictionary
				kernelInput.WriteDecryptors(config.decryptors, batch * keysInBatch, keysInBatch);
			}
			else
			{
				// Make previous last generated key be an initial for current generation batch
				kernelInput.NextDecryptor();
			}

			// Generate decryptors (if available)
			int error = GeneratorBruteforce::PrepareDecryptors(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
			assert(error == 0);

			// do the bruteforce
			auto kernelResults = LaunchKeeloqBruteMain(kernelInput, attackRound.CudaBlocks(), attackRound.CudaThreads());
			match = attackRound.CheckResults(kernelResults);

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() - batchStartTime);

			if (batch == 0 || match)
			{
				console_hide_cursor();
				printf("\n\n\n");
			}

			if (!match)
			{
				auto kilo_result_per_second = duration.count() == 0 ? 0 : keysInBatch / duration.count();
				auto progress_percent = (double)(batch + 1) / batchesInRound;

				console_cursor_ret_up(2);

				printf("[%c][%zd/%zd]\t %llu(ms)/batch Speed: %llu KKeys/s\tNext key:0x%llX (%ul)\n", WAIT_CHAR(batch),
					batch, batchesInRound, duration.count(),
					kilo_result_per_second,
					kernelInput.generator.next.man, kernelInput.generator.next.seed);

				auto overall = std::chrono::duration_cast<std::chrono::seconds>(
					std::chrono::system_clock::now() - roundStartTime);

				console::progress_bar(progress_percent, overall);
			}
		}

		if (!match)
		{
			printf("\n\nAfter: %zd batches no results was found. Keys checked:%zd\n\n",
				batchesInRound, batchesInRound * keysInBatch);
		}
		else if (args.match_stop)
		{
			break;
		}
	}
}

int main(int argc, const char** argv)
{
	assert(Tests::CheckCudaIsWorking());

	const char* commandline[] = {
		"tests",
		"--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
		"--" ARG_MODE"=2,0,1,3",
		//"--" ARG_LTYPE"=6",

		"--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,0xCEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3",

#if _DEBUG
		"--" ARG_BLOCKS"=512",
		"--" ARG_LOOPS"=2",
		"--" ARG_START"=0xCEB6AE48B0000000",
#else
		"--" ARG_BLOCKS"=1024",
		"--" ARG_LOOPS"=2",
		"--" ARG_START"=0xCEB6AE4800000000",
#endif
		"--" ARG_COUNT"=0xFFFFFFFF",

		// "--" ARG_IFILTER"=0x2", include filter let be all (otherwise will have big impact)
		"--" ARG_EFILTER"=96",  // BytesRepeat4 | BytesIncremental should increse performance(?)

		"--" ARG_ALPHABET"=examples/alphabet.bin,CE:B6:AE:48:B5:C6:3E:D2",//:AA:BB:CC:DD:EE:FF:00:11",

		"--" ARG_FMATCH"=0",

		"--" ARG_TEST"=1",
	};

	console_set_width(CONSOLE_WIDTH);

	auto args = console::parse_command_line(sizeof(commandline) / sizeof(char*), commandline); //console::parse_command_line(argc, argv);
	if (args.run_tests)
	{
		printf("\n...RUNNING TESTS...\n");
		console::tests::run();

		Tests::AlphabetGeneration();
		Tests::FiltersGeneration();

		printf("\n...TESTS FINISHED...\n");

		//return;
		console_clear();
	}

	if (!args.isValid())
	{
		return 1;
	}

	if (!CUDA_check_keeloq_works())
	{
		printf("Error: This device cannot compute keeloq right. Single encryption and decryption mismatch.");
		assert(false);
		return 1;
	}

	bruteforce(args);

	// this will free all memory as well
	cudaDeviceReset();
}