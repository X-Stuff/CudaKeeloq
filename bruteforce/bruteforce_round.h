#pragma once

#include "common.h"

#include <vector>
#include <string>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_config.h"
#include "kernels/kernel_result.h"


/**
 *  Round is a set of bruteforce batches
 * Each batch runs N thread
 * Each thread checks 1 or more decryptor
 * Each check is 1 or more (configured in args) keeloq calculation
 *
 * Typical is:
 *  Via command line some rounds were created - e.g. Dictionary and Simple attacks
 *  Each attack has N decryptors to check, though num blocks B, num threads T, and I iteration
 *  This means `(1 BATCH size) = B * T * I` decryptors check
 *  This means num of batches = `N / (1 BATCH size)`
 */
struct BruteforceRound
{
	// Construct round struct without specific learning type (means use all learnings)
	BruteforceRound(const std::vector<EncData>& data, const BruteforceConfig& gen, uint32_t blocks, uint32_t threads, uint32_t iterations) :
		BruteforceRound(data, gen, {}, blocks, threads, iterations) {}

	// Construct round struct with only one selected learning type
	BruteforceRound(const std::vector<EncData>& data, const BruteforceConfig& gen, KeeloqLearningType::Type single_learning,
		uint32_t blocks, uint32_t threads, uint32_t iterations) :
		BruteforceRound(data, gen, std::vector<KeeloqLearningType::Type> { single_learning }, blocks, threads, iterations) {}

	// Standard constructor
	BruteforceRound(const std::vector<EncData>& data, const BruteforceConfig& gen, std::vector<KeeloqLearningType::Type> selected_learning,
		uint32_t blocks, uint32_t threads, uint32_t iterations);

	~BruteforceRound()
	{
		free();
	}

public:
	// Allocates memory
	void Init();

	// Reads results data from GPU memory into internal container and returns const reference to it
	const std::vector<SingleResult>& read_results_gpu();

	// Reads decryptors data from GPU memory into internal container and returns const reference to it
	const std::vector<Decryptor>& read_decryptors_gpu();

	// Checks Kernel's results
	// Return true if Round should be finished
	bool check_results(const KernelResult& result);

	size_t get_mem_size() const;

	size_t num_batches() const;

	size_t results_per_batch() const;

	size_t keys_per_batch() const;

	std::string to_string() const;

public:
	inline uint32_t CudaBlocks() const { return CUDASetup[0]; }

	inline uint32_t CudaThreads() const { return CUDASetup[1]; }

	inline uint32_t CudaThreadIterations() const { return CUDASetup[2]; }

	inline const BruteforceConfig& Config() const { assert(inited); return kernel_inputs.config; }

	inline BruteforceType::Type Type() const { assert(inited); return Config().type; }

	inline KeeloqKernelInput& Inputs() { assert(inited); return kernel_inputs; }

private:

	void alloc();

	void free();

private:

	bool inited = false;

	uint32_t num_decryptors_per_batch = 0;

	//
	KeeloqKernelInput kernel_inputs;

	// Constant per run
	std::vector<EncData> encrypted_data;

	// could be pretty much data here
	std::vector<Decryptor> decryptors;

	// could be pretty much data here
	std::vector<SingleResult> block_results;

	uint32_t CUDASetup[3] = { 0 };
};