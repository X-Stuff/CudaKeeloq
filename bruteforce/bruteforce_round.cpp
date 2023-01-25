#include "bruteforce_round.h"

#include "common.h"
#include "bruteforce_config.h"
#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include <cuda_runtime_api.h>


BruteforceRound::BruteforceRound(const std::vector<EncData>& data, const BruteforceConfig& gen, std::vector<KeeloqLearningType::Type> selected_learning,
	uint32_t blocks, uint32_t threads, uint32_t iterations)
	: encrypted_data(data)
{
	CUDASetup[0] = blocks;
	CUDASetup[1] = threads;
	CUDASetup[2] = iterations;

	num_decryptors_per_batch = iterations * threads * blocks;

	kernel_inputs.generator = gen;
	memset(kernel_inputs.learning_types, 0, sizeof(kernel_inputs.learning_types));

	if (selected_learning.size() > 0)
	{
		for (auto type : selected_learning)
		{
			kernel_inputs.learning_types[type] = true;
		}
	}
	else
	{
		// set all
		kernel_inputs.learning_types[KeeloqLearningType::LAST] = true;
	}
}

const std::vector<SingleResult>& BruteforceRound::ReadResults()
{
	kernel_inputs.results->copy(block_results);
	return block_results;
}

const std::vector<Decryptor>& BruteforceRound::ReadDecryptors()
{
	kernel_inputs.decryptors->copy(decryptors);
	return decryptors;
}

bool BruteforceRound::CheckResults(const KernelResult& result)
{
	if (result.error < 0)
	{
		printf("Kernel fatal error: %d\n Round should be finished!\n", result.error);
		return true;
	}
	else if (result.error != 0)
	{
		printf("CUDA calculations num errors: %d\n", result.error);
	}

	if (result.value > 0)
	{
		auto& all_results = ReadResults();

		printf("Matches count: %d\n", result.value);

		for (const auto& result : all_results)
		{
			if (result.match == KeeloqLearningType::INVALID)
			{
				continue;
			}

			result.print();
		}

		return true;
	}

	return false;
}

size_t BruteforceRound::NumBatches() const
{
	assert(inited);
	if (Type() == BruteforceType::Dictionary)
	{
		uint8_t non_align = Config().dict_size() % KeysCheckedInBatch() == 0 ? 0 : 1;
		return Config().dict_size() / KeysCheckedInBatch() + non_align;
	}
	else
	{
		uint8_t non_align = Config().dict_size() % KeysCheckedInBatch() == 0 ? 0 : 1;
		return Config().brute_size() / KeysCheckedInBatch() + non_align;
	}
}

size_t BruteforceRound::KeysCheckedInBatch() const
{
	assert(inited);
	return decryptors.size();
}

size_t BruteforceRound::ResultsPerBatch() const
{
	assert(inited);
	return block_results.size();
}

void BruteforceRound::Init()
{
	if (!inited)
	{
		// allocated once. updated every run on GPU
		decryptors = std::vector<Decryptor>(num_decryptors_per_batch);

		// allocated once. updated evert run on GPU. copied to CPU only if match found.
		block_results = std::vector<SingleResult>(encrypted_data.size() * decryptors.size());

		alloc();

		inited = true;
	}
}

std::string BruteforceRound::GetLearningTypeName() const
{
	if (kernel_inputs.learning_types[KeeloqLearningType::LAST])
	{
		return "ALL";
	}

	char str[512];
	int len = 0;
	for (auto type = 0; type < KeeloqLearningType::LAST; ++type)
	{
		if (kernel_inputs.learning_types[type])
		{
			len += sprintf_s(&str[len], sizeof(str) - len, "%s, ", KeeloqLearningType::Name(type));
		}
	}

	return std::string(str, len - 2);
}

std::string BruteforceRound::ToString() const
{
	assert(inited);

	char tmp[1024];
	sprintf_s(tmp, "Setup:\n"
		"\tCUDA: Blocks:%u Threads:%u Iterations:%u\n"
		"\tEncrypted data size:%zd\n"
		"\tLearning type:%s\n"
		"\tResults per batch:%zd\n"
		"\tDecryptors per batch:%zd\n"
		"\tConfig: %s",
		CudaBlocks(), CudaThreads(), CudaThreadIterations(),
		encrypted_data.size(), GetLearningTypeName().c_str(), ResultsPerBatch(), KeysCheckedInBatch(), Config().toString().c_str());

	return std::string(tmp);
}


void BruteforceRound::alloc()
{
	//
	assert(kernel_inputs.encdata == nullptr		&& "Encrypted data already allocated on GPU");
	assert(kernel_inputs.decryptors == nullptr	&& "Decryptors data already allocated on GPU");
	assert(kernel_inputs.results == nullptr		&& "Results data already allocated on GPU");

	// ALLOCATE ON GPU
	if (kernel_inputs.encdata == nullptr)
	{
		kernel_inputs.encdata = CudaArray<EncData>::allocate(encrypted_data);
	}

	if (kernel_inputs.decryptors == nullptr)
	{
		kernel_inputs.decryptors = CudaArray<Decryptor>::allocate(decryptors);
	}

	if (kernel_inputs.results == nullptr)
	{
		kernel_inputs.results = CudaArray<SingleResult>::allocate(block_results);
	}
}

void BruteforceRound::free()
{
	if (kernel_inputs.encdata != nullptr)
	{
		kernel_inputs.encdata->free();
		kernel_inputs.encdata = nullptr;
	}

	if (kernel_inputs.decryptors != nullptr)
	{
		kernel_inputs.decryptors->free();
		kernel_inputs.decryptors = nullptr;
	}

	if (kernel_inputs.results != nullptr)
	{
		kernel_inputs.results->free();
		kernel_inputs.results = nullptr;
	}

	encrypted_data.clear();
	decryptors.clear();
	block_results.clear();
}
