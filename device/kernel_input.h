#pragma once

#include "common.h"

#include "device/cuda_array.h"
#include "device/cuda_object.h"

#include "host/types/keeloq_encrypted.h"
#include "host/types/keeloq_decryptor.h"
#include "host/types/keeloq_single_result.h"
#include "host/types/keeloq_learning_types.h"

#include "host/types/bruteforce_config.h"


// Input data for main keeloq calculation kernel
struct KernelInput : TGenericGpuObject<KernelInput>
{
	// Constant per-run input data (captured encoded)
	CudaArray<EncData>* encdata;

	// Single-run set of dectryptors
	CudaArray<Decryptor>* decryptors;

	// Single-run results
	CudaArray<SingleResult>* results;

	// Which type of learning use for decryption // the last one indicates all
	KeeloqLearningType::Type learning_types[KeeloqLearningType::LAST + 1];

	// from this decryptor generation will start
	BruteforceConfig generator;

	KernelInput() : KernelInput(nullptr, nullptr, nullptr, BruteforceConfig())
	{
	}

	KernelInput(CudaArray<EncData>* enc, CudaArray<Decryptor>* dec, CudaArray<SingleResult>* res, const BruteforceConfig& config)
		: TGenericGpuObject<KernelInput>(this), encdata(enc), decryptors(dec), results(res), generator(config), learning_types()
	{
	}

	KernelInput(KernelInput&& other) noexcept : TGenericGpuObject<KernelInput>(this) {
		encdata = other.encdata;
		decryptors = other.decryptors;
		results = other.results;
		generator = other.generator;
		memcpy(learning_types, other.learning_types, sizeof(learning_types));
	}

	KernelInput& operator=(KernelInput&& other) = delete;
	KernelInput& operator=(const KernelInput& other) = delete;

public:
	void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num);

	void NextDecryptor();
};
