#pragma once

#include "common.h"

#include <cstring> // memcpy

#include "device/cuda_array.h"
#include "device/cuda_object.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_config.h"


// Input data for main keeloq calculation kernel
struct KeeloqKernelInput : TGenericGpuObject<KeeloqKernelInput>
{
	// Constant per-run input data (captured encoded)
	CudaArray<EncData>* encdata;

	// Single-run set of decryptors
	CudaArray<Decryptor>* decryptors;

	// Single-run results
	CudaArray<SingleResult>* results;

	// Which type of learning use for decryption // the last one indicates all
	KeeloqLearningType::Type learning_types[KeeloqLearningType::TypeMaskLength];

	// from this decryptor generation will start
	BruteforceConfig config;

	KeeloqKernelInput() : KeeloqKernelInput(nullptr, nullptr, nullptr, BruteforceConfig())
	{
	}

	KeeloqKernelInput(CudaArray<EncData>* enc, CudaArray<Decryptor>* dec, CudaArray<SingleResult>* res, const BruteforceConfig& config)
		: TGenericGpuObject<KeeloqKernelInput>(this), encdata(enc), decryptors(dec), results(res), learning_types(), config(config)
	{
	}

	KeeloqKernelInput(KeeloqKernelInput&& other) noexcept : TGenericGpuObject<KeeloqKernelInput>(this) {
		encdata = other.encdata;
		decryptors = other.decryptors;
		results = other.results;
		config = other.config;
		std::memcpy(learning_types, other.learning_types, sizeof(learning_types));
	}

	KeeloqKernelInput& operator=(KeeloqKernelInput&& other) = delete;
	KeeloqKernelInput& operator=(const KeeloqKernelInput& other) = delete;

public:
	void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num);

	void NextDecryptor();
};
