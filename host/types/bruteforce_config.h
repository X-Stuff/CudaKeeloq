#pragma once

#include "common.h"

#include <vector>
#include <string>

#include <cuda_runtime_api.h>

#include "host/types/keeloq_decryptor.h"

#include "host/types/bruteforce_alphabet.h"
#include "host/types/bruteforce_filters.h"
#include "host/types/bruteforce_type.h"
#include "host/types/bruteforce_config.h"

NS_LOCATION_BEGIN

/**
 *  Single run attack configuration
 * Run - selected type with specific parameters
 */
struct BruteforceConfig
{
	// HOST SET. ONCE. Which generator to use.
	BruteforceType::Type type;

	// HOST SET. UPDATING. PER BATCH. Decryption batch (or decryptors generation) will start from this
	Decryptor start;

	// HOST SET. ONCE. How many generator rounds should be taken (in fact how many times CUDA kernel will be called)
	size_t size;

	// Dictionary - HOST SET. ONCE.
	// Brute -      GPU SET. UPDATING.
	std::vector<Decryptor> decryptors;

	// HOST SET. ONCE. for filtered type.
	BruteforceFilters filters;

	// HOST SET. ONCE. for alphabet type.
	BruteforceAlphabet alphabet;

	// GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
	Decryptor next;

public:

	BruteforceConfig() : BruteforceConfig(0, BruteforceType::LAST, 0) {
	}

public:

	__host__ static BruteforceConfig GetDictionary(std::vector<Decryptor>&& dictionary);

	__host__ static BruteforceConfig GetBruteforce(Decryptor first, size_t size);

	__host__ static BruteforceConfig GetBruteforce(Decryptor first, size_t size, const BruteforceFilters& filters);

	__host__ static BruteforceConfig GetAlphabet(Decryptor first, const BruteforceAlphabet& alphabet, size_t num = (size_t)-1);

	__host__ static BruteforceConfig GetPattern(Decryptor first);

public:

	__host__ uint64_t dict_size() const;

	__host__ uint64_t brute_size() const;

	__host__ std::string toString() const;

	__host__ void next_decryptor();

private:
	BruteforceConfig(Decryptor start, BruteforceType::Type t, size_t num) :
		start(start), type(t), next(start), size(num)
	{
	}
};

NS_LOCATION_END