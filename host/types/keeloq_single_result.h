#pragma once

#include "common.h"

#include <stdint.h>

#include "host/types/keeloq_learning_types.h"


struct SingleResult
{
	static constexpr uint8_t ResultsCount = KeeloqLearningType::LAST;

	struct DecryptedArray
	{
		// fixed side array for every learning type
		uint32_t data[ResultsCount];

		void print(uint8_t element, bool ismatch) const;

		void print() const;
	};

	// Per each learning type
	DecryptedArray results;

	// used manufacturer key and seed for this result
	uint64_t man;
	uint32_t seed;

	// Input data
	uint64_t ota;

	// Set by GPU after analysis if there was a match
	KeeloqLearningType::Type match;

	void print(bool onlymatch = true) const;
};