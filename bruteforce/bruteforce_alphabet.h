#pragma once

#include "common.h"

#include <vector>
#include <string>

#include <cuda_runtime_api.h>

#include "algorithm/multibase_system.h"
#include "algorithm/multibase_number.h"


struct BruteforceAlphabet
{
	BruteforceAlphabet() = default;

	BruteforceAlphabet(const std::vector<uint8_t>& alphabet) : system(alphabet)
	{
	}

	__host__ __device__ inline MultibaseNumber add(const MultibaseNumber& number, uint64_t amount) const;

	__host__ __device__ inline MultibaseNumber cast(uint64_t base10number) const { return system.cast(base10number); }

	__host__ __device__ inline bool valid() const { return system.num_digits() > 0; }

	__host__ __device__ inline size_t size() const { return system.invariants(); }

	__host__ std::string toString() const;

	__host__ std::vector<uint8_t> as_bytes() const { return system.as_bytes(); }

private:

	struct AlphabetSystem : public MultibaseSystem<8>
	{
		AlphabetSystem(const std::vector<uint8_t> numerals) : MultibaseSystem<8>(MultibaseSystem<8>::DigitConfig(numerals))
		{
		}

		AlphabetSystem() : MultibaseSystem<8>(MultibaseSystem<8>::DigitConfig({0}))
		{
		}

		__host__ __device__ size_t num_digits() const { return Digits[0].count(); }

		__host__ __device__ uint8_t get_digit(uint8_t index) const { return Digits[0].numeral(index); }

		__host__ std::vector<uint8_t> as_bytes() const { return Digits[0].as_bytes(); }
	};

	AlphabetSystem system;
};

__host__ __device__ inline MultibaseNumber BruteforceAlphabet::add(const MultibaseNumber& number, uint64_t amount) const
{
	MultibaseNumber result = number;
	return system.increment(result, amount);
}