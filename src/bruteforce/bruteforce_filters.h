#pragma once

#include "common.h"

#include <vector>
#include <tuple>
#include <string>

#include <cuda_runtime_api.h>


constexpr uint8_t KeySizeBytes = sizeof(uint64_t);

constexpr uint8_t KeySizeBits = sizeof(uint64_t) * 8;

/**
 *  Filters for +1 bruteforce.
 * Will apply if `include(value) && !exclude(value)`
 *
 * Very little performance increase over simple +1
 * The filters are not optimized quite good and have a
 * lot of if blocks
 */
struct BruteforceFilters
{
	struct Flags
	{
		using Type = uint64_t;

		enum : Type
		{
			//
			None = 0,

			// filter function return true if key has more than 6 consecutive 0 bits
			Max6ZerosInARow = (1 << 0),

			// filter function return true if key has more than 6 consecutive 1 bits
			Max6OnesInARow = (1 << 1),


			// filter function return true if key has patterns like 11:22:33:44.. or FF:EE:DD:CC
			// 6 bytes by default
			BytesIncremental = (1 << 5),

			// filter function return true if key has repeating patterns like xx11:11:11:11xx or xxAA:AA:AA:AAxx
			BytesRepeat4 = (1 << 6),

			// filter function return true if key consist from only ascii numbers
			AsciiNumbers = (1 << 11),

			// filter function return true if key consist from only letters 'a'-'z' 'A'-'Z'
			AsciiAlpha = (1 << 12),

			// filter function return true if key consist from ascii letters and numbers
			AsciiAlphaNum = AsciiAlpha | AsciiNumbers,

			// filter function return true if key consist from only ASCII special symbols like '^%#&*
			AsciiSpecial = (1 << 13),

			// filter function return true if key consist from only ASCII typed characters
			AsciiAny = AsciiAlphaNum | AsciiSpecial,

			//
			All = (uint64_t)-1,
		};

		__host__ __device__ static inline bool HasAll(Type test, Type check)
		{
			return check == (test & check);
		}
		__host__ __device__ static inline bool HasAny(Type test, Type check)
		{
			return (test & check);
		}
	};

public:

	std::string toString(Flags::Type flags) const;

	std::string toString() const;

	// Return true if key pass current filters
	__host__ __device__ inline bool Pass(uint64_t key) const;

	__host__ __device__ inline static bool check_filters(uint64_t key, Flags::Type filter);

private:

	__host__ __device__ static bool all_any_ascii(uint64_t key);

	__host__ __device__ static bool all_ascii_num(uint64_t key);

	__host__ __device__ static bool all_ascii_alpha(uint64_t key);

	__host__ __device__ static bool all_ascii_symbol(uint64_t key);

	template<uint8_t ValueMin, uint8_t ValueMax>
	__host__ __device__ inline static bool all_min_max(uint64_t key);

	template<uint8_t bit, uint8_t MaxCount = 6>
	__host__ __device__ inline static bool has_consecutive_bits(uint64_t key);

	template<uint8_t MaxCount = 4>
	__host__ __device__ inline static bool has_consecutive_bytes(uint64_t key);

	template<uint8_t MaxCount = 6>
	__host__ __device__ inline static bool has_incremental_pattern(uint64_t key);


private:

	static const std::vector<std::tuple<Flags::Type, const char*>> FilterNames;

public:
	// Filter for keys to include.
	// WARNING:
	//  Could be executed INFINITELY LONG TIME
	//  e.g. start: 0x00000000001 filter SmartFilterFlags::AsciiAny
	//  it will took around trillions and trillions operations just to get to the first valid with simple +1
	//  In case of specific input - use dictionary, pattern or alphabet
	Flags::Type include = Flags::All;

	// Filter for keys to exclude
	Flags::Type exclude = Flags::None;
};


template<uint8_t ValueMin, uint8_t ValueMax>
__host__ __device__ bool BruteforceFilters::all_min_max(uint64_t key)
{
	// for logical AND start should be with true
	bool result = true;
	uint8_t* bPtrKey = (uint8_t*)&key;

	UNROLL
	for (uint8_t i = 0; i < KeySizeBytes; ++i)
	{
		// TODO: Some vector instruction here
		result &= bPtrKey[i] >= ValueMin && bPtrKey[i] <= ValueMax;
	}

	return result;
}

template<uint8_t bit, uint8_t MaxCount /*= 6*/>
__host__ __device__ bool BruteforceFilters::has_consecutive_bits(uint64_t key)
{
	uint8_t result = false;
	uint64_t mask = (1 << MaxCount) - 1;

	key = bit ? key : ~key;

	UNROLL
	for (uint8_t i = 0; i < KeySizeBits; ++i)
	{
		// inverse - filter pass if no consecutive bits
		result |= (key & mask) == mask;
		key = key >> 1;
	}

	return result;
}

template<uint8_t MaxCount /*= 4*/>
__host__ __device__ bool BruteforceFilters::has_consecutive_bytes(uint64_t key)
{
	// for logical OR start should be with false
	bool result = false;

	uint8_t index = 0;
	uint8_t* bPtrKey = (uint8_t*)&key;

	UNROLL
	for (uint8_t i = 1; i < KeySizeBytes; ++i)
	{
		bool equal = bPtrKey[i] == bPtrKey[index];
		index = equal * index + (1 - equal) * i;

		result |= (i - index) >= (MaxCount - 1);
	}

	return result;
}

template<uint8_t MaxCount /*= 6*/>
__host__ __device__ bool BruteforceFilters::has_incremental_pattern(uint64_t key)
{
	// for logical OR start should be with false
	bool result = false;

	uint8_t index = 0;
	uint8_t* bPtrKey = (uint8_t*)&key;

	UNROLL
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


__host__ __device__ inline bool BruteforceFilters::check_filters(uint64_t key, Flags::Type filter)
{
	bool key_has_any = false;

	// fastest should go first
	if (!key_has_any && Flags::HasAll(filter, Flags::AsciiAny))
	{
		key_has_any |= all_any_ascii(key);
	}

	if (!key_has_any && Flags::HasAny(filter, Flags::AsciiNumbers))
	{
		key_has_any |= all_ascii_num(key);
	}

	if (!key_has_any && Flags::HasAny(filter, Flags::AsciiAlpha))
	{
		key_has_any |= all_ascii_alpha(key);
	}

	if (!key_has_any && Flags::HasAny(filter, Flags::AsciiSpecial))
	{
		key_has_any |= all_ascii_symbol(key);
	}

	//
	if (!key_has_any && Flags::HasAny(filter, Flags::Max6OnesInARow))
	{
		key_has_any |= has_consecutive_bits<1>(key);
	}

	if (!key_has_any && Flags::HasAny(filter, Flags::Max6ZerosInARow))
	{
		key_has_any |= has_consecutive_bits<0>(key);
	}

	if (!key_has_any && Flags::HasAny(filter, Flags::BytesRepeat4))
	{
		key_has_any |= has_consecutive_bytes(key);
	}

	if (!key_has_any && Flags::HasAny(filter, BruteforceFilters::Flags::BytesIncremental))
	{
		key_has_any |= has_incremental_pattern(key);
	}

	return key_has_any;
}

__host__ __device__ inline bool BruteforceFilters::Pass(uint64_t key) const
{
	bool pass = true;

	if (include != Flags::All && include != Flags::None)
	{
		// Include keys match patterns
		pass &= check_filters(key, include);
	}

	if (exclude != Flags::None && exclude != Flags::All)
	{
		// Exclude keys  which match patterns
		pass &= !check_filters(key, exclude);
	}

	return pass;
}


__host__ __device__ inline bool BruteforceFilters::all_any_ascii(uint64_t key)
{
	constexpr uint8_t value_min = '!';
	constexpr uint8_t value_max = '~';

	return all_min_max<value_min, value_max>(key);
}

__host__ __device__ inline bool BruteforceFilters::all_ascii_num(uint64_t key)
{
	constexpr uint8_t value_min = '0';
	constexpr uint8_t value_max = '9';

	return all_min_max<value_min, value_max>(key);
}

__host__ __device__ inline bool BruteforceFilters::all_ascii_alpha(uint64_t key)
{
	// for logical AND start should be with true
	bool result = true;
	uint8_t* bPtrKey = (uint8_t*)&key;

	UNROLL
		for (uint8_t i = 0; i < KeySizeBytes; ++i)
		{
			result &= (bPtrKey[i] >= 'a' && bPtrKey[i] <= 'z') || (bPtrKey[i] >= 'A' && bPtrKey[i] <= 'Z');
		}

	return result;
}

__host__ __device__ inline bool BruteforceFilters::all_ascii_symbol(uint64_t key)
{
	// for logical AND start should be with true
	bool result = true;
	uint8_t* bPtrKey = (uint8_t*)&key;

	UNROLL
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
