#include "bruteforce_filters.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <assert.h>

NS_LOCATION_BEGIN

const std::vector<std::tuple<BruteforceFilters::Flags::Type, const char*>> BruteforceFilters::FilterNames =
{
	{ BruteforceFilters::Flags::None,               "None" },
	{ BruteforceFilters::Flags::All,                "All" },

	{ BruteforceFilters::Flags::Max6ZerosInARow,    "6 zero bit in a row" },
	{ BruteforceFilters::Flags::Max6OnesInARow,     "6 one bit in a row" },

	{ BruteforceFilters::Flags::BytesIncremental,   "Incremental bytes pattern" },
	{ BruteforceFilters::Flags::BytesRepeat4,       "4 same byte in a row" },

	{ BruteforceFilters::Flags::AsciiNumbers,       "ASCII numbers" },
	{ BruteforceFilters::Flags::AsciiAlpha,         "ASCII letters" },
	{ BruteforceFilters::Flags::AsciiSpecial,       "ASCII special characters" },
};


__host__ __device__ bool BruteforceFilters::check_filters(uint64_t key, Flags::Type filter)
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


__host__ __device__ bool BruteforceFilters::all_any_ascii(uint64_t key)
{
	constexpr uint8_t value_min = '!';
	constexpr uint8_t value_max = '~';

	return all_min_max<value_min, value_max>(key);
}

__host__ __device__ bool BruteforceFilters::all_ascii_num(uint64_t key)
{
	constexpr uint8_t value_min = '0';
	constexpr uint8_t value_max = '9';

	return all_min_max<value_min, value_max>(key);
}

__host__ __device__ bool BruteforceFilters::all_ascii_alpha(uint64_t key)
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

__host__ __device__ bool BruteforceFilters::all_ascii_symbol(uint64_t key)
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

__host__ __device__ bool BruteforceFilters::Pass(uint64_t key) const
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

std::string BruteforceFilters::toString(Flags::Type flags) const
{
	if (flags == Flags::None) { return "None"; }
	if (flags == Flags::All)  { return "All";  }

	std::string result;

	for (const auto& pair : BruteforceFilters::FilterNames)
	{
		auto check = (uint64_t)std::get<0>(pair);
		if (check != 0 && (check & (uint64_t)flags) == check)
		{
			result += std::get<1>(pair);
			result += " | ";
		}
	}
	if (result.size() > 0)
	{
		result.erase(result.end() - 3, result.end());
	}
	return result;
}

std::string BruteforceFilters::toString() const
{
	std::string include_str = toString(include);
	std::string exclude_str = toString(exclude);

	return "Include: '" + include_str + "'\tExclude: '" + exclude_str + "'";
}

NS_LOCATION_END