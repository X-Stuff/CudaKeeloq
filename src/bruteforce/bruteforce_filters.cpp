#include "bruteforce_filters.h"
#include "common.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


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
