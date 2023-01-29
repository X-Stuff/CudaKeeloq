#include "bruteforce_config.h"
#include "bruteforce_type.h"
#include "bruteforce_filters.h"

#include "algorithm/keeloq/keeloq_decryptor.h"


BruteforceConfig BruteforceConfig::GetDictionary(std::vector<Decryptor>&& dictionary)
{
	BruteforceConfig result(0, BruteforceType::Dictionary, dictionary.size());
	result.decryptors = std::move(dictionary);
	return result;
};

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, size_t size)
{
	return BruteforceConfig(first, BruteforceType::Simple, size);
}

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, size_t size, const BruteforceFilters& filters)
{
	BruteforceConfig result(first, BruteforceType::Filtered, size);
	result.filters = filters;
	return result;
}

BruteforceConfig BruteforceConfig::GetAlphabet(Decryptor first, const MultibaseDigit& alphabet, size_t num)
{
	auto result = GetPattern(first, BruteforcePattern(alphabet), num);
	result.type = BruteforceType::Alphabet;
	return result;
}

BruteforceConfig BruteforceConfig::GetPattern(Decryptor first, const BruteforcePattern& pattern, size_t num)
{
	num = std::min(pattern.size() - 1, num);

	BruteforceConfig result(first, BruteforceType::Pattern, num);
	result.pattern = pattern;
	return result;
}

uint64_t BruteforceConfig::dict_size() const
{
	if (type == BruteforceType::Dictionary)
	{
		return size;
	}
	return 0;
}

uint64_t BruteforceConfig::brute_size() const
{
	if (type != BruteforceType::Dictionary)
	{
		return size;
	}
	return 0;
}

void BruteforceConfig::next_decryptor()
{
	if (type != BruteforceType::Dictionary)
	{
		start = last;

		if (type == BruteforceType::Alphabet || type == BruteforceType::Pattern)
		{
			// +1 for these attacks cause next here is the last *checked*
			auto startnum = pattern.init(start.man);
			start.man = pattern.next(startnum, 1).number();
		}
		else if (type == BruteforceType::Simple || type == BruteforceType::Filtered)
		{
			start.man++;
		}
	}
}

std::string BruteforceConfig::toString() const
{
	char tmp[8196];
	const char* pGeneratorName = BruteforceType::Name(type);
	switch (type)
	{
	case BruteforceType::Simple:
	{
		sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX", pGeneratorName, start.man, start.seed, start.man + brute_size());
		break;
	}
	case BruteforceType::Filtered:
	{
		sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%u). Brute count: %zd.\n\tFilters: %s",
			pGeneratorName, start.man, start.seed, brute_size(), filters.toString().c_str());
		break;
	}
	case BruteforceType::Alphabet:
	case BruteforceType::Pattern:
	{
		MultibaseNumber begin = pattern.init(start.man);
		MultibaseNumber end = pattern.next(begin, brute_size());

		auto written = sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX. (Count: %zd)  All invariants: %zd",
			pGeneratorName, begin.number(), start.seed, end.number(), brute_size(), pattern.size());

		if (type == BruteforceType::Alphabet)
		{
			written = sprintf_s(&tmp[written], sizeof(tmp) - written,
				"\n\tAlphabet: %s", pattern.bytes_variants(0).to_string().c_str());
		}
		else
		{
			std::string pattern_string = pattern.to_string(true);
			written = sprintf_s(&tmp[written], sizeof(tmp) - written, "\nPattern: %s", pattern_string.c_str());
		}
		break;
	}
	case BruteforceType::Dictionary:
	{
		sprintf_s(tmp, "Type: %s. Words num: %zd", pGeneratorName, dict_size());
		break;
	}
	default:
	{
		sprintf_s(tmp, "UNSUPPORTED Type (%d): %s", (int)type, pGeneratorName);
		break;
	}
	}

	return std::string(tmp);
}
