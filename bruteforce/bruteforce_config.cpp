#include "bruteforce_config.h"
#include "bruteforce_type.h"
#include "bruteforce_filters.h"
#include "bruteforce_alphabet.h"

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

BruteforceConfig BruteforceConfig::GetAlphabet(Decryptor first, const BruteforceAlphabet& alphabet, size_t num)
{
	// max operation take to check all keys with alphabet of size
	num = std::min(alphabet.invariants(), num);

	BruteforceConfig result(first, BruteforceType::Alphabet, num);
	result.alphabet = alphabet;
	return result;
}

BruteforceConfig BruteforceConfig::GetPattern(Decryptor first)
{
	// not implemented
	return BruteforceConfig(0, BruteforceType::LAST, 0);
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
		start = next;
	}
}

std::string BruteforceConfig::toString() const
{
	char tmp[384];
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
	{
		uint64_t first = alphabet.value(alphabet.lookup(start.man));
		uint64_t last = alphabet.add(first, brute_size());
		sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX. (Count: %zd)  All invariants: %zd.\n\tAlphabet: %s",
			pGeneratorName, first, start.seed, last, brute_size(), alphabet.invariants(), alphabet.toString().c_str());
		break;
	}
	case BruteforceType::Pattern:
	{
		sprintf_s(tmp, "Type: %s. NOT IMPLEMEMENTED", pGeneratorName);
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
