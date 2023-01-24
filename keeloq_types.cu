#define CU_FILE

#include "keeloq_types.cuh"

#include <algorithm>


std::string BruteforceConfig::toString() const
{
    char tmp[384];
    const char* pGeneratorName =  BruteforceType::Name(type);
    switch (type)
    {
    case BruteforceType::Simple:
        sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX", pGeneratorName, start.man, start.seed, start.man + brute_size());
        break;
    case BruteforceType::Filtered:
        sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%u). Brute count: %zd.\n\tFilters: %s",
            pGeneratorName, start.man, start.seed, brute_size(), filters.toString().c_str());
        break;
    case BruteforceType::Alphabet:
        {
        uint64_t first = alphabet.value(alphabet.lookup(start.man));
        uint64_t last  = alphabet.add(first, brute_size());
        sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX. (Count: %zd)  All invariants: %zd.\n\tAlphabet: %s",
            pGeneratorName, first, start.seed, last, brute_size(), alphabet.invariants(), alphabet.toString().c_str());
        break;
        }
    case BruteforceType::Pattern:
        sprintf_s(tmp, "Type: %s. NOT IMPLEMEMENTED", pGeneratorName);
        break;
    case BruteforceType::Dictionary:
        sprintf_s(tmp, "Type: %s. Words num: %zd", pGeneratorName, dict_size());
        break;
    default:
        sprintf_s(tmp, "UNSUPPORTED Type (%d): %s", (int)type, pGeneratorName);
        break;
    }

    return std::string(tmp);
}

