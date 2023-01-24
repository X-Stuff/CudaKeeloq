#define CU_FILE

#include "keeloq_types.cuh"

#include <algorithm>

extern const char* GeneratorTypeName[] = {
    "Dictionary",
    "Simple",
    "Filtered",
    "Alphabet",
    "Pattern"
};

extern const size_t GeneratorTypeNamesCount = sizeof(GeneratorTypeName) / sizeof(char*);



std::string BruteforceConfig::toString() const
{
    char tmp[384];
    const char* pGeneratorName = (uint8_t)type < GeneratorTypeNamesCount ? GeneratorTypeName[(uint8_t)type] : "<OUT OF RANGE>";
    switch (type)
    {
    case BruteforceConfig::Type::Simple:
        sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX", pGeneratorName, start.man, start.seed, start.man + brute_size());
        break;
    case BruteforceConfig::Type::Filtered:
        sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%u). Brute count: %zd.\n\tFilters: %s",
            pGeneratorName, start.man, start.seed, brute_size(), filters.toString().c_str());
        break;
    case BruteforceConfig::Type::Alphabet:
        {
        uint64_t first = alphabet.value(alphabet.lookup(start.man));
        uint64_t last  = alphabet.add(first, brute_size());
        sprintf_s(tmp, "Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX. (Count: %zd)  All invariants: %zd.\n\tAlphabet: %s",
            pGeneratorName, first, start.seed, last, brute_size(), alphabet.invariants(), alphabet.toString().c_str());
        break;
        }
    case BruteforceConfig::Type::Pattern:
        sprintf_s(tmp, "Type: %s. NOT IMPLEMEMENTED", pGeneratorName);
        break;
    case BruteforceConfig::Type::Dictionary:
        sprintf_s(tmp, "Type: %s. Words num: %zd", pGeneratorName, dict_size());
        break;
    default:
        sprintf_s(tmp, "UNSUPPORTED Type (%d): %s", (int)type, pGeneratorName);
        break;
    }

    return std::string(tmp);
}


BruteforceConfig::Alphabet::Alphabet(const std::vector<uint8_t>& alphabet)
{
    num = 0;

    for (int i = 0; i < alphabet.size(); ++i)
    {
        uint8_t byte = alphabet[i];
        if (!lut[byte])
        {
            lut[byte] = num;
            alp[num] = byte;
            ++num;
        }
    }

    assert(num < Alphabet::Size && "Using all bytes values as alphabet is not efficient");
}

__host__ std::string BruteforceConfig::Alphabet::toString() const
{
    char tmp[255 * 3] = {0}; // one byte is 'XX:' last is XX\0
    int write_index = 0;
    for (int i = 0; i < num; ++i)
    {
        write_index += sprintf(&tmp[write_index], i == 0 ? "%X" : ":%X", alp[i]);
    }
    return std::string(tmp);
}
