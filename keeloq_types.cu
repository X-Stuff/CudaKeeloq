#include "keeloq_types.cuh"

#include <algorithm>

extern const char* LearningNames[] = {
    "KEELOQ_LEARNING_SIMPLE",
    "KEELOQ_LEARNING_SIMPLE_REV",
    "KEELOQ_LEARNING_NORMAL",
    "KEELOQ_LEARNING_NORMAL_REV",
    "KEELOQ_LEARNING_SECURE",
    "KEELOQ_LEARNING_SECURE_REV",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV",
    "KEELOQ_LEARNING_FAAC",
    "KEELOQ_LEARNING_FAAC_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV",
};

extern const char* GeneratorTypeName[] = {
    "Dictionary",
    "Simple",
    "Filtered",
    "Alphabet",
    "Pattern"
};

extern const size_t GeneratorTypeNamesCount = sizeof(GeneratorTypeName) / sizeof(char*);
extern const size_t LearningNamesCount = sizeof(LearningNames) / sizeof(char*);

static const std::vector<std::tuple<SmartFilterFlags, const char*>> FilterNames = {

    { SmartFilterFlags::None,               "None" },
    { SmartFilterFlags::All,                "All" },

    { SmartFilterFlags::Max6ZerosInARow,    "6 zero bit in a row" },
    { SmartFilterFlags::Max6OnesInARow,     "6 one bit in a row" },

    { SmartFilterFlags::BytesIncremental,   "Incremental bytes pattern" },
    { SmartFilterFlags::BytesRepeat4,       "4 same byte in a row" },

    { SmartFilterFlags::AsciiNumbers,       "ASCII numbers" },
    { SmartFilterFlags::AsciiAlpha,         "ASCII letters" },
    { SmartFilterFlags::AsciiSpecial,       "ASCII special characters" },
};

void SingleResult::DecryptedArray::print(uint8_t element, bool ismatch) const
{
    printf("[%-40s] Btn:0x%X\tSerial:0x%X\tCounter:0x%X\t%s\n", LearningNames[element],
        (data[element] >> 28),              // Button
        (data[element] >> 16) & 0x3ff,      // Serial
        data[element] & 0xFFFF,             // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::DecryptedArray::print() const
{
    for(uint8_t i = 0; i < ResultsCount; ++i )
    {
        print(i, false);
    }
}

void SingleResult::print(bool onlymatch /* = true */) const
{
    printf("Results (Input: 0x%llX - Man key: 0x%llX)\n\n", ota, man);

    for (uint8_t i = 0; i < ResultsCount; ++i)
    {
        bool isMatch = (uint8_t)match == i;
        if (!onlymatch)
        {
            results.print(i, isMatch);
        }
        else if (isMatch)
        {
            results.print(i, isMatch);
        }
    }
    printf("\n");
}

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

std::string BruteforceConfig::Filters::toString(SmartFilterFlags flags) const
{
    if (flags == SmartFilterFlags::None) { return "None"; }
    if (flags == SmartFilterFlags::All) { return "All"; }
    std::string result;
    for (const auto& pair : FilterNames)
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

std::string BruteforceConfig::Filters::toString() const
{
    std::string include_str = toString(include);
    std::string exclude_str = toString(exclude);

    return "Include filter: " + include_str + "\t Exclude filter: " + exclude_str;
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
