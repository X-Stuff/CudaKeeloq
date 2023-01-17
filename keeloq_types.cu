#include "keeloq_types.cuh"

#include <algorithm>

static const char* LearningNames[KeeloqLearningType::LAST] = {
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

static const char* GeneratorTypeName[(int)BruteforceConfig::Type::LAST] = {
    "Dictionary",
    "Simple",
    "Filtered",
    "Alphabet",
    "Pattern"
};

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
    printf("Results:\n\tOTA: 0x%llX\tMan key: 0x%llX\n\n", ota, man);

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
}


std::string BruteforceConfig::toString() const
{
    char tmp[128];
    if (type == Type::Dictionary) {
        sprintf_s(tmp, "Type: %s. size: %zd", GeneratorTypeName[(uint8_t)type % (uint8_t)Type::LAST], dict_size());
    }
    else {
        sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%ul). Brute size: %zd",
            GeneratorTypeName[(uint8_t)type % (uint8_t)Type::LAST], start.man, start.seed, brute_size());
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
        if ((uint64_t)std::get<0>(pair) & (uint64_t)flags)
        {
            result += std::get<1>(pair);
            result += " |";
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

BruteforceConfig::Alphabet::Alphabet(std::vector<uint8_t> alphabet)
{
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());

    assert(alphabet.size() < Alphabet::Size && "Using all bytes values as alphabet is not efficient");

    num = (uint8_t)min(alphabet.size(), (size_t)Size);
    memcpy(alp, alphabet.data(), num * sizeof(uint8_t));

    for (uint8_t i = 0; i < num; ++i)
    {
        lut[alp[i]] = i;
    }
}