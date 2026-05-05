#include "bruteforce/bruteforce_type.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>


const char* BruteforceType::GeneratorTypeName[] = {
    "Dictionary",
    "Simple",
    "Filtered",
    "Alphabet",
    "Pattern",
    "Seed"
};

const size_t BruteforceType::GeneratorTypeNamesCount = sizeof(GeneratorTypeName) / sizeof(char*);

const char* BruteforceType::name(Type type)
{
    if (type > GeneratorTypeNamesCount)
    {
        return "UNKNOWN";
    }

    return GeneratorTypeName[type];
}

bool BruteforceType::parse(const char* input, Type& out)
{
    if (input == nullptr || *input == '\0')
    {
        return false;
    }

    std::string lowered(input);
    for (auto& c : lowered)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Numeric form: "0".."5"
    if (std::all_of(lowered.begin(), lowered.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
    {
        unsigned long v = std::strtoul(lowered.c_str(), nullptr, 10);
        if (v < GeneratorTypeNamesCount)
        {
            out = static_cast<Type>(v);
            return true;
        }
        return false;
    }

    // Name form, case-insensitive.
    for (size_t i = 0; i < GeneratorTypeNamesCount; ++i)
    {
        std::string candidate(GeneratorTypeName[i]);
        for (auto& c : candidate)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        if (candidate == lowered)
        {
            out = static_cast<Type>(i);
            return true;
        }
    }

    return false;
}
