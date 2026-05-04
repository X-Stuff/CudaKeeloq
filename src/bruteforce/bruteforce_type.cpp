#include "bruteforce/bruteforce_type.h"


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
