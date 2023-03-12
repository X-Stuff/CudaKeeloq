#pragma once

#include "common.h"

/**
 *  Type of bruteforce attack
 */
struct BruteforceType
{
    using Type = uint8_t;

    enum : Type
    {
        // Generation will be skipped
        Dictionary = 0,
        None = Dictionary,

        // Simple +1 generator (very fast in terms of generation of decryptors candidates)
        Simple,

        // Simple +1 bruteforce but with filters applied performance may degrade)
        Filtered,

        // Specify alphabet and brute over it
        Alphabet,

        // ASCII pattern like ?A:01:??:3?:*
        Pattern,

        // Simple +1 seed bruteforce. Since seed is 32bit value, seed bruteforce may be done in a acceptable amount of time
        Seed,

        // Not for usage
        LAST,
    };

    static const char* Name(Type type);

private:

    static const char* GeneratorTypeName[];

    static const size_t GeneratorTypeNamesCount;
};

