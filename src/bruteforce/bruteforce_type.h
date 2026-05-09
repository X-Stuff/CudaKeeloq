#pragma once

#include "common.h"

/**
 * Enumerates the attack modes supported by the bruteforcer.
 * Each mode corresponds to a specific decryptor generator kernel.
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

        // Simple +1 xor for fix part bruteforce. Used for every learning types, Seed types uses the same value
        XorFix,

        // Not for usage
        LAST,
    };

    /** Human-readable name for a bruteforce type, or "UNKNOWN" for out-of-range values. */
    static const char* name(Type type);

    /** Parse a bruteforce-type name (case-insensitive) or numeric index. */
    static bool parse(const char* name, Type& out);

private:

    static const char* GeneratorTypeName[];

    static const size_t GeneratorTypeNamesCount;
};
