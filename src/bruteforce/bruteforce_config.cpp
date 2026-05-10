#include "bruteforce/bruteforce_config.h"

#include <algorithm>

#include "algorithm/keeloq/keeloq_decryptor.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_type.h"


BruteforceConfig BruteforceConfig::GetDictionary(std::vector<Decryptor>&& dictionary, InputsMutation inputsMutation)
{
    BruteforceConfig result(Decryptor::Invalid(), BruteforceType::Dictionary, inputsMutation, dictionary.size());
    result.decryptors = std::move(dictionary);
    return result;
};

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, InputsMutation inputsMutation, size_t size)
{
    return BruteforceConfig(first, BruteforceType::Simple, inputsMutation, size);
}

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, InputsMutation inputsMutation, size_t size, const BruteforceFilters& filters)
{
    BruteforceConfig result(first, BruteforceType::Filtered, inputsMutation, size);
    result.filters = filters;
    return result;
}

BruteforceConfig BruteforceConfig::GetSeedBruteforce(Decryptor first, InputsMutation inputsMutation, uint32_t size)
{
    return BruteforceConfig(first, BruteforceType::Seed, inputsMutation, size);
}

BruteforceConfig BruteforceConfig::GetXorFixBruteforce(Decryptor first, InputsMutation inputsMutation, uint32_t size /*= static_cast<uint32_t>(-1)*/)
{
    return BruteforceConfig(first, BruteforceType::XorFix, inputsMutation | InputsMutation::XorFix, size);
}

BruteforceConfig BruteforceConfig::GetAlphabet(Decryptor first, InputsMutation inputsMutation, const MultibaseDigit& alphabet, size_t num, const std::string& name)
{
    auto result = GetPattern(first, inputsMutation, BruteforcePattern(alphabet, name.empty() ? str::format<std::string>("Alphabet. Size: %d", alphabet.count()) : name), num);
    result.type = BruteforceType::Alphabet;
    return result;
}

BruteforceConfig BruteforceConfig::GetPattern(Decryptor first, InputsMutation inputsMutation, const BruteforcePattern& pattern, size_t num)
{
    num = std::min(pattern.size(), num);

    first = Decryptor::Make(pattern.init(first.man()).number(), first.seed(), first.has_seed());

    BruteforceConfig result(first, BruteforceType::Pattern, inputsMutation, num);
    result.pattern = pattern;
    return result;
}

uint64_t BruteforceConfig::dictSize() const
{
    if (type == BruteforceType::Dictionary)
    {
        return size;
    }
    return 0;
}

uint64_t BruteforceConfig::bruteSize() const
{
    if (type != BruteforceType::Dictionary)
    {
        return size;
    }
    return 0;
}

void BruteforceConfig::nextDecryptor()
{
    if (type != BruteforceType::Dictionary)
    {
        start = last;

        if (type == BruteforceType::Alphabet || type == BruteforceType::Pattern)
        {
            // +1 for these attacks cause next here is the last *checked*
            auto startnum = pattern.init(start.man());
            start = Decryptor::Make(pattern.next(startnum, 1).number(), start.seed(), start.has_seed());
        }
        else if (type == BruteforceType::Simple || type == BruteforceType::Filtered)
        {
            start = Decryptor::Make(start.man() + 1, start.seed(), start.has_seed());
        }
    }
}

bool BruteforceConfig::hasSeed() const
{
    if (type == BruteforceType::Seed)
    {
        assert(start.has_seed() && "In seed bruteforce type your decryptors MUST have seeds enabled");
        return true;
    }
    else if (type == BruteforceType::Dictionary)
    {
        return std::any_of(decryptors.begin(), decryptors.end(), [](const Decryptor& d) { return d.has_seed(); });
    }
    else
    {
        return start.has_seed();
    }
}


bool BruteforceConfig::hasMutation(InputsMutation m) const
{
    if (maskAsSingleMutation)
    {
        return m == allowedMutations;
    }

    if (type == BruteforceType::XorFix)
    {
        // only mutation with XorFix flag
        return !!(m & InputsMutation::XorFix);
    }

    // Including None if allowed
    return (m & allowedMutations) == m;
}

std::vector<InputsMutation> BruteforceConfig::getMutations() const
{
    std::vector<InputsMutation> mutations;
    for (uint8_t i = 0; i < InputsMutationVariantsCount; ++i)
    {
        const auto flag = static_cast<InputsMutation>(i);
        if (hasMutation(flag))
        {
            mutations.push_back(flag);
        }
    }
    return mutations;
}

std::string BruteforceConfig::mutationsToString() const
{
    char result[512] = {};
    size_t at = 0;

    for (const auto& mutation : getMutations())
    {
        at += snprintf(result + at, sizeof(result) - at, "%s%s", at == 0 ? "" : ", ", name(mutation));
    }

    return std::string(result);
}

std::string BruteforceConfig::toString() const
{
    const char* bruteTypeName = BruteforceType::name(type);
    switch (type)
    {
    case BruteforceType::Simple:
    {
        return str::format<std::string>("Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX",
            bruteTypeName, start.man(), start.seed(), start.man() + bruteSize());
    }
    case BruteforceType::Filtered:
    {
        return str::format<std::string>("Type: %s. Initial: 0x%llX (seed:%u). Brute count: %zd.\n\tFilters: %s",
            bruteTypeName, start.man(), start.seed(), bruteSize(), filters.toString().c_str());
    }
    case BruteforceType::Alphabet:
    case BruteforceType::Pattern:
    {
        MultibaseNumber begin = pattern.init(start.man());
        MultibaseNumber end = pattern.next(begin, bruteSize() - 1);

        auto result =  str::format<std::string>("Type: %s. First: 0x%llX (seed:%u). Last: 0x%llX. (Count: %zd)  All invariants: %zd",
            bruteTypeName, begin.number(), start.seed(), end.number(), bruteSize(), pattern.size());

        if (type == BruteforceType::Alphabet)
        {
            result += str::format<std::string>("\n\tAlphabet: %s", pattern.bytesVariants(0).toString().c_str());
        }
        else
        {
            std::string pattern_string = pattern.toString(true);
            result += str::format<std::string>("\nPattern: %s", pattern_string.c_str());
        }
        return result;
    }
    case BruteforceType::Dictionary:
    {
        return str::format<std::string>("Type: %s. Words num: %zd", bruteTypeName, dictSize());
    }
    case BruteforceType::Seed:
    {
        return str::format<std::string>("Type: %s. Manufacturer key: 0x%llX Start Seed:%u",
            bruteTypeName, start.man(), start.seed());
    }
    case BruteforceType::XorFix:
    {
        return str::format<std::string>("Type: %s. Manufacturer key: 0x%llX Start Xor:%u",
            bruteTypeName, start.man(), start.seed());
    }
    }
    return str::format<std::string>("UNSUPPORTED Type (%d): %s", (int)type, bruteTypeName);
}

void BruteforceConfig::overrideMutationMask(InputsMutation mask, bool alone)
{
    allowedMutations = static_cast<InputsMutation>(static_cast<uint8_t>(mask) & InputsMutationMask);
    maskAsSingleMutation = alone;
}
