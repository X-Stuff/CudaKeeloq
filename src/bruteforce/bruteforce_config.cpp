#include "bruteforce/bruteforce_config.h"

#include <algorithm>

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_thread_result.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_type.h"


BruteforceConfig BruteforceConfig::GetDictionary(std::vector<Decryptor>&& dictionary, InputsTransform inTransform)
{
    BruteforceConfig result(Decryptor::Invalid(), BruteforceType::Dictionary, inTransform, dictionary.size());
    result.dictDecryptors = std::move(dictionary);
    return result;
};

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, InputsTransform inTransform, size_t size)
{
    return BruteforceConfig(first, BruteforceType::Simple, inTransform, size);
}

BruteforceConfig BruteforceConfig::GetBruteforce(Decryptor first, InputsTransform inTransform, size_t size, const BruteforceFilters& filters)
{
    BruteforceConfig result(first, BruteforceType::Filtered, inTransform, size);
    result.filters = filters;
    return result;
}

BruteforceConfig BruteforceConfig::GetSeedBruteforce(Decryptor first, InputsTransform inTransform, uint32_t size)
{
    return BruteforceConfig(first, BruteforceType::Seed, inTransform, size);
}

BruteforceConfig BruteforceConfig::GetXorBruteforce(Decryptor first, InputsTransform inTransform, uint32_t size /*= static_cast<uint32_t>(-1)*/)
{
    return BruteforceConfig(first, BruteforceType::Xor, inTransform | InputsTransform::Xored, size);
}

BruteforceConfig BruteforceConfig::GetAlphabet(Decryptor first, InputsTransform inTransform, const MultibaseDigit& alphabet, size_t num, const std::string& name)
{
    auto result = GetPattern(first, inTransform, BruteforcePattern(alphabet, name.empty() ? str::format<std::string>("Alphabet. Size: %d", alphabet.count()) : name), num);
    result.type = BruteforceType::Alphabet;
    return result;
}

BruteforceConfig BruteforceConfig::GetPattern(Decryptor first, InputsTransform inTransform, const BruteforcePattern& pattern, size_t num)
{
    num = std::min(pattern.size(), num);

    first = Decryptor::Make(pattern.init(first.man()).number(), first.seed(), first.has_seed());

    BruteforceConfig result(first, BruteforceType::Pattern, inTransform, num);
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
        return std::any_of(dictDecryptors.begin(), dictDecryptors.end(), [](const Decryptor& d) { return d.has_seed(); });
    }
    else
    {
        return start.has_seed();
    }
}


CudaConfig BruteforceConfig::cudaConfig(int desiredThreads, int desiredBlocks) const
{
    static constexpr uint16_t DefaultThreads = 256;

    return CudaConfig
    {
        desiredBlocks > 0 ? static_cast<uint32_t>(desiredBlocks) :
            CudaConfig::MaxCudaBlocks(desiredThreads, useSingleLearningKernels ? sizeof(ThreadResult::Single) : sizeof(ThreadResult::Multi)),
        desiredThreads > 0 ? static_cast<uint16_t>(desiredThreads) :
            DefaultThreads
    };
}

void BruteforceConfig::setTransforms(std::vector<InputsTransform> set)
{
    transforms = std::move(set);
}


void BruteforceConfig::setLearningMatrix(const KeeloqLearning::Matrix& matrix)
{
    learningMatrix = reduceMatrix(matrix);
    assert(learningMatrix.numEnabled() > 0 && "Can't brute with no learning types enabled in config");

    useSingleLearningKernels = learningMatrix.numEnabled() <= ThreadResult::Single::MaxLearningsNumInConfig;
}

std::string BruteforceConfig::transformsToString() const
{
    char result[512] = {};
    size_t at = 0;

    for (const auto& t : transforms)
    {
        at += snprintf(result + at, sizeof(result) - at, "%s%s", at == 0 ? "" : ", ", InputTransformName(t).c_str());
    }

    return std::string(result);
}


KeeloqLearning::Matrix BruteforceConfig::reduceMatrix(const KeeloqLearning::Matrix& matrix) const
{
    if (type == BruteforceType::Xor && !hasSeed())
    {
        // If seed is not specified for XorFix we can't brute any learning types, since all of them require seed in this case
        return KeeloqLearning::Matrix::Invalid();
    }

    auto result = matrix;

    for (auto ltype : KeeloqLearning::EveryLearningType{})
    {
        for (auto mtype : KeeloqLearning::EveryModifierType{})
        {
            if (type == BruteforceType::Seed)
            {
                // Seed only brute only for seeded learning types
                if (!KeeloqLearning::hasSeed(ltype))
                {
                    result.disable(ltype, mtype);
                }
            }
            else
            {
                // If current config doesn't have seed, we can't brute learning types that require seed
                if (KeeloqLearning::hasSeed(ltype) && !hasSeed())
                {
                    result.disable(ltype, mtype);
                }
            }
        }
    }

    return result;
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
    case BruteforceType::Xor:
    {
        return str::format<std::string>("Type: %s. Manufacturer key: 0x%llX Start XOR:%u",
            bruteTypeName, start.man(), start.seed());
    }
    }
    return str::format<std::string>("UNSUPPORTED Type (%d): %s", (int)type, bruteTypeName);
}

BruteforceConfig::BruteforceConfig(Decryptor start, BruteforceType::Type t, InputsTransform mask, size_t num) :
    type(t), start(start), size(num), dictDecryptors(), filters(), pattern(), last(start)
{

    for (auto flag : EveryInputTransform{})
    {
        if (t == BruteforceType::Xor)
        {
            if (!!(flag & InputsTransform::Xored))
            {
                transforms.push_back(flag);
            }
        }
        else
        {
            if ((flag & mask) == flag)
            {
                transforms.push_back(flag);
            }
        }
    }
}
