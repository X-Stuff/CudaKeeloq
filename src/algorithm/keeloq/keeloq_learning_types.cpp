#include "keeloq_learning_types.h"
#include "bruteforce/bruteforce_config.h"

#include <string>
#include <array>

namespace KeeloqLearning
{

Matrix::Matrix(const std::initializer_list<Pair>& pairs) : matrix(0)
{
    if (pairs.size() > 0)
    {
        for (const auto& pair : pairs)
        {
            enable(pair.type, pair.mod);
        }
    }
    else
    {
        matrix = kEverything;
    }
}

Matrix::Matrix(const std::vector<LearningType>& types, Modifier::Mask mask) : matrix(0)
{
    if (types.empty())
    {
        if (!!(mask & Modifier::Mask::All))
        {
            // Everything
            matrix = kEverything;
        }
        else
        {
            // Everything but with mask
            for (int lType = 0; lType < LearningTypesCount; ++lType)
            {
                auto type = static_cast<LearningType>(lType);
                for (int mType = 0; mType < Modifier::Count; ++mType)
                {
                    auto mod = static_cast<Modifier::Type>(mType);

                    if (!!(mask & Modifier::ToMask(mod)))
                    {
                        enable(type, mod);
                    }
                }
            }
        }
    }
    else
    {
        // Specific
        for (const auto& type : types)
        {
            for (int mType = 0; mType < Modifier::Count; ++mType)
            {
                auto mod = static_cast<Modifier::Type>(mType);

                if (!!(mask & Modifier::ToMask(mod)))
                {
                    enable(type, mod);
                }
            }
        }
    }

}

std::string Matrix::to_string(const BruteforceConfig* bruteConfig) const
{
    char buffer[8196];
    int at = 0;

    const bool withSeed = bruteConfig == nullptr || bruteConfig->has_seed();
    const bool seedOnly = bruteConfig != nullptr && bruteConfig->type == BruteforceType::Seed;

    at += snprintf(&buffer[at], sizeof(buffer) - at, "Matrix:\n" "      Simple Normal Secure Xor Faac Serial1 Serial2 Serial3\n");

    static constexpr auto ModNames = std::array<const char*, Modifier::Count>{ "Reg", "Rev", "Inv" };

    for (auto i = 0; i < Modifier::Count; ++i)
    {
        auto mod = static_cast<Modifier::Type>(i);

        at += snprintf(&buffer[at], sizeof(buffer) - at, "%s:    %-6s %-6s %-5s %-3s %-5s %-7s %-7s %-7s\n",
            ModNames[i],
            (isEnabled(LearningType::Simple, mod) && !seedOnly)     ? "+" : " ",
            (isEnabled(LearningType::Normal, mod) && !seedOnly)     ? "+" : " ",
            (isEnabled(LearningType::Secure, mod) && withSeed)      ? "+" : " ",
            (isEnabled(LearningType::Xor,   mod) && !seedOnly)      ? "+" : " ",
            (isEnabled(LearningType::Faac,  mod) && withSeed)       ? "+" : " ",
            (isEnabled(LearningType::Serial1, mod) && !seedOnly)    ? "+" : " ",
            (isEnabled(LearningType::Serial2, mod) && !seedOnly)    ? "+" : " ",
            (isEnabled(LearningType::Serial3, mod) && !seedOnly)    ? "+" : " "
        );
    }

    return std::string(buffer);
}

const char* Name(LearningType type)
{
    switch (type)
    {
        case LearningType::Simple: return "Simple";
        case LearningType::Normal: return "Normal";
        case LearningType::Secure: return "Secure";
        case LearningType::Xor: return "Xor";
        case LearningType::Faac: return "Faac";
        case LearningType::Serial1: return "Serial1";
        case LearningType::Serial2: return "Serial2";
        case LearningType::Serial3: return "Serial3";
        default: return "Unknown";
    }
}

const char* Name(Modifier::Type mod)
{
    switch (mod)
    {
        case Modifier::Type::Regular: return "Regular";
        case Modifier::Type::ReversedKey: return "ReversedKey";
        case Modifier::Type::InvertedDec: return "InvertedDec";
        default: return "Unknown";
    }
}

bool Parse(const char* data, LearningType& out)
{
    std::string name(data);
    for (auto& c : name)
    {
        c = std::tolower(c);
    }

    for (int i = 0; i < LearningTypesCount; ++i)
    {
        auto type = static_cast<LearningType>(i);
        std::string typeName(Name(type));

        for (auto& c : typeName)
        {
            c = std::tolower(c);
        }

        if (name == typeName || std::to_string(i) == name)
        {
            out = type;
            return true;
        }
    }
    return false;
}

}