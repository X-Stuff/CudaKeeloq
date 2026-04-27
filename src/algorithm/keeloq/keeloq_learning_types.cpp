#include "keeloq_learning_types.h"

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


Matrix::Matrix(const std::vector<LearningType>& types, Modificators::Mask mask) : matrix(0)
{
    if (types.empty())
    {
        if (!!(mask & Modificators::Mask::All))
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
                for (int mType = 0; mType < Modificators::Count; ++mType)
                {
                    auto mod = static_cast<Modificators::Type>(mType);

                    if (!!(mask & Modificators::ToMask(mod)))
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
            for (int mType = 0; mType < Modificators::Count; ++mType)
            {
                auto mod = static_cast<Modificators::Type>(mType);

                if (!!(mask & Modificators::ToMask(mod)))
                {
                    enable(type, mod);
                }
            }
        }
    }

}

std::string Matrix::to_string() const
{
    if (isAllEnabled())
    {
        return "<ALL>";
    }

    char buffer[8196];
    int at = 0;

    at += snprintf(&buffer[at], sizeof(buffer) - at, "Matrix:\n" "\tSimple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3\n");

    static constexpr auto ModNames = std::array<const char*, Modificators::Count>{ "Reg", "Rev", "Inv" };

    for (auto i = 0; i < Modificators::Count; ++i)
    {
        auto mod = static_cast<Modificators::Type>(i);

        at += snprintf(&buffer[at], sizeof(buffer) - at, "\t%s:   %6s %6s %6s %3s %5s %7s %7s %7s\n",
            ModNames[i],
            isEnabled(LearningType::Simple, mod) ? "+" : " ",
            isEnabled(LearningType::Normal, mod) ? "+" : " ",
            isEnabled(LearningType::Secure, mod) ? "+" : " ",
            isEnabled(LearningType::Xor,   mod) ? "+" : " ",
            isEnabled(LearningType::Faac,  mod) ? "+" : " ",
            isEnabled(LearningType::Serial1, mod) ? "+" : " ",
            isEnabled(LearningType::Serial2, mod) ? "+" : " ",
            isEnabled(LearningType::Serial3, mod) ? "+" : " "
        );
    }

    return std::string(buffer);
}

const char* KeeloqLearning::Name(LearningType type)
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

const char* KeeloqLearning::Name(Modificators::Type mod)
{
    switch (mod)
    {
        case Modificators::Type::Regular: return "Regular";
        case Modificators::Type::ReversedKey: return "ReversedKey";
        case Modificators::Type::InvertedDec: return "InvertedDec";
        default: return "Unknown";
    }
}

}