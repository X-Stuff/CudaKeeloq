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


Matrix::Matrix(const std::vector<Type>& types, Mod::Mask mask) : matrix(0)
{
    if (types.empty())
    {
        if (!!(mask & Mod::Mask::All))
        {
            // Everything
            matrix = kEverything;
        }
        else
        {
            // Everything but with mask
            for (int lType = 0; lType < TypesNum; ++lType)
            {
                auto type = static_cast<Type>(lType);
                for (int mType = 0; mType < Mod::NumTypes; ++mType)
                {
                    auto mod = static_cast<Mod::Type>(mType);

                    if (!!(mask & Mod::ToMask(mod)))
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
            for (int mType = 0; mType < Mod::NumTypes; ++mType)
            {
                auto mod = static_cast<Mod::Type>(mType);

                if (!!(mask & Mod::ToMask(mod)))
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

    static constexpr auto ModNames = std::array<const char*, Mod::NumTypes>{ "Reg", "Rev", "Inv" };

    for (auto i = 0; i < Mod::NumTypes; ++i)
    {
        auto mod = static_cast<Mod::Type>(i);

        at += snprintf(&buffer[at], sizeof(buffer) - at, "\t%s:   %6s %6s %6s %3s %5s %7s %7s %7s\n",
            ModNames[i],
            isEnabled(Type::Simple, mod) ? "+" : " ",
            isEnabled(Type::Normal, mod) ? "+" : " ",
            isEnabled(Type::Secure, mod) ? "+" : " ",
            isEnabled(Type::Xor,   mod) ? "+" : " ",
            isEnabled(Type::Faac,  mod) ? "+" : " ",
            isEnabled(Type::Serial1, mod) ? "+" : " ",
            isEnabled(Type::Serial2, mod) ? "+" : " ",
            isEnabled(Type::Serial3, mod) ? "+" : " "
        );
    }

    return std::string(buffer);
}

const char* KeeloqLearning::Name(Type type)
{
    switch (type)
    {
        case Type::Simple: return "Simple";
        case Type::Normal: return "Normal";
        case Type::Secure: return "Secure";
        case Type::Xor: return "Xor";
        case Type::Faac: return "Faac";
        case Type::Serial1: return "Serial1";
        case Type::Serial2: return "Serial2";
        case Type::Serial3: return "Serial3";
        default: return "Unknown";
    }
}

const char* KeeloqLearning::Name(Mod::Type mod)
{
    switch (mod)
    {
        case Mod::Type::Regular: return "Regular";
        case Mod::Type::ReversedKey: return "ReversedKey";
        case Mod::Type::InvertedDec: return "InvertedDec";
        default: return "Unknown";
    }
}

}