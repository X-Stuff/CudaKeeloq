#include "keeloq_learning_types.h"

#include <string>
#include <array>

namespace KeeloqLearning
{

Matrix::Matrix(std::vector<Pair> pairs)
{
    if (pairs.empty())
    {
        matrix = kEverything;
        return;
    }

    for (const auto& learning : pairs)
    {
        enable(learning.type, learning.mod);
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

    static constexpr auto ModNames = std::array<const char*, ModsNum>{ "Reg", "Rev", "Inv" };

    for (auto i = 0; i < ModsNum; ++i)
    {
        auto mod = static_cast<Mod>(i);

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

const char* KeeloqLearning::Name(Mod mod)
{
    switch (mod)
    {
        case Mod::Regular: return "Regular";
        case Mod::ReversedKey: return "ReversedKey";
        case Mod::InvertedDec: return "InvertedDec";
        default: return "Unknown";
    }
}

}