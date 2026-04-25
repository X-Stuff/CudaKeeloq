#include "keeloq_learning_types.h"

#include <string>
#include <array>


KeeloqLearningMatrix::KeeloqLearningMatrix(std::vector<LearningPair> pairs)
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

std::string KeeloqLearningMatrix::to_string() const
{
    if (isAllEnabled())
    {
        return "<ALL>";
    }

    char buffer[8196];
    int at = 0;

    at += snprintf(&buffer[at], sizeof(buffer) - at, "Matrix:\n" "\tSimple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3\n");

    static constexpr auto ModNames = std::array<const char*, KeeloqLearningModsNum>{ "Reg", "Rev", "Inv" };

    for (auto i = 0; i < KeeloqLearningModsNum; ++i)
    {
        auto mod = static_cast<KeeloqLearningMod>(i);

        at += snprintf(&buffer[at], sizeof(buffer) - at, "\t%s:   %6s %6s %6s %3s %5s %7s %7s %7s\n",
            ModNames[i],
            isEnabled(KeeloqLearningType::Simple, mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Normal, mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Secure, mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Xor,   mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Faac,  mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Serial1, mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Serial2, mod) ? "+" : " ",
            isEnabled(KeeloqLearningType::Serial3, mod) ? "+" : " "
        );
    }

    return std::string(buffer);
}

const char* KeeloqLearning::Name(KeeloqLearningType type)
{
    switch (type)
    {
        case KeeloqLearningType::Simple: return "Simple";
        case KeeloqLearningType::Normal: return "Normal";
        case KeeloqLearningType::Secure: return "Secure";
        case KeeloqLearningType::Xor: return "Xor";
        case KeeloqLearningType::Faac: return "Faac";
        case KeeloqLearningType::Serial1: return "Serial1";
        case KeeloqLearningType::Serial2: return "Serial2";
        case KeeloqLearningType::Serial3: return "Serial3";
        default: return "Unknown";
    }
}

const char* KeeloqLearning::Name(KeeloqLearningMod mod)
{
    switch (mod)
    {
        case KeeloqLearningMod::Regular: return "Regular";
        case KeeloqLearningMod::ReversedKey: return "ReversedKey";
        case KeeloqLearningMod::InvertedDec: return "InvertedDec";
        default: return "Unknown";
    }
}
