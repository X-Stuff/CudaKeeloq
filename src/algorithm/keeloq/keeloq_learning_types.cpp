#include "keeloq_learning_types.h"
#include "bruteforce/bruteforce_config.h"

#include <string>
#include <array>

namespace KeeloqLearning
{

Matrix::Matrix(const std::initializer_list<LearningItem>& params)
{
    if (params.size() > 0)
    {
        for (const auto& param : params)
        {
            enable(param.learning, param.imod, param.amod);
        }
    }
    else
    {
        matrix = kEverything;
    }
}

Matrix::Matrix(const std::vector<LearningType>& types, const std::vector<Modifier::Input>& iMods, const std::vector<Modifier::Algo>& aMods)
{
    if (types.empty() && iMods.empty() && aMods.empty())
    {
        matrix = kEverything;
        return;
    }

    static constexpr auto EveryLearning = EveryLearningType{};

    const auto& typesToEnable = types.empty() ? std::vector<LearningType>(EveryLearning.begin(), EveryLearning.end()) : types;

    assertf(!iMods.empty() && !aMods.empty(), "Input modifiers and Algo Modifiers must be provoided");

    for (auto type : typesToEnable)
    {
        for (auto imod : iMods)
        {
            for (auto amod : aMods)
            {
                enable(type, imod, amod);
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

    uint8_t enabledCount = 0;

    at += snprintf(&buffer[at], sizeof(buffer) - at, "Matrix:\n" "                     |  Simple |  Normal |  Secure |   Xor   |   Faac  | Serial1 | Serial2 | Serial3 |\n");
    at += snprintf(&buffer[at], sizeof(buffer) - at,             "_____________________|_________|_________|_________|_________|_________|_________|_________|_________|\n");

    for (auto i = 0; i < Modifier::InputModCount; ++i)
    {
        for (auto a = 0; a < Modifier::AlgoModCount; ++a)
        {
            auto imod = static_cast<Modifier::Input>(i);
            auto amod = static_cast<Modifier::Algo>(a);

            at += snprintf(&buffer[at], sizeof(buffer) - at, "%8s - %8s: |", KeeloqLearning::Name(imod), KeeloqLearning::Name(amod));

            for (auto learning : EveryLearningType{})
            {
                bool isLearningEnabled = isEnabled(learning, imod, amod);

                if (HasSeed(learning))
                {
                    if (!withSeed)
                    {
                        isLearningEnabled = false;
                    }
                }
                else if (seedOnly)
                {
                    isLearningEnabled = false;
                }

                enabledCount += static_cast<uint8_t>(isLearningEnabled);
                at += snprintf(&buffer[at], sizeof(buffer) - at, "    %s    |", isLearningEnabled ? "+" : " ");
            }
            at += snprintf(&buffer[at], sizeof(buffer) - at, "\n");
        }
    }
    at += snprintf(&buffer[at], sizeof(buffer) - at,            "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n");
    at += snprintf(&buffer[at], sizeof(buffer) - at, "Total enabled calculations: %u\n", enabledCount);

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

const char* Name(Modifier::Algo amod)
{
    switch (amod)
    {
    case Modifier::Algo::Normal:    return "Usual";
    case Modifier::Algo::Inverted:  return "Inverted";
    default: return "Unknown";
    }
}

const char* Name(Modifier::Input imod)
{
    switch (imod)
    {
        case Modifier::Input::Normal:       return "Nrm Key";
        case Modifier::Input::ReversedKey:  return "Rev Key";
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