#include "algorithm/keeloq/keeloq_learning_types.h"

#include <array>
#include <string>

#include "bruteforce/bruteforce_config.h"

namespace KeeloqLearning
{

Matrix::Matrix(const std::initializer_list<LearningItem>& params)
{
    if (params.size() > 0)
    {
        for (const auto& param : params)
        {
            enable(param.learning, param.amod);
        }
    }
    else
    {
        matrix = kEverything;
    }
}

Matrix::Matrix(const std::vector<LearningType>& types, const std::vector<Modifier::Algo>& aMods)
{
    if (types.empty() && aMods.empty())
    {
        matrix = kEverything;
        return;
    }

    static constexpr auto EveryLearning = EveryLearningType{};

    const auto& typesToEnable = types.empty() ? std::vector<LearningType>(EveryLearning.begin(), EveryLearning.end()) : types;

    assert( !aMods.empty() && "Input modifiers and Algo Modifiers must be provided");

    for (auto type : typesToEnable)
    {
        for (auto amod : aMods)
        {
            enable(type, amod);
        }
    }
}

std::string Matrix::toString() const
{
    char buffer[8196];
    int at = 0;

    at += snprintf(&buffer[at], sizeof(buffer) - at, "Learning matrix:\n"
        "           _______________________________________________________________________________\n"
        "          |  Simple |  Normal |  Secure |   Xor   |   Faac  | Serial1 | Serial2 | Serial3 |\n");
    at += snprintf(&buffer[at], sizeof(buffer) - at,
        "__________|_________|_________|_________|_________|_________|_________|_________|_________|\n");

    for (auto amod : EveryModifierType{})
    {
        at += snprintf(&buffer[at], sizeof(buffer) - at, "%8s: |", KeeloqLearning::name(amod));

        for (auto learning : EveryLearningType{})
        {
            const bool isLearningEnabled = isEnabled(learning, amod);
            at += snprintf(&buffer[at], sizeof(buffer) - at, "    %s    |", isLearningEnabled ? "+" : " ");
        }
        at += snprintf(&buffer[at], sizeof(buffer) - at, "\n");
    }

    at += snprintf(&buffer[at], sizeof(buffer) - at,            "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n");
    at += snprintf(&buffer[at], sizeof(buffer) - at, "Total enabled learnings: %u\n", numEnabled());

    return std::string(buffer);
}

std::vector<KeeloqLearning::LearningItem> Matrix::asItems() const
{
    std::vector<KeeloqLearning::LearningItem> items;

    for (auto a = 0; a < Modifier::AlgoModCount; ++a)
    {
        auto amod = static_cast<Modifier::Algo>(a);
        for (auto learning : EveryLearningType{})
        {
            if (isEnabled(learning, amod))
            {
                items.emplace_back(learning, amod);
            }
        }
    }

    return items;
}

const char* name(LearningType type)
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

const char* name(Modifier::Algo amod)
{
    switch (amod)
    {
    case Modifier::Algo::Normal:    return "Regular";
    case Modifier::Algo::Inverted:  return "Inverted";
    default: return "Unknown";
    }
}

bool parse(const char* data, LearningType& out)
{
    std::string nameStr(data);
    for (auto& c : nameStr)
    {
        c = std::tolower(c);
    }

    for (int i = 0; i < LearningTypesCount; ++i)
    {
        auto type = static_cast<LearningType>(i);
        std::string typeName(name(type));

        for (auto& c : typeName)
        {
            c = std::tolower(c);
        }

        if (nameStr == typeName || std::to_string(i) == nameStr)
        {
            out = type;
            return true;
        }
    }
    return false;
}

}
