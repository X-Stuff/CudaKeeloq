#include "keeloq_learning_types.h"

#include <cstring>


const char* KeeloqLearningType::LearningNames[] = {
	"KEELOQ_LEARNING_SIMPLE",
	"KEELOQ_LEARNING_SIMPLE_REV",
	"KEELOQ_LEARNING_NORMAL",
	"KEELOQ_LEARNING_NORMAL_REV",
	"KEELOQ_LEARNING_SECURE",
	"KEELOQ_LEARNING_SECURE_REV",
	"KEELOQ_LEARNING_MAGIC_XOR_TYPE_1",
	"KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV",
	"KEELOQ_LEARNING_FAAC",
	"KEELOQ_LEARNING_FAAC_REV",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3",
	"KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV",
	"ALL"
};

const size_t KeeloqLearningType::LearningNamesCount = sizeof(LearningNames) / sizeof(char*);


std::string KeeloqLearningType::to_string(const std::vector<Type>& learning_types)
{
	if (learning_types.size() == 0)
	{
		return LearningNames[KeeloqLearningType::LAST];
	}

	return to_mask(learning_types).to_string();
}

std::string KeeloqLearningType::Mask::to_string() const
{
    if (is_all_enabled())
    {
        return LearningNames[KeeloqLearningType::LAST];
    }

    std::string result;
    for (auto type = 0; type < KeeloqLearningType::LAST; ++type)
    {
        if (values[type])
        {
            if (result.size() > 0)
            {
                result += ", ";
            }
            result += KeeloqLearningType::Name(type);
        }
    }

    return result;
}

KeeloqLearningType::Mask KeeloqLearningType::to_mask(const std::vector<Type>& in_types)
{
    KeeloqLearningType::Mask result;

    if (in_types.size() > 0)
    {
        for (auto type : in_types)
        {
            result.values[type] = true;
        }
    }
    else
    {
        memcpy(result.values, KeeloqLearningType::Mask::All, sizeof(KeeloqLearningType::Mask::All));
    }

    return result;
}

bool KeeloqLearningType::Mask::is_all_enabled() const
{
    return std::memcmp(values, All, sizeof(All)) == 0;
}
