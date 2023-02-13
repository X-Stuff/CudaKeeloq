#include "keeloq_learning_types.h"

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
		return KeeloqLearningType::Name(KeeloqLearningType::ALL);
	}

	Type pTypes[KeeloqLearningType::TypeMaskLength] = {0};
    to_mask(learning_types, pTypes);

	return to_string(pTypes);
}

std::string KeeloqLearningType::to_string(const Type learning_types[])
{
	if (learning_types[KeeloqLearningType::LAST])
	{
		return KeeloqLearningType::Name(KeeloqLearningType::ALL);
	}

	std::string result;
	for (auto type = 0; type < KeeloqLearningType::LAST; ++type)
	{
		if (learning_types[type])
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

void KeeloqLearningType::to_mask(const std::vector<Type>& in_types, Type out_mask[])
{
    if (in_types.size() > 0)
    {
        for (auto type : in_types)
        {
            out_mask[type] = true;
        }
    }
    else
    {
        // set all
        out_mask[KeeloqLearningType::ALL] = true;
    }
}
