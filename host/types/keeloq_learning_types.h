#pragma once

#include <type_traits>
#include <stdint.h>


struct KeeloqLearningType
{
	using Type = uint8_t;

	enum : Type
	{
		Simple = 0,
		Simple_Rev,

		Normal,
		Normal_Rev,

		Secure,
		Secure_Rev,

		Xor,
		Xor_Rev,

		Faac,
		Faac_Rev,

		Serial1,
		Serial1_Rev,

		Serial2,
		Serial2_Rev,

		Serial3,
		Serial3_Rev,

		LAST,

		INVALID = 0xff,
	};

	static constexpr const char* ValueString(Type type)
	{

		constexpr const char* LUT[]{
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
			"20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
			"30", "31", "32"
		};

		return LUT[type];
	}

	static constexpr const char* Name(Type type)
	{
		if (type >= LearningNamesCount) {
			return "INVALID";
		}

		return LearningNames[type];
	}

private:

	static const char* LearningNames[];

	static const size_t LearningNamesCount;
};
