#pragma once

#include "common.h"

#include <vector>

namespace Tests
{
	bool AlphabetGeneration();
}

inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
	return std::vector<uint8_t>(ascii, ascii + num);
}