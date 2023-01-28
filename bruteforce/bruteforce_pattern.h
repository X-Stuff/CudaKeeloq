#pragma once

#include "common.h"

#include <vector>
#include <string>

struct BruteforcePattern
{
	BruteforcePattern(std::vector<std::vector<uint8_t>>&& pattern_bytes);

public:

	// Convert possible single-byte pattern string to set of bytes
	// 0xDA      -> single byte
	// 0x19-0x2A -> range
	// *         -> full
	// 0x91;0x23 -> set of specific bytes
	static std::vector<uint8_t> ParseBytes(std::string text);

	// full range bytes vector (from 0 to 255)
	static std::vector<uint8_t> BytesFull();

	// tries to parse single byte value like 0xA1 or FF
	static bool TryParseSingleByte(std::string text, uint8_t& out);

	// tries to parse single byte value like 0xA1 or FF
	static std::vector<uint8_t> TryParseRangeBytes(std::string text);
};