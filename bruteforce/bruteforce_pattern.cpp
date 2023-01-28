#include "bruteforce_pattern.h"
#include "bruteforce_alphabet.h"


namespace
{
std::vector<std::string> split(const std::string& delim, std::string input)
{
	std::vector<std::string> result;

	auto delim_index = input.find(delim);
	while (delim_index != std::string::npos)
	{
		std::string part = input.substr(0, delim_index);

		result.push_back(part);

		input.erase(0, delim_index + delim.size());

		delim_index = input.find(delim);
	}

	result.push_back(input);
	return result;
}
}


BruteforcePattern::BruteforcePattern(std::vector<std::vector<uint8_t>>&& pattern_bytes)
{

}

std::vector<uint8_t> BruteforcePattern::BytesFull()
{
	std::vector<uint8_t> result(0xff);
	for (uint8_t i = 0; i < 0xFF; ++i) { result[i] = i; };
	return result;
}

bool BruteforcePattern::TryParseSingleByte(std::string text, uint8_t& out)
{
	if (text.size() == 2 || (text.size() == 4 && text.rfind("0x", 0) == 0))
	{
		auto value = (uint8_t)strtoul(text.c_str(), nullptr, 16);

		if (value != 0 || (text.rfind("00") != std::string::npos))
		{
			out = value;
			return true;
		}
	}

	return false;
}

std::vector<uint8_t> BruteforcePattern::TryParseRangeBytes(std::string text)
{
	auto delimeter_index = text.find("-");
	std::string from = text.substr(0, delimeter_index);
	std::string to = text.substr(delimeter_index + 1);

	uint8_t byte_from;
	uint8_t byte_to;

	if (!TryParseSingleByte(from, byte_from) || !TryParseSingleByte(to, byte_to))
	{
		return { };
	}

	if (byte_from > byte_to)
	{
		byte_from = std::exchange(byte_to, byte_from);
	}

	std::vector<uint8_t> result(byte_to - byte_from + 1);
	for (uint8_t i = 0; i < result.size(); ++i)
	{
		result[i] = i + byte_from;
	}

	return result;
}

std::vector<uint8_t> BruteforcePattern::ParseBytes(std::string text)
{
	// easy - all
	if (text == "*")
	{
		return BytesFull();
	}

	// easy single byte 0xAA or AA
	uint8_t single_byte;
	if (TryParseSingleByte(text, single_byte))
	{
		return { single_byte };
	}

	// range bytes
	auto delimeter_index = text.find("-");
	if (delimeter_index != std::string::npos)
	{
		return TryParseRangeBytes(text);
	}

	// set of bytes
	std::vector<uint8_t> result;
	std::vector<std::string> splitted = split(";", text);
	for (const auto& part : splitted)
	{
		uint8_t byte;
		if (TryParseSingleByte(part, byte))
		{
			result.push_back(byte);
		}
	}

	return result;
}
