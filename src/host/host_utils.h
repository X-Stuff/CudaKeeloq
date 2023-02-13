#pragma once

#include <vector>

struct Decryptor;

namespace Host
{
namespace Utils
{

// Read file with set with hexadecimal string keys
std::vector<Decryptor> ReadWordDictionaryFile(const char* file);

// Read binary file as set of keys 8-bytes each
std::vector<Decryptor> ReadBinaryDictionaryFile(const char* file, uint8_t mode);

// Read first 256 bytes of a binary file as alphabet
std::vector<uint8_t> ReadAlphabetBinaryFile(const char* file);
}
}