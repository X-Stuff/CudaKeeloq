#pragma once

#include <vector>

struct Decryptor;

namespace host
{
namespace utils
{

// Read file with set with hexadecimal string keys
std::vector<Decryptor> read_word_dictionary_file(const char* file);

// Read binary file as set of keys 8-bytes each
std::vector<Decryptor> read_binary_dictionary_file(const char* file, uint8_t mode);

// Read first 256 bytes of a binary file as alphabet
std::vector<uint8_t> read_alphabet_binary_file(const char* file);
}
}