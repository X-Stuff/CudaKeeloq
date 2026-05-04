#pragma once

#include <vector>

struct Decryptor;

/**
 * Loaders for the dictionary / alphabet file formats accepted by the CLI.
 */
namespace host
{
namespace utils
{

/** Read a text file whose lines are hex keys (optionally `key:seed`). */
std::vector<Decryptor> readWordDictionaryFile(const char* file);

/**
 * Read a binary file, interpreting every 8 bytes as a key.
 * `mode`: 0 = as-is, 1 = byte-reversed, 2 = add both variants.
 * `seed` is optional; if null, no seed is attached to the produced decryptors.
 */
std::vector<Decryptor> readBinaryDictionaryFile(const char* file, uint8_t mode, const uint32_t* seed);

/** Read up to the first 256 bytes of a binary file as an alphabet. */
std::vector<uint8_t> readAlphabetBinaryFile(const char* file);

}
}
