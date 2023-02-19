#define _CRT_SECURE_NO_WARNINGS

#include "common.h"

#include <stdlib.h>

#include "host_utils.h"

#include "algorithm/keeloq/keeloq_decryptor.h"


#ifndef _MSC_VER
    #include <byteswap.h>
#else
    #define bswap_64(x) _byteswap_uint64(x)
#endif



std::vector<Decryptor> host::utils::read_word_dictionary_file(const char* file)
{
    // TODO: Support as <key>:<seed>
    constexpr uint32_t SEED_UNSUPPORTED = 0;

    std::vector<Decryptor> results;

    if (FILE* file_dict = fopen(file, "r"))
    {
        char line[66] = { 0 };
        while (fgets(line, sizeof(line), file_dict))
        {
            int base = 0;
            if (line[0] == '0' && (line[1] == 'b' || line[1] == 'B'))
            {
                line[0] = ' ';
                line[1] = ' ';
                base = 2;
            }

            if (auto key = strtoull(line, nullptr, base))
            {
                results.push_back(Decryptor(key, SEED_UNSUPPORTED));
            }
            else
            {
                printf("Error: invalid line: `%s` in file: '%s'\n", line, file);
            }
        }

        fclose(file_dict);
    }

    return results;
}

std::vector<Decryptor> host::utils::read_binary_dictionary_file(const char* file, uint8_t mode)
{
    // TODO: Support as... no idea
    constexpr uint32_t SEED_UNSUPPORTED = 0;

    std::vector<Decryptor> decryptors;

    if (FILE* bin_file = fopen(file, "rb"))
    {
        uint8_t key[sizeof(uint64_t)] = { 0 };

        while (fread(key, sizeof(uint64_t), sizeof(uint8_t), bin_file))
        {
            uint64_t reversed = *(uint64_t*)key;
            uint64_t as_is = bswap_64(reversed);

            uint64_t key = mode == 0 ? as_is : reversed;

            decryptors.push_back(Decryptor(key, SEED_UNSUPPORTED));
            if (mode == 2)
            {
                // reversed already added above
                decryptors.push_back(Decryptor(as_is, SEED_UNSUPPORTED));
            }
        }

        fclose(bin_file);
    }

    return decryptors;
}

std::vector<uint8_t> host::utils::read_alphabet_binary_file(const char* file)
{
    constexpr uint32_t MaxFileSize = 256;

    if (FILE* alphabet_file = fopen(file, "rb"))
    {
        // alphabet with more than 256 bytes is impossible (or just has duplicates)
        uint8_t bytes[MaxFileSize];
        size_t read_bytes = fread(bytes, sizeof(uint8_t), sizeof(bytes), alphabet_file);

        fseek(alphabet_file, 0, SEEK_END);
        uint64_t size = ftell(alphabet_file);
        if (size > MaxFileSize)
        {
            printf("Warning: File's '%s' is %" PRIu64 " bytes. It is bigger than read %u bytes. Alphabet bytes should be unique!\n",
                file, size, MaxFileSize);
        }

        fclose(alphabet_file);

        std::vector<uint8_t> alphabet_bytes(&bytes[0], &bytes[read_bytes]);
        return alphabet_bytes;
    }

    return {};
}
