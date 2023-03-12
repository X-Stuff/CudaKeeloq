#define _CRT_SECURE_NO_WARNINGS

#include "common.h"

#include <stdlib.h>
#include <cstring>

#include "host_utils.h"

#include "algorithm/keeloq/keeloq_decryptor.h"


#ifndef _MSC_VER
    #include <byteswap.h>
#else
    #define bswap_64(x) _byteswap_uint64(x)
#endif


namespace
{

bool parse_manufactorer_key(char* key_str, uint64_t& key)
{
    if (!key_str)
    {
        return 0;
    }

    int base = 0;
    if (key_str[0] == '0' && (key_str[1] == 'b' || key_str[1] == 'B'))
    {
        key_str[0] = ' ';
        key_str[1] = ' ';
        base = 2;
    }

    if (auto parsed = strtoull(key_str, nullptr, base))
    {
        key = parsed;
        return true;
    }
    else
    {
        return false;
    }
}

bool parse_seed(char* seed_str, uint32_t& seed)
{
    if (!seed_str)
    {
        return 0;
    }

    int base = 10;

    if (auto parsed = strtoul(seed_str, nullptr, base))
    {
        seed = parsed;
        return true;
    }
    else
    {
        return false;
    }
}

}



std::vector<Decryptor> host::utils::read_word_dictionary_file(const char* file)
{
    std::vector<Decryptor> results;

    if (FILE* file_dict = fopen(file, "r"))
    {
        char line[256] = { 0 };
        char delim[2] = ":";

        while (fgets(line, sizeof(line), file_dict))
        {
            auto man_str = strtok(line, delim);
            if (man_str == nullptr)
            {
                man_str = line;
            }

            uint64_t man = (uint64_t)0;
            uint32_t seed = (uint32_t)0;

            if (!parse_manufactorer_key(man_str, man))
            {
                printf("Error: invalid line: `%s` in file: '%s'\n", line, file);
            }

            auto seed_str = strtok(NULL, delim);
            parse_seed(seed_str, seed);

            results.emplace_back(man, seed);
        }

        fclose(file_dict);
    }

    return results;
}

std::vector<Decryptor> host::utils::read_binary_dictionary_file(const char* file, uint8_t mode, uint32_t seed)
{
    std::vector<Decryptor> decryptors;

    if (FILE* bin_file = fopen(file, "rb"))
    {
        uint8_t key[sizeof(uint64_t)] = { 0 };

        while (fread(key, sizeof(uint64_t), sizeof(uint8_t), bin_file))
        {
            uint64_t reversed = *(uint64_t*)key;
            uint64_t as_is = bswap_64(reversed);

            uint64_t key = mode == 0 ? as_is : reversed;

            decryptors.push_back(Decryptor(key, seed));
            if (mode == 2)
            {
                // reversed already added above
                decryptors.push_back(Decryptor(as_is, seed));
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
