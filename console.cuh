#pragma once


#include <stdio.h>
#include <conio.h>

#include "keeloq_types.cuh"

#define CXXOPTS_NO_EXCEPTIONS
#include "cxxopts/include/cxxopts.hpp"

#define console_clear() printf("\033[H\033[J")

#define console_cursor_up(lines) printf("\033[%dA", (lines))

#define console_cursor_ret_up() printf("\033[F")
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))

#define save_cursor_pos() printf("\033[s")
#define load_cursor_pos() printf("\033[u")

#define ARG_HELP "help"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_LOOPS "cuda-loops"
#define ARG_MODE "mode"
#define ARG_WORDDICT "word-dict"
#define ARG_BINDICT "bin-dict"
#define ARG_START "start"
#define ARG_COUNT "count"
#define ARG_ALPHABET "alphabet"
#define ARG_PATTERN "pattern"
#define ARG_IFILTER "include-filter"
#define ARG_EFILTER "exclude-filter"
#define ARG_FMATCH "first-match"

namespace console
{
namespace
{

inline void parse_dictionary_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    if (result[ARG_WORDDICT].count() > 0)
    {
        auto dict = result[ARG_WORDDICT].as<std::vector<std::string>>();

        std::vector<Decryptor> decryptors;

        for (const auto& dict_arg : dict)
        {
            if (FILE* file_dict = fopen(dict_arg.c_str(), "r"))
            {
                char line[32] = {0};
                while (fgets(line, sizeof(line), file_dict))
                {
                    if (auto key = strtoull(dict_arg.c_str(), nullptr, 16))
                    {
                        decryptors.push_back(key);
                    }
                    else
                    {
                        printf("Error: invalid line: `%s` in file: '%s'\n", line, dict_arg.c_str());
                    }
                }

                fclose(file_dict);
            }
            else if (auto key = strtoull(dict_arg.c_str(), nullptr, 16))
            {
                decryptors.push_back(key);
            }
            else
            {
                printf("Error: invalid param passed to '--%s' argument: '%s'\n", ARG_WORDDICT, dict_arg.c_str());
            }
        }

        if (decryptors.size() > 0)
        {
            target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
        }
    }

    if (result[ARG_BINDICT].count() > 0)
    {
        auto bin_file_path = result[ARG_BINDICT].as<std::string>();
        if (FILE* bin_file = fopen(bin_file_path.c_str(), "rb"))
        {
            std::vector<Decryptor> decryptors;

            uint8_t key[sizeof(uint64_t)] = {0};

            while (fread_s(key, sizeof(key), sizeof(uint64_t), sizeof(uint8_t), bin_file))
            {
                decryptors.push_back(*(uint64_t*)key);
            }

            fclose(bin_file);

            if (decryptors.size() > 0)
            {
                target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
            }
        }
        else
        {
            printf("Error: invalid param passed to '--%s' argument: '%s'. Cannot open file!\n",
                ARG_BINDICT, bin_file_path.c_str());
        }
    }
}

inline void parse_bruteforce_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto count_key = result[ARG_COUNT].as<size_t>();

    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(start_key, count_key));
}

inline void parse_bruteforce_filtered_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto count_key = result[ARG_COUNT].as<size_t>();

    auto include_filter = result[ARG_IFILTER].as<uint64_t>();
    auto exclude_filter = result[ARG_EFILTER].as<uint64_t>();

    BruteforceConfig::Filters filters
    {
        (SmartFilterFlags)include_filter,
        (SmartFilterFlags)exclude_filter,
    };

    printf("Filters are: %s", filters.toString().c_str());
    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(start_key, count_key, filters));
}

inline void parse_alphabet_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto count_key = result[ARG_COUNT].as<size_t>();

    const auto& alphabet_args = result[ARG_ALPHABET].as<std::vector<std::string>>();

    for (const auto& alphabet_arg : alphabet_args)
    {
        if (FILE* alphabet_file = fopen(alphabet_arg.c_str(), "rb"))
        {
            // alphabet with more than 255 bytes is impossible (oe just has duplicates)
            uint8_t bytes[255];
            size_t read_bytes = fread_s(bytes, sizeof(bytes), sizeof(uint8_t), sizeof(bytes), alphabet_file);
            fclose(alphabet_file);

            std::vector<uint8_t> alphabet_bytes(&bytes[0], &bytes[read_bytes]);
            target.brute_configs.push_back(BruteforceConfig::GetAlphabet(start_key, alphabet_bytes, count_key));
        }
        else
        {
            std::string alphabet_bytes_hex = alphabet_arg;
            std::replace(alphabet_bytes_hex.begin(), alphabet_bytes_hex.end(), ':',',');
            std::vector<uint8_t> alphabet_bytes;
            cxxopts::values::parse_value(alphabet_bytes_hex, alphabet_bytes);

            if (alphabet_bytes.size() > 0)
            {
                if (alphabet_bytes.size() < 0xFF)
                {
                    target.brute_configs.push_back(BruteforceConfig::GetAlphabet(start_key, alphabet_bytes, count_key));
                }
                else
                {
                    printf("Error: Alphabet: '%s' is not valid hex string! Too many entries: %zd (should be less than 255)\n",
                        result[ARG_ALPHABET].as<std::string>().c_str(), alphabet_bytes.size());
                }
            }
            else
            {
                printf("Error: Alphabet: '%s' is not valid file, neither valid alphabet hex string (like: AA:11:b3...)!\n",
                    alphabet_arg.c_str());
            }
        }
    }

}

inline void parse_pattern_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    printf("Error: patterns are not implemented");
}
}

inline void progress_bar(double percent, const std::chrono::seconds& elapsed)
{
    constexpr auto progress_width = 80;
    static char progress_fill[progress_width] = {0};
    static char progress_none[progress_width] = {0};
    if (progress_fill[0] == 0)
    {
        memset(progress_fill, '=', sizeof(progress_fill));
        memset(progress_none, '-', sizeof(progress_none));
    }

    printf("[%.*s>", (int)(progress_width * percent), progress_fill);
    printf("%.*s]", (int)(progress_width * (1 - percent)), progress_none);
    printf("%d%%  %02lld:%02lld:%02lld   \n", (int)(percent * 100),
        elapsed.count() / 3600, (elapsed.count() / 60) % 60, elapsed.count() % 60);
}

inline CommandLineArgs parse_command_line(int argc, const char** argv)
{
    cxxopts::Options options("CUDAKeeloq", "CUDA accelerated bruteforcer for keeloq.");
    options.add_options()
        ("h," ARG_HELP, "Prints this help")

        // What to bruteforce
        (ARG_INPUTS, "Comma separated uint64 values (it's better to have 3+)", cxxopts::value<std::vector<uint64_t>>())

        // CUDA Setup
        (ARG_BLOCKS, "How many thread blocks (block is first multiplier) to launch (default:32)",
            cxxopts::value<uint16_t>()->default_value("32"))
        (ARG_THREADS, "How many threads will be launched in a block (this is second multiplier) (default:256)",
            cxxopts::value<uint16_t>()->default_value("256"))
        (ARG_LOOPS, "How many loop iterations will one thread perform (default:32)",
            cxxopts::value<uint16_t>()->default_value("32"))

        // Mode - what bruteforce type will be used
        (ARG_MODE,
            "Bruteforce modes (comma separated):"
            "\n\t0: - Dictionary (default)."
            "\n\t1: - Simple +1."
            "\n\t2: - Simple +1 with filters."
            "\n\t3: - Alphabet. Bruteforce +1 using only specified bytes."
            "\n\t4: - Pattern. Bruteforce with bytes selected by specified pattern.",
            cxxopts::value<std::vector<uint8_t>>())

        // Dictionaries files
        (ARG_WORDDICT, "Word dictionary file (or words themselves) - contains hexadecimal strings which will be used as keys. e.g: 0xaabb1122 FFbb9800121212",
            cxxopts::value<std::vector<std::string>>())
        (ARG_BINDICT, "Binary dictionary file - each 8 bytes of the file will be used as key (do not check duplicates or zeroes)",
            cxxopts::value<std::string>())

        // Common (Bruteforce, Alphabet) - set start and end of execution
        (ARG_START, "The first key value which will be used for selected mode(s). (default:0)",
            cxxopts::value<std::uint64_t>()->default_value("0"))
        (ARG_COUNT, "How many keys selected mode(s) should check. (default: -1, all possible)",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"))

        // Alphabet
        (ARG_ALPHABET, "Alphabet binary file(s) or alphabet hex sting(s) (like: AA:61:62:bb)",
            cxxopts::value<std::vector<std::string>>())

        // Pattern
        (ARG_PATTERN, "Pattern file (or pattern itself) - contans colon separated patterns for each byte in a key like: ??:ss:d?:3?:88:FF",
            cxxopts::value<std::string>())

        // Bruteforce filters
        (ARG_EFILTER, "Exclude filter: key matching this filters will not be used in bruteforce (default:0,None)",
            cxxopts::value<std::uint64_t>()->default_value("0"))
        (ARG_IFILTER, "Include filter: only keys matching this filters will be used in bruteforce (default:-1,All)",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"))

        // Stop config
        (ARG_FMATCH, "Stop bruteforce on first match. If inputs are 3+ probably should set to true",
            cxxopts::value<bool>()->default_value("true"))
     ;
    options.set_width(140);

    CommandLineArgs args;

    auto result = options.parse(argc, argv);
    if (result.count(ARG_HELP) || result.arguments().size() == 0 || result.count(ARG_INPUTS) == 0)
    {
        printf("%s\n", options.help().c_str());
        return args;
    }

    // Inputs
    args.init_inputs(result[ARG_INPUTS].as<std::vector<uint64_t>>());

    // Stop if need
    args.match_stop = result[ARG_FMATCH].as<bool>();

    // CUDA
    args.init_cuda(result[ARG_BLOCKS].as<uint16_t>(),result[ARG_THREADS].as<uint16_t>(), result[ARG_LOOPS].as<uint16_t>());

    for (const auto& mode : result[ARG_MODE].as<std::vector<uint8_t>>())
    {
        switch (mode)
        {
        case (uint8_t)BruteforceConfig::Type::Dictionary:
            parse_dictionary_mode(args, result);
            break;
        case (uint8_t)BruteforceConfig::Type::Simple:
            parse_bruteforce_mode(args, result);
            break;
        case (uint8_t)BruteforceConfig::Type::Filtered:
            parse_bruteforce_filtered_mode(args, result);
            break;
        case (uint8_t)BruteforceConfig::Type::Alphabet:
            parse_alphabet_mode(args, result);
            break;
        case (uint8_t)BruteforceConfig::Type::Pattern:
            parse_pattern_mode(args, result);
            break;
        default:
            break;
        }

    }

    return args;
}

namespace tests
{
CommandLineArgs run()
{
    const char* commandline[] = {
        "tests",
        "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_BLOCKS"=32",
        "--" ARG_THREADS"=32",
        "--" ARG_LOOPS"=4",
        "--" ARG_MODE"=0,1,2,3,4,5",

        "--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,CEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3,examples/dictionary.words",
        "--" ARG_BINDICT"=examples/dictionary.bin",

        "--" ARG_START"=1",
        "--" ARG_COUNT"=0xFFFF",

        "--" ARG_ALPHABET"=61:62:63:64,examples/alphabet.bin",

        "--" ARG_IFILTER"=0x2" //SmartFilterFlags::Max6OnesInARow  other are very heavy
    };

    return parse_command_line(sizeof(commandline)/ sizeof(char*), commandline);
}
}
}