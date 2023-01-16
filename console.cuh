#pragma once


#include <stdio.h>
#include <conio.h>

#include "keeloq_types.cuh"

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
#define ARG_BRTSTART "brute-start"
#define ARG_BRTCOUNT "brute-count"
#define ARG_ALPHABET "alphabet"
#define ARG_PATTERN "pattern"
#define ARG_IFILTER "include-filter"
#define ARG_EFILTER "exclude-filter"
#define ARG_FMATCH "first-match"

namespace console
{



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

template<typename T = void>
__host__ inline CommandLineArgs parse_command_line(int argc, const char** argv)
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
            "Bruteforce mode:"
            "\n\t0: - Dictionary (default)."
            "\n\t1: - Simple +1."
            "\n\t2: - Simple +1 with filters."
            "\n\t3: - Alphabet. Bruteforce +1 using only specified bytes."
            "\n\t4: - Pattern. Bruteforce with bytes selected by specified pattern.",
            cxxopts::value<uint8_t>())

        // Dictionaries files
        (ARG_WORDDICT, "Word dictionary file (or words themselves) - contains hexadecimal strings which will be used as keys",
            cxxopts::value<std::vector<std::string>>())
        (ARG_BINDICT, "Binary dictionary file - each 8 bytes of the file will be used as key (do not check duplicates or zeroes)",
            cxxopts::value<std::string>())

        // Bruteforce
        (ARG_BRTSTART, "The first value which will be used for +1 increments. (default:0)",
            cxxopts::value<std::uint64_t>()->default_value("0"))
        (ARG_BRTCOUNT, "How many key brute should check. (default: -1, all)",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"))

        // Alphabet
        (ARG_ALPHABET, "Alphabet file (or alphabet itself) - contans colon separated hexadecimal strings which describes alphabet like: AA:61:62:bb",
            cxxopts::value<std::string>())

        // Pattern
        (ARG_PATTERN, "Pattern file (or pattern itself) - contans colon separated patterns for each byte in a key like: ??:ss:d?:3?:88:FF",
            cxxopts::value<std::string>())

        // filters
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

    args.init_inputs(result[ARG_INPUTS].as<std::vector<uint64_t>>());

    return args;
}
}