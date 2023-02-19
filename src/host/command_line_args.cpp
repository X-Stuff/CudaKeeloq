#include "command_line_args.h"

#include <cuda_runtime_api.h>

#include "host/host_utils.h"
#include "host/console.h"

#define CXXOPTS_NO_EXCEPTIONS
#include "cxxopts/include/cxxopts.hpp"


namespace
{

inline void read_alphabets(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    const auto& alphabet_args = result[ARG_ALPHABET].as<std::vector<std::string>>();

    for (const auto& alphabet_arg : alphabet_args)
    {
        auto alphabet_bytes = host::utils::read_alphabet_binary_file(alphabet_arg.c_str());
        if (alphabet_bytes.size() > 0)
        {
            target.alphabets.emplace_back(alphabet_bytes);
        }
        else
        {
            // "61:ab:00:33..." -> "61,ab,00,33..."
            std::string alphabet_bytes_hex = alphabet_arg;
            std::replace(alphabet_bytes_hex.begin(), alphabet_bytes_hex.end(), ':', ',');

            // ["61","ab","00","33"]
            std::vector<std::string> alphabet_hex;
            cxxopts::values::parse_value(alphabet_bytes_hex, alphabet_hex);

            // ["61","ab","00","33"] -> [0x61, 0xAB, 0x00, 0x33]
            std::vector<uint8_t> alphabet_bytes;
            alphabet_bytes.reserve(alphabet_hex.size());

            for (const auto& hex : alphabet_hex)
            {
                auto value = (uint8_t)strtoul(hex.c_str(), nullptr, 16);
                if (value != 0 && hex != "00")
                {
                    alphabet_bytes.push_back(value);
                }
                else
                {
                    printf("Error: cannot parse alphabet byte '%s'!\n", hex.c_str());
                }
            }

            //
            if (alphabet_bytes.size() > 0)
            {
                target.alphabets.emplace_back(alphabet_bytes);

                if (alphabet_bytes.size() > 256)
                {
                    printf("Warning: Alphabet: '%s' has to much bytes in hex string: %zd (should be less than 257)\n",
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

inline void parse_dictionary_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    if (result[ARG_WORDDICT].count() > 0)
    {
        auto dict = result[ARG_WORDDICT].as<std::vector<std::string>>();

        std::vector<Decryptor> decryptors;

        for (const auto& dict_arg : dict)
        {
            std::vector<Decryptor> from_file = host::utils::read_word_dictionary_file(dict_arg.c_str());

            if (from_file.size() > 0)
            {
                decryptors.insert(decryptors.end(), from_file.begin(), from_file.end());
            }
            else if (auto key = strtoull(dict_arg.c_str(), nullptr, 16))
            {
                decryptors.push_back(Decryptor(key, 0));
            }
            else
            {
                printf("Error: invalid param passed to '--%s' argument: '%s' not a dictionary file neither word\n", ARG_WORDDICT, dict_arg.c_str());
            }
        }

        if (decryptors.size() > 0)
        {
            target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
        }
    }

    if (result[ARG_BINDICT].count() > 0)
    {
        auto dicts = result[ARG_BINDICT].as<std::vector<std::string>>();
        auto mode = result[ARG_BINDMODE].as<uint8_t>();

        std::vector<Decryptor> decryptors;

        for (const auto& bin_dict_path : dicts)
        {
            std::vector<Decryptor> decryptors = host::utils::read_binary_dictionary_file(bin_dict_path.c_str(), mode);

            if (decryptors.size() > 0)
            {
                target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
            }
            else
            {
                printf("Error: invalid param passed to '--%s' argument: '%s'. Invalid file!\n",
                    ARG_BINDICT, bin_dict_path.c_str());
            }
        }

    }
}

inline void parse_bruteforce_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    Decryptor first_decryptor(start_key, seed);

    auto count_key = result[ARG_COUNT].as<size_t>();

    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(first_decryptor, count_key));
}

inline void parse_bruteforce_filtered_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    Decryptor first_decryptor(start_key, seed);

    auto count_key = result[ARG_COUNT].as<size_t>();

    auto include_filter = result[ARG_IFILTER].as<BruteforceFilters::Flags::Type>();
    auto exclude_filter = result[ARG_EFILTER].as<BruteforceFilters::Flags::Type>();

    BruteforceFilters filters
    {
        include_filter,
        exclude_filter,
    };

    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(first_decryptor, count_key, filters));
}

inline void parse_alphabet_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    Decryptor first_decryptor(start_key, seed);

    auto count_key = result[ARG_COUNT].as<size_t>();

    for (const auto& alphabet : target.alphabets)
    {
        target.brute_configs.push_back(BruteforceConfig::GetAlphabet(first_decryptor, alphabet, count_key));
    }
}

inline void parse_pattern_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    Decryptor first_decryptor(start_key, seed);

    auto count_key = result[ARG_COUNT].as<size_t>();

    const auto& args = result[ARG_PATTERN].as<std::vector<std::string>>();

    auto full_bytes = DefaultByteArray<>::as_vector<std::vector<uint8_t>>();

    for (const auto& pattern_arg : args)
    {
        std::string pattern_bytes_hex = pattern_arg;
        std::replace(pattern_bytes_hex.begin(), pattern_bytes_hex.end(), ':', ',');

        std::vector<std::string> bytes_hex;
        cxxopts::values::parse_value(pattern_bytes_hex, bytes_hex);


        std::vector<std::vector<uint8_t>> result;
        for (auto& hex : bytes_hex)
        {
            // append alphabets' bytes
            if (hex.find("AL") != std::string::npos)
            {
                if (target.alphabets.size() == 0)
                {
                    printf("ERROR: Cannot use Alphabet in patterns - no provided alphabets. Replacing with *\n");
                    result.push_back(full_bytes);
                    continue;
                }

                auto al_index = strtoul(hex.substr(2).c_str(), nullptr, 10);
                if (al_index >= target.alphabets.size())
                {
                    printf("ERROR: Argument %s referring alphabet: %ld (index: %ld), but there are only %zd available. "
                        "Replacing with first one\n", hex.c_str(), al_index + 1, al_index, target.alphabets.size());
                    al_index = 0;
                }

                result.push_back(target.alphabets[al_index].as_vector());
            }
            else
            {
                // append regular pattern bytes
                auto bytes = BruteforcePattern::ParseBytes(hex);

                if (bytes.size() == 0)
                {
                    printf("ERROR: Invalid string '%s' for byte pattern! Ignoring. Replacing with *\n", hex.c_str());
                    result.push_back(full_bytes);
                }
                else
                {
                    result.push_back(bytes);
                }
            }
        }

        // validate
        if (result.size() > 8)
        {
            printf("Warning: Pattern string: '%s' contains more than 8 per-bytes delimiters (%zd) other will be ignored.\n",
                pattern_arg.c_str(), bytes_hex.size());
        }
        else if (result.size() < 8)
        {
            printf("Warning: Pattern string: '%s' contains less than 8 per-bytes delimiters (%zd) other fill with full pattern.\n",
                pattern_arg.c_str(), bytes_hex.size());

            for (auto i = result.size(); i < 8; ++i)
            {
                result.push_back(full_bytes);
            }
        }


        // reverse bytes
        std::reverse(result.begin(), result.end());

        // Add pattern attack to config
        target.brute_configs.push_back(BruteforceConfig::GetPattern(first_decryptor, BruteforcePattern(std::move(result), pattern_arg), count_key));
    }
}

constexpr const char* Usage()
{
    return ""
        "\nExample:\n"
        "\t./" APP_NAME " --" ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=1 --" ARG_START "=0x9876543210 --" ARG_COUNT "=1000000"
        "\n\n\tThis will launch simple bruteforce (+1) attack with 1 million checks from 0x9876543210. "
        "Will be checked ALL 16 (12 if no seed specified) keeloq learning types"

        "\nExample:\n"
        "\t./" APP_NAME " --" ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=3 --" ARG_LTYPE "=0 --" ARG_ALPHABET "=examples/alphabet.bin,10:20:30:AA:BB:CC:DD:EE:FF:02:33"
        "\n\n\tThis will launch 2 alphabets attacks for all possible combinations for SIMPLE learning Keeloq type. "
        "First alphabet will be taken from file, second - parsed from inputs."

        "\nExample:\n"
        "\t./" APP_NAME " --"  ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=4 --" ARG_LTYPE "=2 --" ARG_ALPHABET "=examples/alphabet.bin --" ARG_PATTERN "=AL0:11:AB|BC:*:00-44:AL0:AA-FF:01"
        "\n\n\tThis will launch pattern attacks with NORMAL keeloq learning type."
        "\n\tPattern applied 'as is' - big endian. The highest byte (0xXX.......) will be taken from 1st alphabet."
        "\n\tNext byte (0x..XX....) will be exact `0x11`."
        "\n\tNext byte (0x....XX..) will be `0xAB` or `0xBC`.\n"
        ;
}

}

CommandLineArgs CommandLineArgs::parse(int argc, const char** argv)
{
    cxxopts::Options options(APP_NAME, R"(

  _______  _____  ___     __ __        __
 / ___/ / / / _ \/ _ |   / //_/__ ___ / /  ___  ___ _
/ /__/ /_/ / // / __ |  / ,< / -_) -_) /__/ _ \/ _ `/
\___/\____/____/_/ |_| /_/|_|\__/\__/____/\___/\_, /
                                                /_/
   ___           __      ___
  / _ )______ __/ /____ / _/__  ___________ ____
 / _  / __/ // / __/ -_) _/ _ \/ __/ __/ -_) __/
/____/_/  \_,_/\__/\__/_/ \___/_/  \__/\__/_/

)");
    options.set_width(::console::get_width())
        .allow_unrecognised_options()
        .add_options()
        ("h," ARG_HELP, "Prints this help")

        // What to bruteforce
        (ARG_INPUTS, "Comma separated uint64 values (it's better to have 3)",
            cxxopts::value<std::vector<uint64_t>>(), "[k1, k1, k3]")

        // CUDA Setup
        (ARG_BLOCKS, "How many thread blocks to launch.",
            cxxopts::value<uint16_t>()->default_value("32"), "<num>")
        (ARG_THREADS, "How many threads will be launched in a block (if 0 - will use value from device).",
            cxxopts::value<uint16_t>()->default_value("0"), "<num>")

#ifndef NO_INNER_LOOPS
        (ARG_LOOPS, "How many loop iterations will one thread perform (keep it low).",
            cxxopts::value<uint16_t>()->default_value("2"), "<num>")
#endif

        // Mode - what bruteforce type will be used
        (ARG_MODE,
            "Bruteforce modes (comma separated):"
            "\n\t0: - Dictionary."
            "\n\t1: - Simple +1."
            "\n\t2: - Simple +1 with filters."
            "\n\t3: - Alphabet. Bruteforce +1 using only specified bytes."
            "\n\t4: - Pattern. Bruteforce with bytes selected by specified pattern.",
            cxxopts::value<std::vector<uint8_t>>(), "[m1,m2..]")
        (ARG_LTYPE,
            "Specific learning type (if you know your target well). Increases approximately x16 times (since doesn't calculate other types)"
            "\n\tV+1 means with reverse key (There are also more types. see source code):"
            "\n\t0: - Simple"
            "\n\t2: - Normal"
            "\n\t4: - Secure"
            "\n\t6: - Xor"
            "\nALL",
            cxxopts::value<std::vector<uint8_t>>()->default_value(KeeloqLearningType::ValueString(KeeloqLearningType::LAST)), "<type>")

        // Dictionaries files
        (ARG_WORDDICT, "Word dictionary file(s) or word(s) - contains hexadecimal strings which will be used as keys. e.g: 0xaabb1122 FFbb9800121212",
            cxxopts::value<std::vector<std::string>>(), "[f1,w1,...]")
        (ARG_BINDICT, "Binary dictionary file(s) - each 8 bytes of the file will be used as key (do not check duplicates or zeros)",
            cxxopts::value<std::vector<std::string>>(), "[b1,b2,...]")
        (ARG_BINDMODE, "Byte order mode for binary dictionary. 0 - as is. 1 - reverse, 2 - add both",
            cxxopts::value<uint8_t>()->default_value("0"), "<mode>")

        // Common (Bruteforce, Alphabet) - set start and end of execution
        (ARG_START, "The first key value which will be used for selected mode(s)",
            cxxopts::value<std::uint64_t>()->default_value("0"), "<value>")
        (ARG_SEED, "The seed which is used for bruteforce. If you specify it, most probably you need to check seed-only learning types (SECURE, FAAC)",
            cxxopts::value<std::uint32_t>()->default_value("0"), "<value>")
        (ARG_COUNT, "How many keys selected mode(s) should check.",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"), "<value>")

        // Alphabet
        (ARG_ALPHABET, "Alphabet binary file(s) or alphabet hex string(s) (like: AA:61:62:bb)",
            cxxopts::value<std::vector<std::string>>(), "[f1,a1,...]")

        // Pattern
        (ARG_PATTERN, "Pattern file (or pattern itself) - contains comma separated patterns like: AL1:0A:0x10-0x32:*:33|44|FA:FF\n"
            "Pattern is in big endian. That means first byte in patter is highest byte (e.g. 01:.... equals key 0x01......)\n"
            "Each byte in pattern separated by `:`, pattern types:\n"
            "\tAL[0-N]   - alphabet N (index in " ARG_ALPHABET " )\n"
            "\t0A        - constant. might be any byte as hex string\n"
            "\t0x10-0x32 - range. bytes from first to second (including)\n"
            "\t*         - any byte\n"
            "\t33|44|FA  - exact 3 bytes",
            cxxopts::value<std::vector<std::string>>(), "[f1,p1,...]")

        // Bruteforce filters
        (ARG_EFILTER, "Exclude filter: key matching this filters will not be used in bruteforce.",
            cxxopts::value<std::uint64_t>()->default_value("0"), "<value>")
        (ARG_IFILTER, "Include filter: only keys matching this filters will be used in bruteforce. (WARNING: may be EXTREMELY heavy to compute)",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"), "<value>")

        // Stop config
        (ARG_FMATCH, "Boolean. Stop bruteforce on first match. If inputs are 3+ probably should set to true",
            cxxopts::value<bool>()->default_value("true"))

        // Tests run
        (ARG_TEST, "Boolean. Run application tests. You'd better use them in debug.",
            cxxopts::value<bool>()->default_value("false"))

        // Benchmarks run
        (ARG_BENCHMARK, "Boolean. Run application benchmarks. You can specify learning and num loops type from command line also.",
            cxxopts::value<bool>()->default_value("false"))
        ;

    CommandLineArgs args;

    auto result = options.parse(argc, argv);

    // tests
    args.run_tests = result[ARG_TEST].as<bool>();

    // benchmarks
    args.run_bench = result[ARG_BENCHMARK].as<bool>();

    // CUDA setup
    args.init_cuda(result[ARG_BLOCKS].as<uint16_t>(), result[ARG_THREADS].as<uint16_t>(),
        result.count(ARG_LOOPS) > 0 ? result[ARG_LOOPS].as<uint16_t>() : 0);

    if (result.count(ARG_HELP) || result.arguments().size() == 0 || result.count(ARG_INPUTS) == 0)
    {
        if (!args.run_tests && !args.run_bench)
        {
            printf("\n%s\n", options.help().c_str());
            printf("%s\n", Usage());
        }
        return args;
    }

    // Inputs
    if (result.count(ARG_INPUTS) > 0)
    {
        args.init_inputs(result[ARG_INPUTS].as<std::vector<uint64_t>>());
        if (args.inputs.size() < 3)
        {
            printf("WARNING: No enough inputs: '%zd'! Need at least 3!\nHowever we'll proceed...\n", args.inputs.size());
        }
    }
    else
    {
        printf("Error: No inputs! Nothing to brute!\n%s\n", options.help().c_str());
        return args;
    }

    // Stop if need
    args.match_stop = result[ARG_FMATCH].as<bool>();

    // Alphabets
    read_alphabets(args, result);

    // Bruteforce configs
    if (result.count(ARG_MODE) > 0)
    {
        for (const auto& mode : result[ARG_MODE].as<std::vector<uint8_t>>())
        {
            switch (mode)
            {
            case (uint8_t)BruteforceType::Dictionary:
                parse_dictionary_mode(args, result);
                break;
            case (uint8_t)BruteforceType::Simple:
                parse_bruteforce_mode(args, result);
                break;
            case (uint8_t)BruteforceType::Filtered:
                parse_bruteforce_filtered_mode(args, result);
                break;
            case (uint8_t)BruteforceType::Alphabet:
                parse_alphabet_mode(args, result);
                break;
            case (uint8_t)BruteforceType::Pattern:
                parse_pattern_mode(args, result);
                break;
            default:
                break;
            }
        }

        if (args.brute_configs.size() == 0)
        {
            printf("Error: Cannot parse inputs to at least one brute config! '%s'\n",
                result[ARG_MODE].as<std::string>().c_str());
        }
    }
    else
    {
        printf("Error: you need to specify bruteforce mode!\n%s\n",
            options.help().c_str());
    }

    auto learning_type_bytes = result[ARG_LTYPE].as<std::vector<uint8_t>>();
    if (learning_type_bytes.size() > 0)
    {
        args.selected_learning.clear();
        for (auto value : learning_type_bytes)
        {
            if (value < KeeloqLearningType::LAST)
            {
                args.selected_learning.push_back(value);
            }
        }
    }

    return args;
}

bool CommandLineArgs::can_bruteforce()
{
	return inputs.size() > 0 && brute_configs.size() > 0;
}

void CommandLineArgs::init_inputs(const std::vector<uint64_t>& inp)
 {
	 inputs.reserve(inp.size());
     for (uint64_t ota : inp)
     {
        inputs.push_back(EncParcel(ota));
     }
 }

 void CommandLineArgs::init_cuda(uint16_t b, uint16_t t, uint16_t l)
 {
	 cuda_blocks	= b;
	 assert(cuda_blocks < max_cuda_blocks() && "This GPU cannot use this much blocks!");

	 cuda_threads	= t;
	 cuda_loops		= l;

	 if (cuda_threads == 0)
	 {
		 cuda_threads = (uint16_t)max_cuda_threads();
	 }
 }

 uint32_t CommandLineArgs::max_cuda_threads()
 {
	 cudaDeviceProp prop;
	 cudaGetDeviceProperties(&prop, 0);

	 return prop.maxThreadsPerBlock;
 }

 uint32_t CommandLineArgs::max_cuda_blocks()
 {
	 cudaDeviceProp prop;
	 cudaGetDeviceProperties(&prop, 0);

	 return prop.maxGridSize[0];
 }
