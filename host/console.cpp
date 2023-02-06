#include "console.h"

#include "bruteforce\bruteforce_pattern.h"

#pragma warning(push)
#pragma warning(disable: 4996)
#include "cpp-terminal\terminal_base.h"
#include "cpp-terminal\terminal.h"
#pragma warning(pop)

namespace
{

inline void read_alphabets(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    const auto& alphabet_args = result[ARG_ALPHABET].as<std::vector<std::string>>();

    for (const auto& alphabet_arg : alphabet_args)
    {
        FILE* alphabet_file;
        if (fopen_s(&alphabet_file, alphabet_arg.c_str(), "rb") == 0)
        {
            // alphabet with more than 255 bytes is impossible (oe just has duplicates)
            uint8_t bytes[255];
            size_t read_bytes = fread_s(bytes, sizeof(bytes), sizeof(uint8_t), sizeof(bytes), alphabet_file);
            fclose(alphabet_file);

            std::vector<uint8_t> alphabet_bytes(&bytes[0], &bytes[read_bytes]);
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
                if (alphabet_bytes.size() < 0xFF)
                {
                    target.alphabets.emplace_back(alphabet_bytes);
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


inline void parse_dictionary_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    if (result[ARG_WORDDICT].count() > 0)
    {
        auto dict = result[ARG_WORDDICT].as<std::vector<std::string>>();

        std::vector<Decryptor> decryptors;

        for (const auto& dict_arg : dict)
        {
            FILE* file_dict;
            if (fopen_s(&file_dict, dict_arg.c_str(), "r") == 0)
            {
                char line[66] = {0};
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
        auto dicts = result[ARG_BINDICT].as<std::vector<std::string>>();
        auto mode = result[ARG_BINDMODE].as<uint8_t>();

        std::vector<Decryptor> decryptors;

        for (const auto& bin_dict_path : dicts)
        {
            FILE* bin_file;
            if (fopen_s(&bin_file, bin_dict_path.c_str(), "rb") == 0)
            {
                std::vector<Decryptor> decryptors;

                uint8_t key[sizeof(uint64_t)] = {0};

                while (fread_s(key, sizeof(key), sizeof(uint64_t), sizeof(uint8_t), bin_file))
                {
                    uint64_t reversed = *(uint64_t*)key;
                    uint64_t as_is = _byteswap_uint64(reversed);

                    decryptors.push_back(mode == 0 ? as_is : reversed);
                    if (mode == 2)
                    {
                        // reversed already added above
                        decryptors.push_back(as_is);
                    }
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
                    ARG_BINDICT, bin_dict_path.c_str());
            }
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

    auto include_filter = result[ARG_IFILTER].as<BruteforceFilters::Flags::Type>();
    auto exclude_filter = result[ARG_EFILTER].as<BruteforceFilters::Flags::Type>();

    BruteforceFilters filters
    {
        include_filter,
        exclude_filter,
    };

    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(start_key, count_key, filters));
}

inline void parse_alphabet_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto count_key = result[ARG_COUNT].as<size_t>();

    for (const auto& alphabet : target.alphabets)
    {
        target.brute_configs.push_back(BruteforceConfig::GetAlphabet(start_key, alphabet, count_key));
    }
}

inline void parse_pattern_mode(CommandLineArgs& target, cxxopts::ParseResult& result)
{
    auto start_key = result[ARG_START].as<uint64_t>();
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
                    printf("ERROR: Argument %s referring alphabet: %d (index: %d), but there are only %zd available. "
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


        // Add pattern attack to config
        target.brute_configs.push_back(BruteforceConfig::GetPattern(start_key, BruteforcePattern(pattern_arg, std::move(result)), count_key));
    }
}

}

CommandLineArgs console::parse_command_line(int argc, const char** argv)
{
    cxxopts::Options options("CUDAKeeloq", "CUDA accelerated bruteforcer for keeloq.");
    options.add_options()
        ("h," ARG_HELP, "Prints this help")

        // What to bruteforce
        (ARG_INPUTS, "Comma separated uint64 values (it's better to have 3+)",
            cxxopts::value<std::vector<uint64_t>>(), "[k1,k1,k3...]")

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
            cxxopts::value<std::vector<uint8_t>>()->default_value(KeeloqLearningType::ValueString(KeeloqLearningType::LAST)), "type")

        // Dictionaries files
        (ARG_WORDDICT, "Word dictionary file(s) or word(s) - contains hexadecimal strings which will be used as keys. e.g: 0xaabb1122 FFbb9800121212",
            cxxopts::value<std::vector<std::string>>(), "[f1,w1,...]")
        (ARG_BINDICT, "Binary dictionary file(s) - each 8 bytes of the file will be used as key (do not check duplicates or zeros)",
            cxxopts::value<std::vector<std::string>>(), "[b1,b2,...]")
        (ARG_BINDMODE, "Byteorder mode for binary dictionary. 0 - as is. 1 - reverse, 2 - add both",
            cxxopts::value<uint8_t>()->default_value("0"), "mode")

        // Common (Bruteforce, Alphabet) - set start and end of execution
        (ARG_START, "The first key value which will be used for selected mode(s)",
            cxxopts::value<std::uint64_t>()->default_value("0"), "first")
        (ARG_COUNT, "How many keys selected mode(s) should check.",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"), "len")

        // Alphabet
        (ARG_ALPHABET, "Alphabet binary file(s) or alphabet hex string(s) (like: AA:61:62:bb)",
            cxxopts::value<std::vector<std::string>>(), "[f1,a1,...]")

        // Pattern
        (ARG_PATTERN, "Pattern file (or pattern itself) - contains colon separated patterns for each byte in a key like: AL1:0A:0x10-0x32:*:33;44;FA:FF\n"
            "Each byte in pattern separated by `:`, pattern types:\n"
            "\tAL[0-N]   - alphabet N (index in " ARG_ALPHABET " )\n"
            "\t0A        - constant. might be any byte as hex string\n"
            "\t0x10-0x32 - range. bytes from first to second (including)\n"
            "\t*         - any byte\n"
            "\t33;44;FA  - exact 3 bytes\n",
            cxxopts::value<std::vector<std::string>>(), "[f1,p1,...]")

        // Bruteforce filters
        (ARG_EFILTER, "Exclude filter: key matching this filters will not be used in bruteforce.",
            cxxopts::value<std::uint64_t>()->default_value("0"), "value")
        (ARG_IFILTER, "Include filter: only keys matching this filters will be used in bruteforce. (WARNING: may be EXTREMELY heavy to compute)",
            cxxopts::value<std::uint64_t>()->default_value("0xFFFFFFFFFFFFFFFF"), "value")

        // Stop config
        (ARG_FMATCH, "Stop bruteforce on first match. If inputs are 3+ probably should set to true",
            cxxopts::value<bool>()->default_value("true"), "0|1")

        // Tests run
        (ARG_TEST, "Run application tests. You'd better use them in debug.",
            cxxopts::value<bool>()->default_value("false"), "0|1")

        // Benchmarks run
        (ARG_BENCHMARK, "Run application benchmarks. You can specify learning and num loops type from command line also.",
            cxxopts::value<bool>()->default_value("false"), "0|1")
        ;
    options.set_width(CONSOLE_WIDTH);

    CommandLineArgs args;

    auto result = options.parse(argc, argv);
    if (result.count(ARG_HELP) || result.arguments().size() == 0 || result.count(ARG_INPUTS) == 0)
    {
        printf("\n%s\n", options.help().c_str());
        return args;
    }

    // tests
    args.run_tests = result[ARG_TEST].as<bool>();

    // benchmarks
    args.run_bench = result[ARG_BENCHMARK].as<bool>();

    // Inputs
    if (result.count(ARG_INPUTS) > 0)
    {
        args.init_inputs(result[ARG_INPUTS].as<std::vector<uint64_t>>());
        if (args.inputs.size() < 3)
        {
            printf("WARNING: No engough inputs: '%zd'! Need at least 3!\nHowever we'll proceed...\n", args.inputs.size());
        }
    }
    else
    {
        printf("Error: No inputs! Nothing to brute!\n%s\n", options.help().c_str());
        return args;
    }

    // Stop if need
    args.match_stop = result[ARG_FMATCH].as<bool>();

    // CUDA setup
    args.init_cuda(result[ARG_BLOCKS].as<uint16_t>(), result[ARG_THREADS].as<uint16_t>(), result[ARG_LOOPS].as<uint16_t>());

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

void console::progress_bar(double percent, const std::chrono::seconds& elapsed)
{
    constexpr auto progress_width = 80;
    static char progress_fill[progress_width] = { 0 };
    static char progress_none[progress_width] = { 0 };
    if (progress_fill[0] == 0)
    {
        memset(progress_fill, '=', sizeof(progress_fill));
        memset(progress_none, '-', sizeof(progress_none));
    }

    std::chrono::seconds eta = elapsed.count() > 0 ?
        std::chrono::seconds((uint64_t)(elapsed.count() / percent)) - elapsed : std::chrono::seconds(0);

    printf("[%.*s>", (int)(progress_width * percent), progress_fill);
    printf("%.*s]", (int)(progress_width * (1 - percent)), progress_none);
    printf("%d%%  %02lld:%02lld:%02lld   ETA:%02lld:%02lld:%02lld\t\n", (int)(percent * 100),
        elapsed.count() / 3600, (elapsed.count() / 60) % 60, elapsed.count() % 60,
        eta.count() / 3600, (eta.count() / 60) % 60, eta.count() % 60);
}


void console::clear_line(int width /*= 0*/)
{
    Term::Terminal term(false);

    int tWidth = 0;
    int tHeight = 0;

    term.get_term_size(tHeight, tWidth);
    printf("\r%*s", width > 0 ? width : (tWidth - 1), "");
    printf("\r");
}


int console::read_esc_press()
{
    Term::Terminal term(true);
    return term.read_key0() == Term::ESC;
}

namespace console::tests
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
        "--" ARG_LTYPE"=0,1,2,3,4",

        "--" ARG_WORDDICT"=0xCEB6AE48B5C63ED1,CEB6AE48B5C63ED2,0xCEB6AE48B5C63ED3,examples/dictionary.words",
        "--" ARG_BINDICT"=examples/dictionary.bin",
        "--" ARG_BINDMODE"=2",

        "--" ARG_START"=1",
        "--" ARG_COUNT"=0xFFFF",

        "--" ARG_ALPHABET"=61:62:63:64:zz:AB,examples/alphabet.bin",

        "--" ARG_IFILTER"=0x2", //SmartFilterFlags::Max6OnesInARow  other are very heavy, this one will allow all numbers less than 0x03FFFFFFFFFFFFFF
        "--" ARG_EFILTER"=64",  //SmartFilterFlags::BytesRepeat4


        "--" ARG_PATTERN"=0x01:*:0x43-0x10:0xA0-FF:AA|0x34|0xBB:0x66|0x77:AL0,0x88:asd:w1:88:*:AL2:BB:73",

        "--" ARG_FMATCH,
    };

    const char* help[] = {
        "skip",
        "-h"
    };

    CommandLineArgs args = parse_command_line(sizeof(help)/ sizeof(char*), help);
    args = parse_command_line(sizeof(commandline)/ sizeof(char*), commandline);

    return args;
}
}