#include "host/command_line_args.h"

#include <cuda_runtime_api.h>

#include "host/console.h"
#include "host/host_utils.h"

#include "bruteforce/bruteforce_round.h"

//#define CXXOPTS_NO_EXCEPTIONS
#include "cxxopts/include/cxxopts.hpp"


namespace
{

// Small sink that routes warnings to stdout and errors to stderr, and respects
// the caller's verbosity preference. Errors also flip `has_errors` on the
// target so the caller can stop after parse() returns.
struct Reporter
{
    CommandLineArgs& target;
    CommandLineArgs::Verbosity verbosity;

    bool silent() const { return verbosity == CommandLineArgs::Verbosity::Silent; }

    template <typename... Args>
    void warn(const char* fmt, Args... args) const
    {
        emit(stdout, "WARNING: ", fmt, args...);
    }

    template <typename... Args>
    void error(const char* fmt, Args... args) const
    {
        target.has_errors = true;
        emit(stderr, "ERROR: ", fmt, args...);
    }

private:
    template <typename... Args>
    void emit(std::FILE* out, const char* prefix, const char* fmt, Args... args) const
    {
        if (silent())
        {
            return;
        }
        std::fputs(prefix, out);
        // Zero-arg path avoids GCC's -Wformat-security complaint about non-literal
        // formats; callers pass literals, but the template can't prove that.
        if constexpr (sizeof...(Args) == 0)
        {
            std::fputs(fmt, out);
        }
        else
        {
            std::fprintf(out, fmt, args...);
        }
        std::fputc('\n', out);
    }
};

inline void read_alphabets(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& report)
{
    if (result.count(ARG_ALPHABET) == 0)
    {
        // no alphabets available
        return;
    }

    const auto& alphabet_args = result[ARG_ALPHABET].as<std::vector<std::string>>();

    for (const auto& alphabet_arg : alphabet_args)
    {
        auto alphabet_bytes = host::utils::readAlphabetBinaryFile(alphabet_arg.c_str());
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
                    report.error("cannot parse alphabet byte '%s'!", hex.c_str());
                }
            }

            //
            if (alphabet_bytes.size() > 0)
            {
                target.alphabets.emplace_back(alphabet_bytes);

                if (alphabet_bytes.size() > 256)
                {
                    report.warn("Alphabet: '%s' has too many bytes in hex string: %zd (should be less than 257)",
                        result[ARG_ALPHABET].as<std::string>().c_str(), alphabet_bytes.size());
                }
            }
            else
            {
                report.error("Alphabet: '%s' is not valid file, neither valid alphabet hex string (like: AA:11:b3...)!",
                    alphabet_arg.c_str());
            }
        }
    }
}

inline void parse_dictionary_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& report)
{
    auto bruteConfigsBefore = target.brute_configs.size();

    if (result[ARG_WORDDICT].count() > 0)
    {
        // In case of dict is passed directly as values, seed may be passed as well
        const auto seed = result[ARG_SEED].as<uint32_t>();
        const bool seed_valid = result[ARG_SEED].count() > 0;

        auto dict = result[ARG_WORDDICT].as<std::vector<std::string>>();

        std::vector<Decryptor> decryptors;

        for (const auto& dict_arg : dict)
        {
            std::vector<Decryptor> from_file = host::utils::readWordDictionaryFile(dict_arg.c_str());

            if (from_file.size() > 0)
            {
                decryptors.insert(decryptors.end(), from_file.begin(), from_file.end());
            }
            else if (auto key = strtoull(dict_arg.c_str(), nullptr, 16))
            {
                decryptors.push_back(Decryptor::Make(key, seed, seed_valid));
            }
            else
            {
                report.error("invalid param passed to '--%s' argument: '%s' not a dictionary file neither word",
                    ARG_WORDDICT, dict_arg.c_str());
            }
        }

        if (decryptors.size() > 0)
        {
            target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
        }
    }

    if (result[ARG_BINDICT].count() > 0)
    {
        auto seed = result[ARG_SEED].as<uint32_t>();
        auto seed_valid = result[ARG_SEED].count() > 0;

        auto dicts = result[ARG_BINDICT].as<std::vector<std::string>>();
        auto mode = result[ARG_BINDMODE].as<uint8_t>();

        std::vector<Decryptor> decryptors;

        for (const auto& bin_dict_path : dicts)
        {
            std::vector<Decryptor> decryptors = host::utils::readBinaryDictionaryFile(bin_dict_path.c_str(), mode, (seed_valid ? &seed : nullptr));

            if (decryptors.size() > 0)
            {
                target.brute_configs.push_back(BruteforceConfig::GetDictionary(std::move(decryptors)));
            }
            else
            {
                report.error("invalid param passed to '--%s' argument: '%s'. Invalid file!",
                    ARG_BINDICT, bin_dict_path.c_str());
            }
        }
    }

    if (bruteConfigsBefore == target.brute_configs.size())
    {
        report.error("Cannot parse inputs to even single brute config! Check your --%s and --%s arguments",
            ARG_WORDDICT, ARG_BINDICT);
    }
}

inline void parse_bruteforce_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& /*report*/)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    auto seed_valid = result[ARG_SEED].count() > 0;

    Decryptor first_decryptor = Decryptor::Make(start_key, seed, seed_valid);

    auto count_key = result[ARG_COUNT].as<size_t>();

    target.brute_configs.push_back(BruteforceConfig::GetBruteforce(first_decryptor, count_key));
}

inline void parse_seed_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& report)
{
    if (result.count(ARG_START) == 0)
    {
        report.error("For seed mode, it's necessary to specify a manufacturer key with '--" ARG_START "' argument!");
        return;
    }

    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    constexpr auto seed_valid = true;

    Decryptor first_decryptor = Decryptor::Make(start_key, seed, seed_valid);

    target.brute_configs.push_back(BruteforceConfig::GetSeedBruteforce(first_decryptor));
}

inline void parse_bruteforce_filtered_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& /*report*/)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    auto seed_valid = result[ARG_SEED].count() > 0;

    Decryptor first_decryptor = Decryptor::Make(start_key, seed, seed_valid);

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

inline void parse_alphabet_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& /*report*/)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    auto seed_valid = result[ARG_SEED].count() > 0;

    Decryptor first_decryptor = Decryptor::Make(start_key, seed, seed_valid);

    auto count_key = result[ARG_COUNT].as<size_t>();

    for (const auto& alphabet : target.alphabets)
    {
        target.brute_configs.push_back(BruteforceConfig::GetAlphabet(first_decryptor, alphabet, count_key));
    }
}

inline void parse_pattern_mode(CommandLineArgs& target, cxxopts::ParseResult& result, const Reporter& report)
{
    auto start_key = result[ARG_START].as<uint64_t>();
    auto seed = result[ARG_SEED].as<uint32_t>();
    auto seed_valid = result[ARG_SEED].count() > 0;

    Decryptor first_decryptor = Decryptor::Make(start_key, seed, seed_valid);

    auto count_key = result[ARG_COUNT].as<size_t>();


    if (result.count(ARG_PATTERN) == 0)
    {
        report.error("For pattern mode, it's necessary to specify a pattern with '--" ARG_PATTERN "' argument!");
        return;
    }

    const auto& args = result[ARG_PATTERN].as<std::vector<std::string>>();

    auto full_bytes = DefaultByteArray<>::asVector<std::vector<uint8_t>>();

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
                    report.warn("Cannot use Alphabet in patterns - no provided alphabets. Replacing with *");
                    result.push_back(full_bytes);
                    continue;
                }

                auto al_index = strtoul(hex.substr(2).c_str(), nullptr, 10);
                if (al_index >= target.alphabets.size())
                {
                    report.warn("Argument %s referring alphabet: %ld (index: %ld), but there are only %zd available. Replacing with first one",
                        hex.c_str(), al_index + 1, al_index, target.alphabets.size());
                    al_index = 0;
                }

                result.push_back(target.alphabets[al_index].asVector());
            }
            else
            {
                // append regular pattern bytes
                auto bytes = BruteforcePattern::parseBytes(hex);

                if (bytes.size() == 0)
                {
                    report.warn("Invalid string '%s' for byte pattern! Ignoring. Replacing with *",
                        hex.c_str());
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
            report.warn("Pattern string: '%s' contains more than 8 per-bytes delimiters (%zd) other will be ignored.",
                pattern_arg.c_str(), bytes_hex.size());
        }
        else if (result.size() < 8)
        {
            report.warn("Pattern string: '%s' contains less than 8 per-bytes delimiters (%zd) other fill with full pattern.",
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
        " --" ARG_MODE "=1 --" ARG_START "=0x9876543210 --" ARG_COUNT "=0xFFFFFFFF"
        "\n\n\tThis will launch simple bruteforce (+1) attack with 2^32 checks, starting from 0x9876543210. "
        "Will check ALL 19 (14 if no seed specified) keeloq learning types with all modifications"

        "\nExample:\n"
        "\t./" APP_NAME " --" ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=3 --" ARG_LTYPE "=Simple --" ARG_ALPHABET "=examples/alphabet.bin,10:20:30:AA:BB:CC:DD:EE:FF:02:33"
        "\n\n\tThis will launch 2 alphabets attacks for all possible combinations for SIMPLE learning Keeloq type. "
        "First alphabet will be taken from file, second - parsed from inputs."

        "\nExample:\n"
        "\t./" APP_NAME " --"  ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=4 --" ARG_LTYPE "=Normal --" ARG_ALPHABET "=examples/alphabet.bin --" ARG_PATTERN "=AL0:11:AB|BC:*:00-44:AL0:AA-FF:01"
        "\n\n\tThis will launch pattern attacks with NORMAL keeloq learning type."
        "\n\tPattern applied 'as is' - big endian. The highest byte (0xXX.......) will be taken from 1st alphabet."
        "\n\tNext byte (0x..XX....) will be exact `0x11`."
        "\n\tNext byte (0x....XX..) will be `0xAB` or `0xBC`.\n"

        "\nExample:\n"
        "\t./" APP_NAME " --"  ARG_INPUTS " xxx,yy,zzz"
        " --" ARG_MODE "=5 --" ARG_START "=0xAABBCCDDEEFF"
        "\n\n\tThis will launch seed bruteforce attack for all seed learning types. "
        "\n\tSpecifying '--" ARG_LTYPE "=a,b,c' will narrow learning types to provided ones."
        ;
}

cxxopts::Options buildOptions()
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

)" "                                               version:" APP_VERSION_STRING);
    options.set_width(::console::getWidth())
        .allow_unrecognised_options()
        .add_options()
        ("h," ARG_HELP, "Prints this help")
        ("v," ARG_VERSION, "Prints version information")

        // What to bruteforce
        (ARG_INPUTS, "Comma separated uint64 values (it's better to have 3), hopping first: 0x<HOPPING_32><FIXED_32>",
            cxxopts::value<std::vector<uint64_t>>(), "[k1, k1, k3]")

        // CUDA Setup
        (ARG_BLOCKS, "How many thread blocks to launch, leave it 0 to calculate best value.",
            cxxopts::value<uint16_t>()->default_value("0"), "<num>")
        (ARG_THREADS, "How many threads will be launched in a block (if 0 - will use best value from device).",
            cxxopts::value<uint16_t>()->default_value("0"), "<num>")

#ifndef NO_INNER_LOOPS
        (ARG_LOOPS, "How many loop iterations will one thread perform (keep it low).",
            cxxopts::value<uint16_t>()->default_value("2"), "<num>")
#endif

        // Mode - what bruteforce type will be used (accepts index or name, case-insensitive)
        (ARG_MODE,
            "Bruteforce modes (comma separated, index or name):"
            "\n\t0 | Dictionary"
            "\n\t1 | Simple   (Simple +1)"
            "\n\t2 | Filtered (Simple +1 with filters)"
            "\n\t3 | Alphabet"
            "\n\t4 | Pattern"
            "\n\t5 | Seed",
            cxxopts::value<std::vector<std::string>>(), "[m1,m2..]")
        (ARG_LTYPE,
            "Comma separated specific learning type(s), if you know your target well. Increases approximately x16 times (since doesn't calculate other types):"
            "\n\t0: - Simple"
            "\n\t1: - Normal"
            "\n\t2: - Secure"
            "\n\t3: - Xor"
            "\n\t4: - FAAC"
            "\n\t5: - Serial1"
            "\n\t6: - Serial2"
            "\n\t7: - Serial3"
            "\n\tALL",
            cxxopts::value<std::vector<std::string>>()->default_value("ALL"), "<type>")

        (ARG_CHECKREV,
            "Check also byte-reversed man keys during bruteforce, some manufacturers mixes up or do this intentionally. "
            "You need this setting set to false only if you are doing, full 2^64 bruteforce.\n",
            cxxopts::value<bool>()->default_value("true"), "true|false")
        (ARG_NO_REGKEYS,
            "Disable regular keys checking during bruteforce. Set if to 'true' only if you want force check ONLY reversed keys.\n",
            cxxopts::value<bool>()->default_value("false"), "true|false")
        (ARG_CHECKINV,
            "Check also inverted algorithms during bruteforce, some manufacturers mixes up or do this intentionally (multiplies time x2). "
            "Affects only: Normal, Secure, FAAC learning types.\n",
            cxxopts::value<bool>()->default_value("true"), "true|false")
        (ARG_NO_NRMALGS,
            "Disable normal algorithms check. Set if to 'true' only if you want force check only other algorithms.",
            cxxopts::value<bool>()->default_value("false"), "true|false")

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

        // Benchmarks run
        (ARG_BENCHMARK, "Boolean. Run application benchmarks. You can specify learning and num loops type from command line also.",
            cxxopts::value<bool>()->default_value("false"))
        ;

    return options;
}

}

void CommandLineArgs::printHelp(std::FILE* out)
{
    auto options = buildOptions();
    std::fprintf(out, "\n%s\n", options.help().c_str());
    std::fprintf(out, "%s\n", Usage());
}

CommandLineArgs CommandLineArgs::parse(int argc, const char** argv, Verbosity verbosity)
{
    auto options = buildOptions();

    CommandLineArgs args;
    Reporter report{ args, verbosity };

    auto result = options.parse(argc, argv);

    args.print_version = result.count(ARG_VERSION) > 0;

    // benchmarks
    args.run_bench = result[ARG_BENCHMARK].as<bool>();

    // CUDA setup
    args.initCuda(result[ARG_BLOCKS].as<uint16_t>(), result[ARG_THREADS].as<uint16_t>(),
        result.count(ARG_LOOPS) > 0 ? result[ARG_LOOPS].as<uint16_t>() : 1);

    if (result.count(ARG_HELP) || result.arguments().size() == 0 || result.count(ARG_INPUTS) == 0 || args.print_version)
    {
        args.has_errors = result.arguments().size() == 0 || result.count(ARG_INPUTS) == 0;

        // Caller decides whether to show usage (app: yes; tests: no).
        args.print_help = !args.run_bench && !args.print_version;
        return args;
    }

    // Inputs
    if (result.count(ARG_INPUTS) > 0)
    {
        args.initInputs(result[ARG_INPUTS].as<std::vector<uint64_t>>());
        if (args.inputs.size() < 3)
        {
            report.warn("Not enough inputs: '%zd'! Need at least 3! However we'll proceed...",
                args.inputs.size());
        }

        if (args.inputs.size() > 0)
        {
            auto fix = args.inputs[0].fix();

            for (size_t i = 1; i < args.inputs.size(); ++i)
            {
                if (args.inputs[i].fix() != fix)
                {
                    report.warn("Invalid input at index:%zu (0x%016" PRIX64 ") fixed code doesn't match first input!",
                        i, args.inputs[i].ota);
                }
            }
        }
    }
    else
    {
        report.error("No inputs! Nothing to brute!");
        args.print_help = true;
        return args;
    }

    // Stop if need
    args.match_stop = result[ARG_FMATCH].as<bool>();

    // Alphabets
    read_alphabets(args, result, report);

    // Bruteforce configs
    if (result.count(ARG_MODE) > 0)
    {
        for (const auto& modeStr : result[ARG_MODE].as<std::vector<std::string>>())
        {
            BruteforceType::Type mode;
            if (!BruteforceType::parse(modeStr.c_str(), mode))
            {
                report.error("Unknown bruteforce mode: '%s'. Expected index (0-5) or name (Dictionary, Simple, Filtered, Alphabet, Pattern, Seed).",
                    modeStr.c_str());
                return args;
            }

            switch (mode)
            {
            case (uint8_t)BruteforceType::Dictionary:
                parse_dictionary_mode(args, result, report);
                break;
            case (uint8_t)BruteforceType::Simple:
                parse_bruteforce_mode(args, result, report);
                break;
            case (uint8_t)BruteforceType::Filtered:
                parse_bruteforce_filtered_mode(args, result, report);
                break;
            case (uint8_t)BruteforceType::Alphabet:
                parse_alphabet_mode(args, result, report);
                break;
            case (uint8_t)BruteforceType::Pattern:
                parse_pattern_mode(args, result, report);
                break;
            case (uint8_t)BruteforceType::Seed:
                parse_seed_mode(args, result, report);
                break;
            default:
                break;
            }

            if (args.has_errors)
            {
                return args;
            }
        }

        if (args.brute_configs.size() == 0)
        {
            report.error("Cannot parse inputs to even single brute config! Result arguments:\n'%s'",
                result.arguments_string().c_str());
            return args;
        }
    }
    else
    {
        report.error("You need to specify bruteforce mode!");
        args.print_help = true;
        return args;
    }

    // Learning Types if any specific selected
    if (result.count(ARG_LTYPE) > 0)
    {
        auto learning_type_names = result[ARG_LTYPE].as<std::vector<std::string>>();

        args.selected_learning.clear();
        for (const auto& name : learning_type_names)
        {
            KeeloqLearning::LearningType value;
            if (KeeloqLearning::parse(name.c_str(), value))
            {
                args.selected_learning.push_back(static_cast<KeeloqLearning::LearningType>(value));
            }
        }
    }

    // Modifications
    {
        // By default we check normal keys
        if (!result.count(ARG_NO_REGKEYS) || !result[ARG_NO_REGKEYS].as<bool>())
        {
            args.selected_input_mods.push_back(KeeloqLearning::Modifier::Input::Normal);
        }

        if (result[ARG_CHECKREV].as<bool>())
        {
            args.selected_input_mods.push_back(KeeloqLearning::Modifier::Input::ReversedKey);
        }

        // By default we use normal algos
        if (!result.count(ARG_NO_NRMALGS) || !result[ARG_NO_NRMALGS].as<bool>())
        {
            args.selected_algo_mods.push_back(KeeloqLearning::Modifier::Algo::Normal);
        }

        if (result[ARG_CHECKINV].as<bool>())
        {
            args.selected_algo_mods.push_back(KeeloqLearning::Modifier::Algo::Inverted);
        }
    }

    return args;
}

bool CommandLineArgs::canBruteforce()
{
    return inputs.size() > 0 && brute_configs.size() > 0;
}

void CommandLineArgs::initInputs(const std::vector<uint64_t>& inp)
 {
     inputs.reserve(inp.size());
     for (uint64_t ota : inp)
     {
        inputs.push_back(EncParcel(ota));
     }
 }

void CommandLineArgs::initCuda(uint16_t blocks, uint16_t threads, uint8_t numSubSteps)
{
    auto optimal = CudaConfig::Optimal();

    cuda_loops      = numSubSteps;
    cuda_threads    = threads ? threads : optimal.threads;
    cuda_blocks     = blocks ? blocks : optimal.blocks;
 }
