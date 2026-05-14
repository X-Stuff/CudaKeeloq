#include "doctest/doctest.h"

#include <filesystem>
#include <fstream>
#include <cstddef>

#include "host/command_line_args.h"


namespace
{
template <size_t N>
CommandLineArgs parseArgs(const char* (&argv)[N], AppVerbosity verbosity = AppVerbosity::Silent)
{
    return CommandLineArgs::parse(static_cast<int>(N), argv, verbosity);
}
CommandLineArgs parseArgs(std::vector<const char*> argv, AppVerbosity verbosity = AppVerbosity::Silent)
{
    return CommandLineArgs::parse(static_cast<int>(argv.size()), argv.data(), verbosity);
}
}

TEST_CASE("cli: --help flags help but parses silently")
{
    const char* argv[] = { APP_NAME, "-h" };
    auto args = parseArgs(argv);

    CHECK(args.print_help);
    CHECK(args.brute_configs.empty());
    CHECK_FALSE(args.run_bench);
}

TEST_CASE("cli: seed mode without a start key is a hard error")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_BLOCKS "=32",
        "--" ARG_MODE   "=5",
    };

    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.brute_configs.empty());
}

TEST_CASE("cli: full invocation populates alphabets, modes, and run flags")
{
    auto temp_bin_dict = std::filesystem::temp_directory_path() / "test.bin";

    std::ofstream temp_file(temp_bin_dict, std::ios::out | std::ios::binary);
    CHECK(temp_file.is_open());

    auto data = 0xAABBCCDDEEFF0011;
    temp_file.write(reinterpret_cast<const char*>(&data), sizeof(data));
    temp_file.close();

    auto binDictArg = str::format<std::string>("--" ARG_BINDICT   "=%s", temp_bin_dict.string().c_str());

    std::vector<const char*> argv =
    {
        APP_NAME,
        "--" ARG_INPUTS    "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_BLOCKS    "=32",
        "--" ARG_THREADS   "=32",
        "--" ARG_LOOPS     "=4",
        "--" ARG_MODE      "=0,1,2,3,4,5",
        "--" ARG_LTYPE     "=0,1,2,3,4",

        "--" ARG_WORDDICT  "=0xFDE4531BBACAD12,FDE4531BBACAD13,0xFDE4531BBACAD14,examples/dictionary.words",
        binDictArg.c_str(),
        "--" ARG_BINDMODE  "=2",

        "--" ARG_START     "=1",
        "--" ARG_SEED      "=777",
        "--" ARG_COUNT     "=0xFFFF",

        "--" ARG_ALPHABET  "=61:62:63:64:54:AB,examples/alphabet.bin",

        "--" ARG_IFILTER   "=0x2",
        "--" ARG_EFILTER   "=64",

        "--" ARG_PATTERN   "=0x01:*:0x43-0x10:0xA0-FF:AA|0x34|0xBB:0x66|0x77:AL0,0x88:ingored:w1:88:*:AL2:BB:73",

        "--" ARG_FMATCH,
        "--" ARG_BENCHMARK "=1",
    };

    auto args = parseArgs(argv);
    CHECK_FALSE(args.has_errors);

    CHECK(args.alphabets.size() == 2);
    CHECK(args.brute_configs.size() == 9);
    CHECK(args.match_stop);
    CHECK(args.run_bench);
    CHECK(args.inputs.size() == 3);

    std::filesystem::remove(temp_bin_dict);
}

// ---------------------------------------------------------------------------
// Hard-error paths: every report.error() site in command_line_args.cpp.
// parse() is expected to flip has_errors and stop producing further results
// (subsequent options may still be applied best-effort, but the caller must
// treat the invocation as failed).
// ---------------------------------------------------------------------------

TEST_CASE("cli error: no --inputs")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_MODE "=1",
        "--" ARG_START "=0x1",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.print_help);
}

TEST_CASE("cli error: no --mode")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.print_help);
    CHECK(args.brute_configs.empty());
}

TEST_CASE("cli error: seed mode without --start")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE "=5",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.brute_configs.empty());
}

TEST_CASE("cli error: alphabet argument is neither a file nor a valid hex string")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS   "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE     "=3",
        "--" ARG_ALPHABET "=/nonexistent/path/does-not-exist.bin",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
}

TEST_CASE("cli error: word-dict argument is neither a file nor a hex word")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS   "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE     "=0",
        "--" ARG_WORDDICT "=zzz-not-a-word-nor-a-file",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
}

TEST_CASE("cli error: bin-dict argument points to an invalid file")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS  "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE    "=0",
        "--" ARG_BINDICT "=/nonexistent/path/does-not-exist.bin",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
}

TEST_CASE("cli error: unknown mode produces no bruteforce config")
{
    // mode 99 doesn't match any BruteforceType → default: break → loop finishes
    // with zero configs → final "Cannot parse inputs to even single brute config" error.
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE   "=99",
    };
    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.brute_configs.empty());
}

TEST_CASE("cli: --mode accepts bruteforce-type names as well as numeric indices")
{
    // Numeric form was already exercised elsewhere; here we assert names
    // resolve to the same mode set, case-insensitive and mixed with indices.
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS     "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_START      "=0xAABBCCDD00112233",
        "--" ARG_MODE       "=dictionary,Simple,FILTERED,Pattern,5",  // dict + simple + filtered + pattern + seed
        "--" ARG_PATTERN    "=0x01:*:",
        "--" ARG_WORDDICT   "=0x1122334455667788",
    };

    auto args = parseArgs(argv);
    CHECK_FALSE(args.has_errors);
    CHECK(args.brute_configs.size() == 5);
}

TEST_CASE("cli: --mode rejects unknown bruteforce-type names")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE   "=not-a-mode",
    };

    auto args = parseArgs(argv);
    CHECK(args.has_errors);
    CHECK(args.brute_configs.empty());
}

TEST_CASE("cli: --learning-type accepts learning names as well as numeric indices")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE   "=simple",
        "--" ARG_START  "=0x1",
        "--" ARG_LTYPE  "=Simple,normal,SECURE,3",  // names (mixed case) + numeric index for Xor
    };

    auto args = parseArgs(argv);
    CHECK_FALSE(args.has_errors);

    REQUIRE(args.selected_learning.size() == 4);
    CHECK(args.selected_learning[0] == KeeloqLearning::LearningType::Simple);
    CHECK(args.selected_learning[1] == KeeloqLearning::LearningType::Normal);
    CHECK(args.selected_learning[2] == KeeloqLearning::LearningType::Secure);
    CHECK(args.selected_learning[3] == KeeloqLearning::LearningType::Xor);
}

TEST_CASE("cli: mode-5 variation selects only the single chosen input/algo modifier")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS    "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE      "=5",
        "--" ARG_START      "=0xAABBCCDD00112233",
        "--" ARG_CHECKINV  "=true",
        "--" ARG_CHECKREV  "=true",
        "--" ARG_NO_NRMALGS"=true",
    };

    auto args = parseArgs(argv);
    CHECK_FALSE(args.has_errors);

    REQUIRE(args.selected_algo_mods.size() == 1);
    CHECK(args.selected_algo_mods[0] == KeeloqLearning::Modifier::Algo::Inverted);
    CHECK(args.inputsMutation == InputsMutation::RevKey);
    REQUIRE(args.brute_configs.size() == 1);
    CHECK(args.brute_configs[0].hasMutation(InputsMutation::RevKey));
}

TEST_CASE("cli: --check-xorfix enables every selected input mutation permutation")
{
    const char* argv[] = {
        APP_NAME,
        "--" ARG_INPUTS      "=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_MODE        "=simple",
        "--" ARG_START       "=0x1",
        "--" ARG_CHECKREV    "=true",
        "--" ARG_CHECKXORFIX "=true",
    };

    auto args = parseArgs(argv);
    CHECK_FALSE(args.has_errors);

    const auto allMutations = InputsMutation::RevKey | InputsMutation::XorFix;
    CHECK(args.inputsMutation == allMutations);
    REQUIRE(args.brute_configs.size() == 1);
    CHECK(args.brute_configs[0].hasMutation(InputsMutation::None));
    CHECK(args.brute_configs[0].hasMutation(InputsMutation::RevKey));
    CHECK(args.brute_configs[0].hasMutation(InputsMutation::XorFix));
    CHECK(args.brute_configs[0].hasMutation(allMutations));
}
