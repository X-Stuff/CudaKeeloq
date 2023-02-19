#include "test_console.h"

#include "host/command_line_args.h"
#include "host/console.h"

namespace
{
void print_args(const char** args, size_t num)
{
    printf("\n\t*** TESTING COMMAND LINE:\n");
    for (size_t i = 0; i < num; ++i)
    {
        printf("%s ", args[i]);
    }
    printf("\n\n");
}
}


CommandLineArgs tests::console::run()
{
    const char* commandline[] = {
        "CudaKeeloq.exe",
        "--" ARG_INPUTS"=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504",
        "--" ARG_BLOCKS"=32",
        "--" ARG_THREADS"=32",
        "--" ARG_LOOPS"=4",
        "--" ARG_MODE"=0,1,2,3,4,5",
        "--" ARG_LTYPE"=0,1,2,3,4",

        "--" ARG_WORDDICT"=0xFDE4531BBACAD12,FDE4531BBACAD13,0xFDE4531BBACAD14,examples/dictionary.words",
        "--" ARG_BINDICT"=examples/dictionary.bin",
        "--" ARG_BINDMODE"=2",

        "--" ARG_START"=1",
        "--" ARG_COUNT"=0xFFFF",

        "--" ARG_ALPHABET"=61:62:63:64:zz:AB,examples/alphabet.bin",

        "--" ARG_IFILTER"=0x2", //SmartFilterFlags::Max6OnesInARow  other are very heavy, this one will allow all numbers less than 0x03FFFFFFFFFFFFFF
        "--" ARG_EFILTER"=64",  //SmartFilterFlags::BytesRepeat4

        "--" ARG_PATTERN"=0x01:*:0x43-0x10:0xA0-FF:AA|0x34|0xBB:0x66|0x77:AL0,0x88:asd:w1:88:*:AL2:BB:73",

        "--" ARG_FMATCH,
        "--" ARG_BENCHMARK "=1",
        "--" ARG_TEST "=true",
    };

    const char* help[] = {
        "CudaKeeloq.exe",
        "-h"
    };

    // Print Help
    print_args(help, sizeof(help) / sizeof(char*));
    CommandLineArgs args = CommandLineArgs::parse(sizeof(help) / sizeof(char*), help);

    print_args(commandline, sizeof(commandline) / sizeof(char*));
    args = CommandLineArgs::parse(sizeof(commandline) / sizeof(char*), commandline);

    assert(args.alphabets.size() == 2);
    assert(args.brute_configs.size() == 8);

    assert(args.match_stop);
    assert(args.run_bench);
    assert(args.run_tests);

    return args;
}
