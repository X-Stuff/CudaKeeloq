#pragma once

#include "common.h"

#include <stdio.h>
#include <chrono>

#include "host/command_line_args.h"

#define CXXOPTS_NO_EXCEPTIONS
#include "cxxopts/include/cxxopts.hpp"


constexpr char WAIT_SPIN[] = "|/-\\";
#define WAIT_CHAR(i) (WAIT_SPIN[i % (sizeof(WAIT_SPIN) - 1)])

#define CONSOLE_WIDTH 160

#define console_clear() printf("\033[H\033[J")

#define console_cursor_up(lines) printf("\033[%dA", (lines))
#define console_set_width(col) printf("\033[%du", (col))

#define console_cursor_ret_up(lines) printf("\033[%dF", (lines))
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))

#define save_cursor_pos() printf("\033[s")
#define load_cursor_pos() printf("\033[u")

#define APP_NAME "CudaKeeloq"

#define ARG_HELP "help"
#define ARG_TEST "test"
#define ARG_BENCHMARK "benchmark"
#define ARG_INPUTS "inputs"
#define ARG_BLOCKS "cuda-blocks"
#define ARG_THREADS "cuda-threads"
#define ARG_LOOPS "cuda-loops"
#define ARG_MODE "mode"
#define ARG_LTYPE "learning-type"
#define ARG_WORDDICT "word-dict"
#define ARG_BINDICT "bin-dict"
#define ARG_BINDMODE "bin-dict-mode"
#define ARG_START "start"
#define ARG_COUNT "count"
#define ARG_ALPHABET "alphabet"
#define ARG_PATTERN "pattern"
#define ARG_IFILTER "include-filter"
#define ARG_EFILTER "exclude-filter"
#define ARG_FMATCH "first-match"

namespace console
{

CommandLineArgs parse_command_line(int argc, const char** argv);

void progress_bar(double percent, const std::chrono::seconds& elapsed);

void clear_line(int width = 0);

int read_esc_press();

void set_cursor_state(bool visible);

}