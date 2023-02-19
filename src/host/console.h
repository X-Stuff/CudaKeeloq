#pragma once

#include "common.h"

#include <stdio.h>
#include <chrono>

#include "host/command_line_args.h"

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

namespace console
{

void progress_bar(double percent, const std::chrono::seconds& elapsed);

void clear_line(int width = 0);

int read_esc_press();

void set_cursor_state(bool visible);

uint32_t get_width();

}