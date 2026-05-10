#pragma once

#include <chrono>
#include <cstdio>

#include "common.h"

#include "host/command_line_args.h"


constexpr char WAIT_SPIN[] = "|/-\\";
#define WAIT_CHAR(i) (WAIT_SPIN[i % (sizeof(WAIT_SPIN) - 1)])

#define CONSOLE_WIDTH 180

#define console_clear() printf("\033[H\033[J")
#define console_clear_line() printf("\033[2K\r")

#define console_cursor_up(lines) printf("\033[%dA", (lines))
#define console_set_width(col) printf("\033[%du", (col))

#define console_cursor_ret_up(lines) printf("\033[%dF", (lines))
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))

#define save_cursor_pos() printf("\033[s")
#define load_cursor_pos() printf("\033[u")

/**
 * Thin terminal helpers used by the interactive progress UI.
 * Backed by cpp-terminal; printf-based escape sequences are exposed as macros above.
 */
namespace console
{

/** Render a 0..1 progress bar with elapsed/ETA time annotation. */
void progressBar(double percent, const std::chrono::seconds& elapsed);

/** Clear the current terminal line (full width by default). */
void clearLine(int width = 0);

/** Move the cursor up `numlines` and clear each line in that range. */
void clearLinesUp(int numlines, int width = 0);

/** Poll for an Escape key press; returns non-zero if pressed since last call. */
int readEscPress();

/** Show or hide the terminal cursor. */
void setCursorState(bool visible);

/** Current terminal width in columns. */
uint32_t getWidth();

/** RAII cursor hiding */
struct ScopedHideCursor
{
    ScopedHideCursor() { console::setCursorState(false); }
    ~ScopedHideCursor() { console::setCursorState(true); }
};

}
