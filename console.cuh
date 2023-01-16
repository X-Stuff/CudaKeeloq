#pragma once

#include <stdio.h>
#include <conio.h>
#include <unistd.h>

#define console_clear() printf("\033[H\033[J")

#define console_cursor_up(lines) printf("\033[%dA", (lines))

#define console_cursor_ret_up() printf("\033[F")
#define console_set_cursor(x,y) printf("\033[%d;%dH", (y), (x))

#define save_cursor_pos() printf("\033[s")
#define load_cursor_pos() printf("\033[u")


namespace console
{

struct Args
{
    std::vector<uint64_t> Inputdata;
};

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

}
