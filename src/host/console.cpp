#include "console.h"

#include <cstring>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif
    #include "cpp-terminal/terminal_base.h"
    #include "cpp-terminal/terminal.h"
#ifdef _MSC_VER
    #pragma warning(pop)
#endif

static Term::Terminal s_term = Term::Terminal(true, false);


void console::progress_bar(double percent, const std::chrono::seconds& elapsed)
{
    constexpr auto progress_width = 80;

    static char progress_fill[progress_width] = { 0 };
    static char progress_none[progress_width] = { 0 };
    if (progress_fill[0] == 0)
    {
        std::memset(progress_fill, '=', sizeof(progress_fill));
        std::memset(progress_none, '-', sizeof(progress_none));
    }

    std::chrono::seconds eta = elapsed.count() > 0 ?
        std::chrono::seconds((uint64_t)(elapsed.count() / percent)) - elapsed : std::chrono::seconds(0);

    printf("[%.*s>", (int)(progress_width * percent), progress_fill);
    printf("%.*s]", (int)(progress_width * (1 - percent)), progress_none);
    printf("%d%%  %02" PRId64 ":%02" PRId64 ":%02" PRId64 "   ETA:%02" PRId64 ":%02" PRId64 ":%02" PRId64 "   \n", (int)(percent * 100),
        elapsed.count() / 3600, (elapsed.count() / 60) % 60, elapsed.count() % 60,
        eta.count() / 3600, (eta.count() / 60) % 60, eta.count() % 60);
}

void console::clear_line(int width /*= 0*/)
{
    int tWidth = 0;
    int tHeight = 0;

    s_term.get_term_size(tHeight, tWidth);
    printf("\r%*s", width > 0 ? width : (tWidth - 1), "");
    printf("\r");
}


int console::read_esc_press()
{
    return s_term.read_key0() == Term::ESC;
}


void console::set_cursor_state(bool visible)
{
    s_term.write(visible ? Term::cursor_on() : Term::cursor_off());
}

uint32_t console::get_width()
{
    int cols = CONSOLE_WIDTH;
    int rows = 0;

    s_term.get_term_size(rows, cols);

    return cols;
}
