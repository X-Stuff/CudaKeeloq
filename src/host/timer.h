#pragma once

#include <chrono>

/**
 * Non-owning stopwatch: records `start_point` at construction, exposes elapsed helpers.
 * Template parameter picks the underlying clock (`std::chrono::steady_clock` is typical).
 */
template<typename TClock = std::chrono::high_resolution_clock>
struct Timer
{
    /** Start a new timer anchored at "now". */
    static inline Timer<TClock> start()
    {
        return Timer<TClock>(TClock::now());
    }

    /** Elapsed time since construction, cast to the requested duration type (default: ms). */
    template <typename TDuration = std::chrono::milliseconds>
    inline TDuration elapsed()
    {
        return std::chrono::duration_cast<TDuration>(TClock::now() - start_point);
    }

    /** Reset the timer and return the elapsed time. */
    template <typename TDuration = std::chrono::milliseconds>
    inline TDuration reset()
    {
        const auto elapsedTime = this->template elapsed<TDuration>();
        start_point = TClock::now();
        return elapsedTime;
    }

    /** Elapsed time in whole seconds. */
    inline std::chrono::seconds elapsedSeconds()
    {
        return elapsed<std::chrono::seconds>();
    }

protected:
    Timer(typename TClock::time_point initial) : start_point(initial)
    {
    }

private:

    typename TClock::time_point start_point;
};

/**
 * Scope-local variant of Timer: on destruction, writes the elapsed duration into `container`.
 * Useful for accumulating time inside a block without manual calls.
 */
template<typename TClock = std::chrono::high_resolution_clock, typename TDuration = std::chrono::milliseconds>
struct ScopeTimer : public Timer<TClock>
{
    ScopeTimer(TDuration* container) : Timer<TClock>(TClock::now()), container(container)
    {
    }

    ~ScopeTimer()
    {
        *container = this->template elapsed<TDuration>();
    }
private:
    TDuration* container;
};
