#pragma once

#include <chrono>

// Simple helper timer
template<typename TClock = std::chrono::high_resolution_clock>
struct Timer
{
    static inline Timer<TClock> start()
    {
        return Timer<TClock>(TClock::now());
    }

    template <typename TDuration = std::chrono::milliseconds>
    inline TDuration elapsed()
    {
        return std::chrono::duration_cast<TDuration>(TClock::now() - start_point);
    }

    inline std::chrono::seconds elapsed_secods()
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