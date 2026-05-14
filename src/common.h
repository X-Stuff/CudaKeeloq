#pragma once

#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>


/**
 *  Project-wide macros, compile-time constants and tiny utility templates.
 * This header is included by host, device, and kernel translation units alike.
 */

#define _CUDA_CHECK(e, ret) \
    if (e != 0) { printf("\nASSERTION FAILED. CUDA ERROR!\n%s: %s\n", cudaGetErrorName((cudaError_t)e), cudaGetErrorString((cudaError_t)e)); assert(e == cudaSuccess); ret; }

// Argument-count dispatch helpers
#define CUDA_CHECK(e) _CUDA_CHECK(e, (void)0)
#define CUDA_CHECK_RETURN(e, r) _CUDA_CHECK(e, return r)


#if _DEBUG
    #if __CUDA_ARCH__
        #define assertf(cond, fmt, ...) if (!(cond)) { printf(fmt, __VA_ARGS__); __trap(); }
    #else
        #define assertf(cond, fmt, ...) if (!(cond)) { printf("\n"); printf(fmt, __VA_ARGS__); printf("\n"); assert(cond && fmt && " *** details in console *** "); }
    #endif
#else
    #define assertf(...)
#endif


#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif // !WARP_SIZE


/************************************************************************/
/*                  Bruteforce generator helper macros                  */
/************************************************************************/

#ifndef KERNEL_LAUNCH_BOUNDS_MAX_THREADS
    #define KERNEL_LAUNCH_BOUNDS_MAX_THREADS 1024
#endif

#define KERNEL_LAUNCH_BOUNDS __launch_bounds__(KERNEL_LAUNCH_BOUNDS_MAX_THREADS, 1)

#define GENERATOR_KERNEL_NAME(name) \
    Kernel_##name

#define GENERATOR_KERNEL_GETTER_NAME(name) \
    GetKernel_##name

#define DEFINE_GENERATOR_KERNEL(name, ...) \
    KERNEL_LAUNCH_BOUNDS GENERATOR_KERNEL_NAME(name)(__VA_ARGS__)

#define DEFINE_GENERATOR_GETTER(name) \
    extern "C" void* GENERATOR_KERNEL_GETTER_NAME(name)() { return (void*)&Kernel_##name; }


/************************************************************************/
/*                  Macros for common .inl files                        */
/************************************************************************/
#if __CUDA_ARCH__
    #define NOINLINE __noinline__
    #define UNROLL #pragma unroll
#else
    #define NOINLINE                                    /* nothing in c++ */
    #define UNROLL
#endif


#ifndef NO_INNER_LOOPS
    #define NO_INNER_LOOPS 1
#endif // !NO_INNER_LOOPS


#define APP_VERSION_MAJOR 0
#define APP_VERSION_MINOR 2
#define APP_VERSION_PATCH 0

// Helper macros to turn numbers into string literals
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define APP_VERSION_STRING STR(APP_VERSION_MAJOR) "." STR(APP_VERSION_MINOR) "." STR(APP_VERSION_PATCH)
static constexpr auto AppVersion = APP_VERSION_STRING;

enum AppVerbosity : uint8_t
{
    Debug = 0,

    Info = 1,

    Progress = 2,

    Warning = 4,

    Error = 5,

    Silent = 255,
};

#define APP_LOG_DEBUG(s, format, ...)       if (s <= AppVerbosity::Debug) { printf(format, ##__VA_ARGS__); }
#define APP_LOG_INFO(s, format, ...)        if (s <= AppVerbosity::Info) { printf(format, ##__VA_ARGS__); }
#define APP_LOG_PROGRESS(s, format, ...)    if (s <= AppVerbosity::Progress) { printf(format, ##__VA_ARGS__); }
#define APP_LOG_WARNING(s, format, ...)    if (s <= AppVerbosity::Warning) { printf(format, ##__VA_ARGS__); }
#define APP_LOG_ERROR(s, format, ...)       if (s <= AppVerbosity::Error) { printf(format, ##__VA_ARGS__); }

/**
 * Compile-time identity byte array — element[i] == i.
 * Primarily used to seed "accept every byte" alphabets without hand-writing the 0..255 list.
 */
template<uint8_t NSize = 255>
struct DefaultByteArray
{
    uint8_t element[NSize];

    constexpr DefaultByteArray() : element()
    {
        for (uint8_t i = 0; i < NSize; ++i)
        {
            element[i] = i;
        }
    }

    /** Materialise the array as a container (e.g. `std::vector<uint8_t>`). */
    template<typename Vector>
    static inline Vector asVector()
    {
        constexpr DefaultByteArray array = DefaultByteArray();
        return Vector(&array.element[0], &array.element[0] + NSize);
    }
};


namespace str
{

/** printf-style formatting that returns a `std::string` (or any string-like type). */
template<typename String, typename ... Args>
inline String format(const String& format, Args ... args)
{
    int size_s = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        return {};
    }

    auto size = static_cast<size_t>(size_s);
    String result(size, '\0');

    snprintf(&result[0], size, format.c_str(), args ...);
    result.pop_back(); // remove '\0' from the end

    return result;
}

}

/** UDL that packs up to 8 ASCII characters into a uint64_t (big-endian byte order). */
constexpr inline uint64_t operator "" _u64(const char* ascii, size_t num)
{
    const uint8_t size = sizeof(uint64_t);

    uint64_t number = 0;

    for (uint8_t i = 0; i < size; ++i)
    {
        number = (number << 8) | (num > i ? ascii[i] : 0);
    }

    return number;
}
