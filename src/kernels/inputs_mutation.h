#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

enum class InputsMutation : uint8_t
{
    None = 0,
    RevKey = 1 << 0,
    XorFix = 1 << 1,

    All = static_cast<uint8_t>(-1),
};

static constexpr uint8_t InputsMutationMask = 0b11;
static constexpr uint8_t InputsMutationVariantsCount = InputsMutationMask + 1;

__host__ __device__ __forceinline__ constexpr InputsMutation operator|(InputsMutation a, InputsMutation b)
{
    return static_cast<InputsMutation>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr InputsMutation operator&(InputsMutation a, InputsMutation b)
{
    return static_cast<InputsMutation>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr bool operator!(InputsMutation a)
{
    return static_cast<uint8_t>(a) == 0;
}

__host__ __device__ __forceinline__ constexpr bool has_flag(InputsMutation value, InputsMutation flag)
{
    return static_cast<uint8_t>(value & flag) == static_cast<uint8_t>(flag);
}

__host__ __device__ __forceinline__ constexpr bool is_valid(InputsMutation value)
{
    return (static_cast<uint8_t>(value) & ~InputsMutationMask) == 0;
}

constexpr auto name(InputsMutation mutation) -> const char*
{
    switch (mutation)
    {
    case InputsMutation::None:
        return "Normal";
    case InputsMutation::RevKey:
        return "RevKey";
    case InputsMutation::XorFix:
        return "XorFix";
    case InputsMutation::RevKey | InputsMutation::XorFix:
        return "RevKey | XorFix";
    default:
        return "Unknown";
    }
}
