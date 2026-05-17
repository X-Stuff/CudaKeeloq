#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

/**
 * Flags describing how captured OTA inputs are pre-processed before decryption.
 * Each flag represents a different interpretation of how the manufacturer might
 * have encoded the key or fixed data. Used as compile-time template parameters
 * in kernel launches for zero-overhead branching.
 */
enum class InputTransform : uint8_t
{
    None = 0,
    RevKey = 1 << 0,
    XorFix = 1 << 1,

    All = static_cast<uint8_t>(-1),
};

static constexpr uint8_t InputTransformMask = 0b11;
static constexpr uint8_t InputTransformVariantsCount = InputTransformMask + 1;

__host__ __device__ __forceinline__ constexpr InputTransform operator|(InputTransform a, InputTransform b)
{
    return static_cast<InputTransform>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr InputTransform operator&(InputTransform a, InputTransform b)
{
    return static_cast<InputTransform>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr bool operator!(InputTransform a)
{
    return static_cast<uint8_t>(a) == 0;
}

__host__ __device__ __forceinline__ constexpr bool has_flag(InputTransform value, InputTransform flag)
{
    return static_cast<uint8_t>(value & flag) == static_cast<uint8_t>(flag);
}

__host__ __device__ __forceinline__ constexpr bool is_valid(InputTransform value)
{
    return (static_cast<uint8_t>(value) & ~InputTransformMask) == 0;
}

constexpr auto name(InputTransform transform) -> const char*
{
    switch (static_cast<uint8_t>(transform))
    {
    case static_cast<uint8_t>(InputTransform::None):
        return "Normal";
    case static_cast<uint8_t>(InputTransform::RevKey):
        return "RevKey";
    case static_cast<uint8_t>(InputTransform::XorFix):
        return "XorFix";
    case static_cast<uint8_t>(InputTransform::RevKey | InputTransform::XorFix):
        return "RevKey | XorFix";
    default:
        return "Unknown";
    }
}
