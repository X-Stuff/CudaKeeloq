#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

/**
 * Flags describing how captured OTA inputs are pre-processed before decryption.
 * Each flag represents a different interpretation of how the manufacturer might
 * have encoded the key or fixed data. Used as compile-time template parameters
 * in kernel launches for zero-overhead branching.
 */
enum class InputsTransform : uint8_t
{
    None = 0,
    RevKey = 1 << 0,
    XorFix = 1 << 1,

    All = static_cast<uint8_t>(-1),
};

static constexpr uint8_t InputTransformMask = 0b11;
static constexpr uint8_t InputTransformVariantsCount = InputTransformMask + 1;

__host__ __device__ __forceinline__ constexpr InputsTransform operator|(InputsTransform a, InputsTransform b)
{
    return static_cast<InputsTransform>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr InputsTransform operator&(InputsTransform a, InputsTransform b)
{
    return static_cast<InputsTransform>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

__host__ __device__ __forceinline__ constexpr bool operator!(InputsTransform a)
{
    return static_cast<uint8_t>(a) == 0;
}

__host__ __device__ __forceinline__ constexpr bool has_flag(InputsTransform value, InputsTransform flag)
{
    return static_cast<uint8_t>(value & flag) == static_cast<uint8_t>(flag);
}

__host__ __device__ __forceinline__ constexpr bool is_valid(InputsTransform value)
{
    return (static_cast<uint8_t>(value) & ~InputTransformMask) == 0;
}

constexpr auto name(InputsTransform transform) -> const char*
{
    switch (static_cast<uint8_t>(transform))
    {
    case static_cast<uint8_t>(InputsTransform::None):
        return "Normal";
    case static_cast<uint8_t>(InputsTransform::RevKey):
        return "RevKey";
    case static_cast<uint8_t>(InputsTransform::XorFix):
        return "XorFix";
    case static_cast<uint8_t>(InputsTransform::RevKey | InputsTransform::XorFix):
        return "RevKey | XorFix";
    default:
        return "Unknown";
    }
}
