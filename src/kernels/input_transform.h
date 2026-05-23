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

    // Inputs transformation: reverse the key bits before using them in decryption.
    RevKey = 1 << 0,

    // Inputs transformation: XOR the fixed part of the input with the seed value (if present) before using it in decryption.
    XorFix = 1 << 1,

    // Inputs transformation: XOR the hopping part of the input with the seed value (if present) before using it in decryption.
    XorHop = 1 << 2,

    // Inputs transformation: XOR the unencrypted part of the input with the seed value (if present) before using it in encryption.
    //      NOTE: In decryption does reverse, decrypted part is XORed to restore the original unencrypted value.
    XorDec_TODO = 1 << 3,

    // Combination of two XOR modes
    Xored = XorFix | XorHop,

    All = static_cast<uint8_t>(-1),
};

static constexpr uint8_t InputTransformMask = 0b111;
static constexpr uint8_t InputTransformVariantsCount = InputTransformMask + 1;

/** Every input transform as mask variation */
using EveryInputTransform = helpers::MakeTypedValuesSet<InputsTransform, std::make_index_sequence<InputTransformVariantsCount>>::type;


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
    case static_cast<uint8_t>(InputsTransform::XorHop):
        return "XorHop";
    case static_cast<uint8_t>(InputsTransform::RevKey | InputsTransform::XorFix):
        return "RevKey | XorFix";
    case static_cast<uint8_t>(InputsTransform::RevKey | InputsTransform::XorHop):
        return "RevKey | XorHop";
    case static_cast<uint8_t>(InputsTransform::XorFix | InputsTransform::XorHop):
        return "XorFix | XorHop";
    case static_cast<uint8_t>(InputsTransform::RevKey | InputsTransform::XorFix | InputsTransform::XorHop):
        return "RevKey | XorFix | XorHop";
    default:
        return "Unknown";
    }
}
