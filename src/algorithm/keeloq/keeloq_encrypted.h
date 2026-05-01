#pragma once

#include "common.h"

#include "device/cuda_common.h"


/**
 *  struct for convenience
 * Represents sent over the air data
 * Since from engineering perspective normal byte (bit) order is big endian
 * In order to get fixed and hopping codes OTA has to be bit-reversed
 */
struct EncParcel
{
    // Raw data transmitted over the air
    uint64_t ota = 0;

    __device__ __host__ constexpr EncParcel() : ota(0), fixed(0), hopping(0) { }

    __device__ __host__ EncParcel(uint64_t atad) : ota(atad)
    {
        uint64_t data = misc::rev_bits(atad);

        fixed = (uint32_t)(data >> 32);
        hopping = (uint32_t)(data);
    }

    __device__ __host__ EncParcel(uint32_t fix, uint32_t hop) : fixed(fix), hopping(hop)
    {
        auto rev_hop = misc::rev_bits(hop);
        auto rev_fix = misc::rev_bits(fix);

        ota = rev_hop | (rev_fix >> 32);
    }

    // Fixed code in parcel
    __device__ __host__ inline uint32_t fix() const { return fixed; }

    // hopping code in parcel
    __device__ __host__ inline uint32_t hop() const { return hopping; }

    // first 10 bits of fixed code - serial (can be used in decryption)
    __device__ __host__ inline uint32_t srl() const { return fixed & 0x3FF; }

    // last 4 bits of fixed code - button (can be used in decryption)
    __device__ __host__ inline uint32_t btn() const { return fixed >> 28; }

private:

    // Fixed part of the parcel ( 28-bit serial | 4-bit button )
    uint32_t fixed = 0;

    // Encrypted hopping code ( keeloq encrypted serial, button, counter )
    uint32_t hopping = 0;
};