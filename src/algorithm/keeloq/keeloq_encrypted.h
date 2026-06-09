#pragma once

#include "common.h"

#include "device/cuda_common.h"


/**
 * An over-the-air parcel as transmitted by a keeloq remote.
 *
 * The wire format is bit-reversed relative to the engineering-friendly big-endian
 * representation, so this struct stores both views (`ota` and the fixed/hopping halves).
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

    /** Fixed code portion of the parcel. */
    __device__ __host__ inline uint32_t fix() const { return fixed; }

    /** Hopping (encrypted) code portion of the parcel. */
    __device__ __host__ inline uint32_t hop() const { return hopping; }

    /** 10-bit serial extracted from the fixed code (used in some learning algorithms). */
    __device__ __host__ inline uint32_t srl() const { return fixed & 0x3FF; }

    /** 4-bit button id extracted from the fixed code (used in some learning algorithms). */
    __device__ __host__ inline uint32_t btn() const { return fixed >> 28; }

    /** 28-bit serial extracted from the fixed part of OTA. In hopping only 10 bit of serial is encoded */
    __device__ __host__ inline uint32_t serial() const { return fixed & 0x0FFFFFFF; }

private:

    // Fixed part of the parcel ( 28-bit serial | 4-bit button )
    uint32_t fixed = 0;

    // Encrypted hopping code ( keeloq encrypted serial, button, counter )
    uint32_t hopping = 0;
};
