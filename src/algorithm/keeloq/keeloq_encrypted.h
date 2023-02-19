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
    uint64_t ota;

    __host__ EncParcel() : EncParcel(0) { }

    __device__ __host__ EncParcel(uint64_t data) : ota(data)
    {
        uint64_t key = misc::rev_bits(ota, sizeof(ota) * 8);

        fixed = (uint32_t)(key >> 32);
        hopping = (uint32_t)(key);
    }

    // Fixed code in parcel
    __device__ __host__ inline uint32_t fix() const { return fixed; }

    // hopping code in parcel
    __device__ __host__ inline uint32_t hop() const { return hopping; }

    // first 18 bits of fixed code - serial (can be used in decryption)
    __device__ __host__ inline uint32_t srl() const { return fixed & 0x3FF; }

    // last 4 bits of fixed code - button (can be used in decryption)
    __device__ __host__ inline uint32_t btn() const { return fixed >> 28; }

private:

    // Fixed part of the parcel ( 28-bit serial | 4-bit button )
    uint32_t fixed;

    // Encrypted hopping code ( keeloq encrypted serial, button, counter )
    uint32_t hopping;
};