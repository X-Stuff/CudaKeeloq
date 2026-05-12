#pragma once

#include <cstdint>

#include "common.h"

#include "kernels/inputs_mutation.h"

#include "algorithm/keeloq/keeloq_decryptor.h"


/**
 *  Experimental result for flat decryption,
 * Flat decryption means that we choose learning type and modifier from CPU level
 * On GPU level we just process bigger amount of data, with the same decryptors,
 * allocated memory for results is reused for every learning type.
 */
struct SingleLearningResult
{
    // Packed: bit 0, match or not, bit 1-3 - input index (0,1,2), bit 4-7 - input mutation
    uint8_t results = 0;

    // used manufacturer key and seed for this result
    Decryptor decryptor = {};

    // Processed values for each known learning type (decrypted or encrypted depending on the call), indexed by learning type and modifier
    uint32_t decrypted = 0;

    /** Extract the serial portion of a decrypted entry (10 bits). */
    __host__ __device__ __forceinline__ uint32_t srl() const
    {
        return (decrypted >> 16) & 0x3ff;
    }

    /** Extract the button portion of a decrypted entry (top 4 bits). */
    __host__ __device__ __forceinline__ uint16_t btn() const
    {
        return (decrypted >> 28);
    }

    /** Extract the counter portion of a decrypted entry (lower 16 bits). */
    __host__ __device__ __forceinline__ uint16_t cnt() const
    {
        return (decrypted & 0x0000FFFF);
    }

    __host__ __device__ __forceinline__ bool hasMatch() const
    {
        return results & 0x01;
    }

    __host__ __device__ __forceinline__ void setHasMatch(uint8_t isMatch)
    {
        results = (results & 0xFE) | (isMatch > 0);
    }

    __host__ __device__ __forceinline__ void setInputIndex(uint8_t index)
    {
        results = (results & 0xF1) | ((index << 1) & 0x0E);
    }

    __host__ __device__ __forceinline__ uint8_t inputIndex() const
    {
        return (results >> 1) & 0x07;
    }

    __host__ __device__ __forceinline__ void setInputsMutation(InputsMutation m)
    {
        results = (results & 0x0F) | (static_cast<uint8_t>(m) << 4);
    }

    __host__ __device__ __forceinline__ InputsMutation inputsMutation() const
    {
        return static_cast<InputsMutation>(results >> 4);
    }
};