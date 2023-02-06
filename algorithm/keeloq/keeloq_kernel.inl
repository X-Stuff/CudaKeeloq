#include "device/cuda_common.h"

#include <cuda_runtime_api.h>


/**
 *  reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.c
 * Getting real encryption key
 */
namespace
{
    __device__ __host__ inline uint64_t keeloq_common_secure_learning(uint32_t data, uint32_t seed, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFF;
        k1 = keeloq_common_decrypt(data, key);
        k2 = keeloq_common_decrypt(seed, key);

        return ((uint64_t)k1 << 32) | k2;
    }

    __device__ __host__ inline uint64_t keeloq_common_magic_xor_type1_learning(uint32_t data, uint64_t xor)
    {
        data &= 0x0FFFFFFF;
        return (((uint64_t)data << 32) | data)^xor;
    }

    __device__ __host__ inline uint64_t keeloq_common_normal_learning(uint32_t data, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFF;
        data |= 0x20000000;
        k1 = keeloq_common_decrypt(data, key);

        data &= 0x0FFFFFFF;
        data |= 0x60000000;
        k2 = keeloq_common_decrypt(data, key);

        return ((uint64_t)k2 << 32) | k1; // key - shifrovanoya
    }

    __device__ __host__ inline uint64_t keeloq_common_faac_learning(const uint32_t seed, const uint64_t key)
    {
        uint16_t hs = seed >> 16;
        const uint16_t ending = 0x544D;
        uint32_t lsb = (uint32_t)hs << 16 | ending;
        uint64_t man = (uint64_t)keeloq_common_encrypt(seed, key) << 32 | keeloq_common_encrypt(lsb, key);
        return man;
    }

    __device__ __host__ inline uint64_t keeloq_common_magic_serial_type1_learning(uint32_t data, uint64_t man)
    {
        return (man & 0xFFFFFFFF) | ((uint64_t)data << 40) |
            ((uint64_t)(((data & 0xff) + ((data >> 8) & 0xFF)) & 0xFF) << 32);
    }

    __device__ __host__ inline uint64_t keeloq_common_magic_serial_type2_learning(uint32_t data, uint64_t man)
    {
        uint8_t* p = (uint8_t*)&data;
        uint8_t* m = (uint8_t*)&man;
        m[7] = p[0];
        m[6] = p[1];
        m[5] = p[2];
        m[4] = p[3];
        return man;
    }

    __device__ __host__ inline uint64_t keeloq_common_magic_serial_type3_learning(uint32_t data, uint64_t man)
    {
        return (man & 0xFFFFFFFFFF000000) | (data & 0xFFFFFF);
    }

    __device__ __host__ inline uint64_t keeloq_common_key_reverse(uint64_t key, uint8_t bit_count)
    {
        uint64_t reverse_key = 0;
        for (uint8_t i = 0; i < bit_count; i++) {
            reverse_key = reverse_key << 1 | bit(key, i);
        }
        return reverse_key;
    }
}

namespace
{
    template<bool forceall>
    __device__ KeeloqLearningType::Type analyze_results_srl(const SingleResult* results, uint8_t num, const KeeloqLearningMask type_mask)
    {
        uint8_t match_count = 0; // 0 or 1. if bigger - double match
        KeeloqLearningType::Type match_learning_type = KeeloqLearningType::INVALID;

        // outer loop - over all learning types
        UNROLL
        for (uint8_t lrn = 0; lrn < SingleResult::ResultsCount; ++lrn)
        {
            if (forceall || type_mask[lrn])
            {
                uint32_t expected_srl = (SingleResult::read_results_from_cache(results[0], lrn) >> 16) & 0x3ff;
                uint32_t lrn_matches = 1;

                // inner loop for every result
                for (uint8_t i = 1; i < num; ++i)
                {
                    const SingleResult& item = results[i];

                    uint32_t srl = (SingleResult::read_results_from_cache(item, lrn) >> 16) & 0x3ff;
                    lrn_matches += srl == expected_srl && srl != 0;
                }

                bool has_match = lrn_matches == num;

                match_count += has_match;
                match_learning_type = (has_match * lrn + !has_match * match_learning_type);

#if !STRICT_ANALYSIS
                // break imitation
                lrn += has_match * KeeloqLearningType::LAST;
#endif
            }
        }

        return match_learning_type;
    }

    template<KeeloqLearningType::Type type>
    __device__ __host__ inline uint32_t keeloq_decrypt_single(uint32_t data, uint32_t fix, const uint64_t key, const uint32_t seed)
    {
        uint64_t key_rev = misc::rev_bytes(key);
        uint64_t n_key = 0;

        switch (type)
        {
        case KeeloqLearningType::Simple:
            return keeloq_common_decrypt(data, key);
        case KeeloqLearningType::Simple_Rev:
            return keeloq_common_decrypt(data, key_rev);
        case KeeloqLearningType::Normal:
            n_key = keeloq_common_normal_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Normal_Rev:
            n_key = keeloq_common_normal_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);;
        case KeeloqLearningType::Secure:
            assert(seed != 0);
            n_key = keeloq_common_secure_learning(fix, seed, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Secure_Rev:
            assert(seed != 0);
            n_key = keeloq_common_secure_learning(fix, seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Xor:
            n_key = keeloq_common_magic_xor_type1_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Xor_Rev:
            n_key = keeloq_common_magic_xor_type1_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Faac:
            assert(seed != 0);
            n_key = keeloq_common_faac_learning(seed, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Faac_Rev:
            assert(seed != 0);
            n_key = keeloq_common_faac_learning(seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial1:
            n_key = keeloq_common_magic_serial_type1_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial1_Rev:
            n_key = keeloq_common_magic_serial_type1_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial2:
            n_key = keeloq_common_magic_serial_type2_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial2_Rev:
            n_key = keeloq_common_magic_serial_type2_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial3:
            n_key = keeloq_common_magic_serial_type3_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::Serial3_Rev:
            n_key = keeloq_common_magic_serial_type3_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        case KeeloqLearningType::INVALID:
        default:
            assert(false && "Invalid type for single decryption");
            return 0;
        }
    }

    template<bool forceall>
    __device__ __host__ inline void keeloq_decrypt_all(uint32_t data, uint32_t fix, const uint64_t key, const uint32_t seed,
        const KeeloqLearningType::Type type_mask[], SingleResult::DecryptedArray& decrypted)
    {
        // Check for mirrored man
        uint64_t key_rev = misc::rev_bytes(key);
        uint64_t n_key = 0;

        // Simple Learning
        {
            if (forceall || type_mask[KeeloqLearningType::Simple])
            {
                decrypted.data[KeeloqLearningType::Simple] = keeloq_common_decrypt(data, key);
            }
            if (forceall || type_mask[KeeloqLearningType::Simple_Rev])
            {
                decrypted.data[KeeloqLearningType::Simple_Rev] = keeloq_common_decrypt(data, key_rev);
            }
        }
        //###########################
        // Normal Learning
        // https://phreakerclub.com/forum/showpost.php?p=43557&postcount=37
        {
            if (forceall || type_mask[KeeloqLearningType::Normal])
            {
                n_key = keeloq_common_normal_learning(fix, key);
                decrypted.data[KeeloqLearningType::Normal] = keeloq_common_decrypt(data, n_key);
            }
            if (forceall || type_mask[KeeloqLearningType::Normal_Rev])
            {
                n_key = keeloq_common_normal_learning(fix, key_rev);
                decrypted.data[KeeloqLearningType::Normal_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // Secure Learning
        {
            if ((forceall && seed != 0) || type_mask[KeeloqLearningType::Secure])
            {
                assert(seed != 0);

                n_key = keeloq_common_secure_learning(fix, seed, key);
                decrypted.data[KeeloqLearningType::Secure] = keeloq_common_decrypt(data, n_key);
            }

            if ((forceall && seed != 0) || type_mask[KeeloqLearningType::Secure_Rev])
            {
                assert(seed != 0);

                n_key = keeloq_common_secure_learning(fix, seed, key_rev);
                decrypted.data[KeeloqLearningType::Secure_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // Magic xor type1 learning
        {
            if (forceall || type_mask[KeeloqLearningType::Xor])
            {
                n_key = keeloq_common_magic_xor_type1_learning(fix, key);
                decrypted.data[KeeloqLearningType::Xor] = keeloq_common_decrypt(data, n_key);
            }

            if (forceall || type_mask[KeeloqLearningType::Xor_Rev])
            {
                n_key = keeloq_common_magic_xor_type1_learning(fix, key_rev);
                decrypted.data[KeeloqLearningType::Xor_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // FAAC
        {
            if ((forceall && seed != 0) || type_mask[KeeloqLearningType::Faac])
            {
                assert(seed != 0);

                n_key = keeloq_common_faac_learning(seed, key);
                decrypted.data[KeeloqLearningType::Faac] = keeloq_common_decrypt(data, n_key);
            }

            if ((forceall && seed != 0) || type_mask[KeeloqLearningType::Faac_Rev])
            {
                assert(seed != 0);

                n_key = keeloq_common_faac_learning(seed, key_rev);
                decrypted.data[KeeloqLearningType::Faac_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // SERIAL_TYPE_1
        {
            if (forceall || type_mask[KeeloqLearningType::Serial1])
            {
                n_key = keeloq_common_magic_serial_type1_learning(fix, key);
                decrypted.data[KeeloqLearningType::Serial1] = keeloq_common_decrypt(data, n_key);
            }

            if (forceall || type_mask[KeeloqLearningType::Serial1_Rev])
            {
                n_key = keeloq_common_magic_serial_type1_learning(fix, key_rev);
                decrypted.data[KeeloqLearningType::Serial1_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // SERIAL_TYPE_2
        {
            if (forceall || type_mask[KeeloqLearningType::Serial2])
            {
                n_key = keeloq_common_magic_serial_type2_learning(fix, key);
                decrypted.data[KeeloqLearningType::Serial2] = keeloq_common_decrypt(data, n_key);
            }

            if (forceall || type_mask[KeeloqLearningType::Serial2_Rev])
            {
                n_key = keeloq_common_magic_serial_type2_learning(fix, key_rev);
                decrypted.data[KeeloqLearningType::Serial2_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }

        // SERIAL_TYPE_3
        {
            if (forceall || type_mask[KeeloqLearningType::Serial3])
            {
                n_key = keeloq_common_magic_serial_type3_learning(fix, key);
                decrypted.data[KeeloqLearningType::Serial3] = keeloq_common_decrypt(data, n_key);
            }

            if (forceall || type_mask[KeeloqLearningType::Serial3_Rev])
            {
                n_key = keeloq_common_magic_serial_type3_learning(fix, key_rev);
                decrypted.data[KeeloqLearningType::Serial3_Rev] = keeloq_common_decrypt(data, n_key);
            }
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ uint8_t analyze_results_cnt(const SingleResult* results, uint32_t num, KeeloqLearningType::Type learning_type)
{
    uint8_t counter_maxdiff = num + 1;

    uint32_t expected_cnt = results[0].results.data[learning_type] & 0x0000FFFF;
    uint32_t lrn_matches = 0;

    for (uint8_t item = 0; item < num; ++item)
    {
        uint32_t cnt = results[item].results.data[learning_type] & 0x0000FFFF;
        lrn_matches += __usad(expected_cnt, cnt, 0) < counter_maxdiff;
    }

    return lrn_matches == num;
}


__device__ uint8_t analyze_results_btn(const SingleResult* results, uint32_t num, KeeloqLearningType::Type learning_type, uint8_t bit_tolerance = 0)
{
    uint32_t expected_btn = results[0].results.data[learning_type] >> 28;
    uint32_t lrn_matches = 1;

    for (uint8_t item = 1; item < num; ++item)
    {
        uint32_t btn = results[item].results.data[learning_type] >> 28;
        lrn_matches += (btn ^ expected_btn) <= bit_tolerance;
    }

    return lrn_matches == num;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
__device__ uint8_t keeloq_find_matches(const CudaContext& ctx, SingleResult* results, uint32_t num, const KeeloqLearningMask type_mask)
{
    uint8_t result_error = 0;

    KeeloqLearningType::Type learning_type = KeeloqLearningType::AllEnabled(type_mask) ?
        analyze_results_srl<true>(results, num, type_mask) :
        analyze_results_srl<false>(results, num, type_mask);

    if (learning_type != KeeloqLearningType::INVALID)
    {
        // Check same button and check continuous presses
        if (analyze_results_btn(results, num, learning_type) && analyze_results_cnt(results, num, learning_type))
        {
            for (int i = 0; i < num; ++i)
            {
                results[i].match = learning_type;
            }
        }

    #if STRICT_ANALYSIS
        uint64_t first_man = results[0].man;
        for (int i = 0; i < num; ++i)
        {
            result_error += first_man != results[i].man;
        }
    #endif
    }

    return result_error;
}

// aggregate matches into count
__device__ uint8_t keeloq_analyze_results(const CudaContext& ctx, const CudaArray<SingleResult>& all_results, uint32_t num_decryptors, uint32_t num_inputs)
{
    uint8_t num_matches = 0;

    // outer loop for thread decryptor
    CUDA_FOR_THREAD_ID(ctx, decryptor_index, num_decryptors)
    {
        const SingleResult* decryptor_results = &all_results[decryptor_index * num_inputs];

        // inner loop for each result of this decryptor
        for (uint8_t r = 0; r < num_inputs; ++r)
        {
            num_matches += (decryptor_results[r].match != KeeloqLearningType::INVALID);
        }
    }

    return num_matches;
}


// run decryption parallel per thread and find matches
__device__ uint8_t keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelInput& input)
{
    auto& encrypted = *input.encdata;
    auto& decryptors = *input.decryptors;
    auto& results = *input.results;

    uint8_t result_error = 0;

    // outer loop for each thread's decryptor
    CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
    {
        uint64_t man = decryptors[decryptor_index].man;
        uint32_t seed = decryptors[decryptor_index].seed;

        // inner loop for each input for decryptor - make decryption
        for (uint32_t enc_index = 0; enc_index < encrypted.num; enc_index++)
        {
            size_t result_index = decryptor_index * encrypted.num + enc_index;

            SingleResult& result = results[result_index];

            result.man = man;
            result.seed = seed;
            result.match = KeeloqLearningType::INVALID;

            result.ota = encrypted[enc_index];

            keeloq_decrypt(result.ota, result.man, result.seed, input.learning_types, result.results);
        }

        // now check find matches in check decryptors
        result_error += keeloq_find_matches(ctx, &results[decryptor_index * encrypted.num], encrypted.num, input.learning_types);
    }

    return result_error;
}

__device__ __host__ void keeloq_decrypt(uint64_t ota, uint64_t man, uint32_t seed, const KeeloqLearningMask type_mask, SingleResult::DecryptedArray& results)
{
    uint64_t key = keeloq_common_key_reverse(ota, sizeof(ota) * 8);

    uint32_t fix = (uint32_t)(key >> 32);
    uint32_t hop = (uint32_t)(key);

    if (KeeloqLearningType::AllEnabled(type_mask)) // force all decryption learnings
    {
        // compiler should optimize out the ifs in that case since condition became `(true || XXXX`
        keeloq_decrypt_all<true>(hop, fix, man, seed, type_mask, results);
    }
    else
    {
        keeloq_decrypt_all<false>(hop, fix, man, seed, type_mask, results);
    }
}
