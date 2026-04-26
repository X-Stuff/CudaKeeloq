#include "device/cuda_common.h"
#include "device/cuda_span.h"

#include <cuda_runtime_api.h>

constexpr uint8_t OneEncInput = 1;
constexpr uint8_t TwoEncInputs = 2;
constexpr uint8_t ThreeEncInputs = 3;


/**
 *  reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.c
 * Getting real encryption key
 *
 * Additional variation of learning algorithms:
 *  - Using encrypt instead of decrypt for key generation (for some learning types)
 */
namespace
{
    template<bool UseDecrypt = true>
    __device__ __host__ inline uint64_t keeloq_common_secure_learning(uint32_t data, uint32_t seed, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFFUL;

        if constexpr (UseDecrypt)
        {
            k1 = keeloq_common_decrypt(data, key);
            k2 = keeloq_common_decrypt(seed, key);
        }
        else
        {
            k1 = keeloq_common_encrypt(data, key);
            k2 = keeloq_common_encrypt(seed, key);
        }

        return ((uint64_t)k1 << 32) | k2;
    }

    __device__ __host__ inline uint64_t keeloq_common_magic_xor_type1_learning(uint32_t data, uint64_t pxor)
    {
        data &= 0x0FFFFFFFUL;
        return (((uint64_t)data << 32) | data) ^ pxor;
    }

    template<bool UseDecrypt = true>
    __device__ __host__ inline uint64_t keeloq_common_normal_learning(uint32_t data, const uint64_t key)
    {
        uint32_t k1, k2;

        data &= 0x0FFFFFFFUL;
        if constexpr (UseDecrypt)
        {
            k1 = keeloq_common_decrypt(data | 0x20000000UL, key);
            k2 = keeloq_common_decrypt(data | 0x60000000UL, key);
        }
        else
        {
            k1 = keeloq_common_encrypt(data | 0x20000000UL, key);
            k2 = keeloq_common_encrypt(data | 0x60000000UL, key);
        }

        return ((uint64_t)k2 << 32) | k1; // key - shifrovanoya
    }

    template<bool UseDecrypt = false>
    __device__ __host__ inline uint64_t keeloq_common_faac_learning(const uint32_t seed, const uint64_t key)
    {
        const uint16_t hs = seed >> 16;
        const uint16_t ending = 0x544D;
        const uint32_t lsb = (uint32_t)hs << 16 | ending;

        if constexpr (UseDecrypt)
        {
            return (uint64_t)keeloq_common_decrypt(seed, key) << 32 | keeloq_common_decrypt(lsb, key);
        }
        else
        {
            return (uint64_t)keeloq_common_encrypt(seed, key) << 32 | keeloq_common_encrypt(lsb, key);
        }
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace debug
{
// Some problems with printf - looks like silently hit instruction limit
__device__ __noinline__ void assert_single_match_count(uint8_t match_count, bool has_match, KeeloqLearning::ResultIndex curr_res_indx, KeeloqLearning::ResultIndex new_res_index,
    const Span<SingleResult>& results)
{
    if (match_count > 0 && has_match)
    {
        auto ctx = CudaContext::Get();
        printf("assert_single_match_count(): %u, detected multiple match. Prev resIndx: %d Now resIndx: %d. Man: 0x%llX Inputs:[ 0x%llX; 0x%llX; 0x%llX]\n",
            ctx.thread_id,
            curr_res_indx, new_res_index,
            results[0].decryptor.man(),
            results[0].encrypted.ota, results[1].encrypted.ota, results[2].encrypted.ota);
    }
}
}

namespace
{

template<uint8_t NumInputs>
__device__ uint8_t is_cnt_match(const Span<SingleResult>& results, KeeloqLearning::ResultIndex resIndex)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    const uint8_t counter_maxdiff = NumInputs + 1;

    const uint32_t expected_cnt = results[0].decrypted.cnt(resIndex);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        const uint32_t cnt = results[item].decrypted.cnt(resIndex);
        lrn_matches += __usad(expected_cnt, cnt, 0) < counter_maxdiff;
    }

    return lrn_matches == NumInputs;
}

template<uint8_t NumInputs>
__device__ uint8_t is_btn_match(const Span<SingleResult>& results, KeeloqLearning::ResultIndex resIndex)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    const uint32_t expected_btn = results[0].decrypted.btn(resIndex);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        lrn_matches += results[item].decrypted.btn(resIndex) == expected_btn;
    }

    return lrn_matches == NumInputs;
}

template<uint8_t NumInputs>
__device__ uint8_t is_srl_match(const Span<SingleResult>& results, KeeloqLearning::ResultIndex resIndex)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    const uint32_t expected_srl = results[0].decrypted.srl(resIndex);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        const bool decryptor_valid = results[item].decryptor.is_valid();
        lrn_matches += decryptor_valid && results[item].decrypted.srl(resIndex) == expected_srl;
    }

    return lrn_matches == NumInputs && expected_srl != 0; // 0 check at the end to save instructions in loop
}

template<uint8_t NumInputs, bool ForceAllLearningTypes>
__device__ KeeloqLearning::ResultIndex get_match_index(const Span<SingleResult>& results, const KeeloqLearning::Matrix& learnings_matrix)
{
    uint8_t match_count = 0; // 0 or 1. if bigger - double match
    KeeloqLearning::ResultIndex match_res_index = KeeloqLearning::NoMatch;

    // outer loop - over all learning types
    UNROLL
    for (auto resIndex = 0; resIndex < KeeloqLearning::InvalidResultIndex; ++resIndex)
    {
        // Compiler should optimize this block and throw away `if` check in case of ForceAllLearningTypes = true
        const bool allowed = ForceAllLearningTypes || learnings_matrix.isEnabled(resIndex);

        if (allowed)
        {
            const bool has_match = is_srl_match<NumInputs>(results, resIndex) && is_btn_match<NumInputs>(results, resIndex) && is_cnt_match<NumInputs>(results, resIndex);

        #if _DEBUG
            debug::assert_single_match_count(match_count, has_match, match_res_index, resIndex, results);
        #endif

            match_count += has_match;
            match_res_index = (has_match * resIndex + !has_match * match_res_index);
        }

    }

    assert(match_count <= 1 && "get_match_index() Multiple match - should not be possible!");
    return match_res_index;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
template<uint8_t NumInputs, LearningDecryptionMode LearningMode>
__device__ KeeloqLearning::ResultIndex analyze_multiple_results(const CudaContext& ctx, Span<SingleResult>& results, const KeeloqLearning::Matrix& learnings_matrix)
{
    auto match_res_indx =  get_match_index<NumInputs, LearningMode == LearningDecryptionMode::ForceAll>(results, learnings_matrix);

    UNROLL
    for (int i = 0; i < NumInputs; ++i)
    {
        // If no match will return INVALID - rewrite same value better than have if block
        results[i].match = match_res_indx;
    }

    return match_res_indx;
}

// In case of single input we checking fixed part of parcel's serial (28-bit serial | 4-bit button)
// with decoded serial
__device__ KeeloqLearning::ResultIndex analyze_single_result(const SingleResult& result, uint32_t exp_srl, uint8_t exp_btn, const KeeloqLearning::Matrix& learnings_matrix, uint8_t& match_count)
{
    match_count = 0; // 0 or 1. if bigger - double match
    auto match_res_index = KeeloqLearning::NoMatch;

    // outer loop - over all learning types
    UNROLL
    for (auto resIndex = 0; resIndex < KeeloqLearning::InvalidResultIndex; ++resIndex)
    {
        // No if block
        const bool allowed = learnings_matrix.isEnabled(resIndex);

        const uint32_t srl = result.decrypted.srl(resIndex);
        const uint32_t btn = result.decrypted.btn(resIndex);

        const bool has_match = allowed && srl == exp_srl && srl != 0 && btn == exp_btn;

        match_count += has_match;
        match_res_index = (has_match * resIndex + !has_match * match_res_index);
    }

    assert(match_count <= 1 && "analyze_single_result() Multiple match - should not be possible!");
    return match_res_index;
}

}

namespace
{

__host__ __device__ constexpr bool operator&(const LearningDecryptionMode& a, const LearningDecryptionMode& b)
{
    return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

__host__ __device__ constexpr LearningDecryptionMode operator|(const LearningDecryptionMode& a, const LearningDecryptionMode& b)
{
    return static_cast<LearningDecryptionMode>(static_cast<int>(a) | static_cast<int>(b));
}

template<KeeloqLearning::Type type, KeeloqLearning::Mod::Type mod>
__device__ __host__ __forceinline__ uint32_t keeloq_decrypt_single(uint32_t data, uint32_t fix, const Decryptor& decryptor)
{
    const uint64_t key = decryptor.man();
    const uint64_t key_rev = decryptor.nam();
    const uint32_t seed = decryptor.seed();

    static_assert(mod == KeeloqLearning::Mod::Type::Regular || mod == KeeloqLearning::Mod::Type::ReversedKey || mod == KeeloqLearning::Mod::Type::InvertedDec, "Unsupported KeeloqLearningMod");
    static_assert(type == KeeloqLearning::Type::Simple || type == KeeloqLearning::Type::Normal || type == KeeloqLearning::Type::Secure ||
        type == KeeloqLearning::Type::Xor || type == KeeloqLearning::Type::Faac || type == KeeloqLearning::Type::Serial1 ||
        type == KeeloqLearning::Type::Serial2 || type == KeeloqLearning::Type::Serial3, "Unsupported KeeloqLearningType");

    static_assert(KeeloqLearning::Matrix::getIndex<type, mod>() != KeeloqLearning::InvalidResultIndex, "Unsupported KeeloqLearningType/Mod combination");

    if constexpr (mod == KeeloqLearning::Mod::Type::Regular)
    {
        if constexpr (type == KeeloqLearning::Type::Simple)
        {
            return keeloq_common_decrypt(data, key);
        }
        else if constexpr (type == KeeloqLearning::Type::Normal)
        {
            uint64_t n_key = keeloq_common_normal_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Secure)
        {
            assert(seed != 0);
            uint64_t n_key = keeloq_common_secure_learning(fix, seed, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Xor)
        {
            uint64_t n_key = keeloq_common_magic_xor_type1_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Faac)
        {
            assert(seed != 0);
            uint64_t n_key = keeloq_common_faac_learning(seed, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial1)
        {
            uint64_t n_key = keeloq_common_magic_serial_type1_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial2)
        {
            uint64_t n_key = keeloq_common_magic_serial_type2_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial3)
        {
            uint64_t n_key = keeloq_common_magic_serial_type3_learning(fix, key);
            return keeloq_common_decrypt(data, n_key);
        }
    }
    else if constexpr (mod == KeeloqLearning::Mod::Type::ReversedKey)
    {
        if constexpr (type == KeeloqLearning::Type::Simple)
        {
            return keeloq_common_decrypt(data, key_rev);
        }
        else if constexpr (type == KeeloqLearning::Type::Normal)
        {
            uint64_t n_key = keeloq_common_normal_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Secure)
        {
            assert(seed != 0);
            uint64_t n_key = keeloq_common_secure_learning(fix, seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Xor)
        {
            uint64_t n_key = keeloq_common_magic_xor_type1_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Faac)
        {
            assert(seed != 0);
            uint64_t n_key = keeloq_common_faac_learning(seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial1)
        {
            uint64_t n_key = keeloq_common_magic_serial_type1_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial2)
        {
            uint64_t n_key = keeloq_common_magic_serial_type2_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Serial3)
        {
            uint64_t n_key = keeloq_common_magic_serial_type3_learning(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
    }
    else if constexpr (mod == KeeloqLearning::Mod::Type::InvertedDec)
    {
        if constexpr (type == KeeloqLearning::Type::Normal)
        {
            uint64_t n_key = keeloq_common_normal_learning<false>(fix, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Secure)
        {
            assert(seed != 0);
            uint64_t n_key = keeloq_common_secure_learning<false>(fix, seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
        else if constexpr (type == KeeloqLearning::Type::Faac)
        {
            // Faac uses encrypt by default, Inverted logic here will be decrypt
            assert(seed != 0);
            uint64_t n_key = keeloq_common_faac_learning<true>(seed, key_rev);
            return keeloq_common_decrypt(data, n_key);
        }
    }
}


template<KeeloqLearning::Type type, KeeloqLearning::Mod::Type mod>
__device__ __host__ __forceinline__ void keeloq_decrypt_single(uint32_t data, uint32_t fix, const Decryptor& decryptor, KeeloqLearning::DecryptedResults& results)
{
    constexpr auto index = KeeloqLearning::Matrix::getIndex<type, mod>();
    assert(index != KeeloqLearning::InvalidResultIndex && "Invalid learning type/mod combination");

    results[index] = keeloq_decrypt_single<type, mod>(data, fix, decryptor);
}

template<bool ForceDecrypt, bool DisableRev = false>
__device__ __host__ __forceinline__ void keeloq_decrypt_seed_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const KeeloqLearning::Matrix& learnings_matrix, SingleResult::DecryptedArray& decrypted)
{
    if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Secure, KeeloqLearning::Mod::Type::Regular))
    {
        keeloq_decrypt_single<KeeloqLearning::Type::Secure, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
    }

    if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Faac, KeeloqLearning::Mod::Type::Regular))
    {
        keeloq_decrypt_single<KeeloqLearning::Type::Faac, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
    }

    if constexpr (!DisableRev)
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Secure, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Secure, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }

        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Faac, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Faac, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
    }
}

template<bool ForceDecrypt, bool DisableRev = false>
__device__ __host__ __forceinline__ void keeloq_decrypt_normal_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const KeeloqLearning::Matrix& learnings_matrix, SingleResult::DecryptedArray& decrypted)
{
    // Simple Learning
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Simple, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Simple, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }
    }
    //###########################
    // Normal Learning
    // https://phreakerclub.com/forum/showpost.php?p=43557&postcount=37
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Normal, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Normal, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }
    }

    // Magic xor type1 learning
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Xor, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Xor, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }

    }

    // SERIAL_TYPE_1
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial1, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial1, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }

    }

    // SERIAL_TYPE_2
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial2, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial2, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }

    }

    // SERIAL_TYPE_3
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial3, KeeloqLearning::Mod::Type::Regular))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial3, KeeloqLearning::Mod::Type::Regular>(data, fix, decryptor, decrypted.data);
        }
    }


    // Reverse key calculations
    if constexpr (!DisableRev)
    {
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Simple, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Simple, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Normal, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Normal, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Xor, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Xor, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial1, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial1, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial2, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial2, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
        if (ForceDecrypt || learnings_matrix.isEnabled(KeeloqLearning::Type::Serial3, KeeloqLearning::Mod::Type::ReversedKey))
        {
            keeloq_decrypt_single<KeeloqLearning::Type::Serial3, KeeloqLearning::Mod::Type::ReversedKey>(data, fix, decryptor, decrypted.data);
        }
    }
}

template<LearningDecryptionMode Mode>
__device__ __host__ inline void keeloq_decrypt(const EncParcel& enc, const Decryptor& decryptor, const KeeloqLearning::Matrix& learnings_matrix, SingleResult::DecryptedArray& results)
{
    constexpr bool ForceDecrypt = Mode & LearningDecryptionMode::Force;
    constexpr bool ExplicitDecrypt = Mode & LearningDecryptionMode::Explicit;

    constexpr bool DisableReverseKeys = Mode & LearningDecryptionMode::NoRev;

    static_assert(ForceDecrypt != ExplicitDecrypt, "Can't be only Force or Explicit");

    if constexpr (Mode & LearningDecryptionMode::Normal)
    {
        keeloq_decrypt_normal_all<ForceDecrypt, DisableReverseKeys>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
    }

    if constexpr (Mode & LearningDecryptionMode::Seeded)
    {
        keeloq_decrypt_seed_all<ForceDecrypt, DisableReverseKeys>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
    }
}


/**
 *  Generally will decrypt encrypted input(s) and set their `.match` fields according of decryption result
 * with provided Decryptor
 *
 * Most common way to use this:
 *  - You have 3 inputs
 *  - You call this method to decrypt and analyze 1-st input
 *  - if first input has at least 1 match ( by fixed part srl and button )
 *    * call decrypt for the rest of inputs
 *    * properly analyze their decrypted parts (each 3 decrypted inputs should have same serial, button, and (increasing) counter
 *  - if first input doesn't have any match - further calculations pointless
 */
template<uint8_t NumResults, LearningDecryptionMode LearningMode, uint8_t FirstResultIndex>
__device__ uint8_t inline keeloq_decrypt_and_analyze(const CudaContext& ctx,
    const CudaArray<EncParcel>& encrypted, const Decryptor& decryptor, const KeeloqLearning::Matrix& learning_matrix, Span<SingleResult>& results)
{
    static_assert(NumResults > 1, "Use `keeloq_decrypt_and_quick_analyze()` if results number is less than 2");
    static_assert(FirstResultIndex < NumResults, "Invalid template parameters!");

    assert(encrypted.num >= NumResults && "Encrypted array size doesn't match results array!");

    // inner loop for each input for decryptor - make decryption
    UNROLL
    for (uint32_t i = FirstResultIndex; i < NumResults; ++i)
    {
        SingleResult& result = results[i];
        result.match = KeeloqLearning::NoMatch;

        result.decryptor = decryptor;
        result.encrypted = encrypted[i];

        keeloq_decrypt<LearningMode>(result.encrypted, decryptor, learning_matrix, result.decrypted);
    }

    // now check all decrypted results if they match somehow
    analyze_multiple_results<NumResults, LearningMode>(ctx, results, learning_matrix);
    return 0;
}

/**
 *  This function decrypts input specified in template argument @InputIndex
 * And writes `result.match` if fixed part matches decrypted hopping.
 * This is used as first filter before robust check over all inputs.
 * There could be more than one match, in this method we do not care,
 * we want to have indication that this decryptor has a possibility to be the correct one.
 */
template<LearningDecryptionMode LearningMode, uint8_t InputIndex>
__device__ uint8_t inline keeloq_decrypt_and_quick_analyze(const CudaContext& ctx,
    const CudaArray<EncParcel>& encrypted, const Decryptor& decryptor, const KeeloqLearning::Matrix& learning_matrix, Span<SingleResult>& results)
{
    static_assert(InputIndex < 2, "You want Input index be 1 or 2 MOST LIKELY. Since last check should be the robust one");

    const EncParcel& enc = encrypted[InputIndex];

    uint8_t match_count = 0;
    SingleResult& result = results[InputIndex];

    result.decryptor = decryptor;
    result.encrypted = enc; // useless copy

    keeloq_decrypt<LearningMode>(enc, decryptor, learning_matrix, result.decrypted);

    result.match = analyze_single_result(result, enc.srl(), enc.btn(), learning_matrix, match_count);

    // In case of single input that means that key has 2 (most probably) phantoms
    return match_count > 1;
}


/**
 *  Run decryption parallel per thread and find matches
 * NumInputs - means how many inputs (encrypted OTA data) we have at all, single, two or three.
 *  depending on that num different optimization are applied deeper (like loop unrolling when writing results)
 * LearningMode - determines which learning calculation could be thrown away. (we could have no seed, that mean we don't need to even check ifs)
 * UseFastCheck - allows to early exit if first decrypted input doesn't match some bits from fixed part
 */
template<uint8_t NumInputs, LearningDecryptionMode LearningMode, bool UseFastCheck = true>
__device__ uint8_t inline keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelInput& input)
{
    constexpr bool UseFullCheck = NumInputs > 1 && !UseFastCheck;
    constexpr uint8_t First = 0;
    constexpr uint8_t Second = 1;
    constexpr uint8_t Third = 2;

    static_assert(NumInputs > 0 && NumInputs <= 3, "Invalid inputs number!");
    static_assert(First < Second && Second < Third, "Static assert just to get rid of warning");

    const auto& encrypted = *input.encdata;
    const auto& decryptors = *input.decryptors;

    auto& all_results = *input.results;

    assert(encrypted.num >= NumInputs && "Number of encrypted inputs less than expected with template argument");
    assert(NumInputs == 1 || encrypted[0].ota != encrypted[1].ota && "Inputs are the same!");

    uint8_t result_error = 0;

    // outer loop for each thread's decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, decryptors.num)
    {
        const Decryptor& decryptor = decryptors[decryptor_index];

        Span<SingleResult> results(&all_results[decryptor_index * encrypted.num], NumInputs);

        if constexpr (UseFullCheck)
        {
            // only multiple input - decrypt all and then check
            keeloq_decrypt_and_analyze<NumInputs, LearningMode, First>(ctx, encrypted, decryptor, input.GetLearningMask(), results);
        }
        else
        {
            // Single input
            auto multiple_match = keeloq_decrypt_and_quick_analyze<LearningMode, First>(ctx, encrypted, decryptor, input.GetLearningMask(), results);
            if constexpr (NumInputs == 1)
            {
                // Single mode only. in multiple - it's ok, we'll check just the rest
                result_error += multiple_match;
            }

            // Multiple input
            if constexpr (NumInputs > 1)
            {
                assert(encrypted[First].fix() == encrypted[Second].fix() && "Cannot `UseFastCheck` if fixed part of encrypted packets are not the same!");

                // Even if this is `if` - still it's faster
                if (results[First].match != KeeloqLearning::NoMatch)
                {
                    if constexpr (NumInputs > 2)
                    {
                        // Calling decrypt and fast check on next input
                        keeloq_decrypt_and_quick_analyze<LearningMode, Second>(ctx, encrypted, decryptor, input.GetLearningMask(), results);

                        if (results[Second].match != KeeloqLearning::NoMatch)
                        {
                            // since 1-st and 2-nd was already decrypted - starting index is 2
                            // This will also reset the `result.match`
                            keeloq_decrypt_and_analyze<NumInputs, LearningMode, Third>(ctx, encrypted, decryptor, input.GetLearningMask(), results);
                        }
                        else
                        {
                            UNROLL
                            for (uint32_t i = First; i < NumInputs; ++i)
                            {
                                results[i].match = KeeloqLearning::NoMatch;
                            }
                        }
                    }
                    else
                    {
                        // since 1-st was already decrypted - starting index is 1
                        // This will also reset the `result.match`
                        keeloq_decrypt_and_analyze<NumInputs, LearningMode, Second>(ctx, encrypted, decryptor, input.GetLearningMask(), results);
                    }
                }
                else
                {
                    UNROLL
                    for (uint32_t i = First; i < NumInputs; ++i)
                    {
                        results[i].match = KeeloqLearning::NoMatch;
                    }
                }

            }
        }
    }

    return result_error;
}

// aggregate matches into count
template<uint8_t NumInputs>
__device__ uint8_t inline keeloq_check_matches(const CudaContext& ctx, const CudaArray<SingleResult>& all_results, const KeeloqKernelInput::TCudaPtr KernelInputs)
{
    uint8_t num_matches = 0;

    // outer loop for thread decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, KernelInputs->decryptors->num)
    {
        if constexpr (NumInputs == 1)
        {
            num_matches += (all_results[decryptor_index].match != KeeloqLearning::NoMatch);
        }
        else
        {
            const SingleResult* decryptor_results = &all_results[decryptor_index * KernelInputs->encdata->num];

            // inner loop for each result of this decryptor
            UNROLL
            for (uint8_t r = 0; r < NumInputs; ++r)
            {
                num_matches += (decryptor_results[r].match != KeeloqLearning::NoMatch);
            }
        }
    }

    return num_matches;
}


template <uint8_t NumInputs, bool UseFastCheck = true>
__device__ void __noinline__ Kernel_keeloq_main(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();
    auto& results = *KernelInputs->results;

    bool forceAll = KernelInputs->AllLearningsEnabled();

    bool hasSeed = KernelInputs->GetConfig().start.seed() != 0 || KernelInputs->GetConfig().type == BruteforceType::Dictionary;
    bool seedOnly = KernelInputs->GetConfig().type == BruteforceType::Seed;

    uint8_t num_errors = 0;

    if (seedOnly)
    {
        num_errors = forceAll ?
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceSeeded | LearningDecryptionMode::NoRev, UseFastCheck>(ctx, *KernelInputs) :
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitSeeded, UseFastCheck>(ctx, *KernelInputs);
    }
    else if (forceAll)
    {
        num_errors = hasSeed ?
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceAll, UseFastCheck>(ctx, *KernelInputs) :
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceNormal, UseFastCheck>(ctx, *KernelInputs);
    }
    else if (hasSeed)
    {
        num_errors = keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitAll, UseFastCheck>(ctx, *KernelInputs);
    }
    else
    {
        num_errors = keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitNormal, UseFastCheck>(ctx, *KernelInputs);
    }


    uint8_t num_matches = keeloq_check_matches<NumInputs>(ctx, results, KernelInputs);

    atomicAdd(&ret->error, num_errors);
    atomicAdd(&ret->value, num_matches);
}

}


namespace
{

__global__ void Kernel_keeloq_test(KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();

    if (ctx.thread_id == 0) {

        uint32_t pln_test = 0x11223344;
        uint64_t key_test = 0xDEADBEEF00226688;
        uint32_t enc_test = keeloq_common_encrypt(pln_test, key_test);
        if (pln_test != keeloq_common_decrypt(enc_test, key_test))
        {
            ret->error = 1;
        }
        else
        {
            ret->value = 1;
        }
    }
}


__global__ void Kenrel_keeloq_single_check(uint64_t ota, uint64_t man, uint32_t seed, KeeloqLearning::Type type, KeeloqLearning::Mod::Type mod, KernelResult::TCudaPtr ret)
{
    SingleResult::DecryptedArray decrypted = {};

    EncParcel enc(ota);
    Decryptor decryptor(man, seed);
    KeeloqLearning::Matrix full(KeeloqLearning::Matrix::kEverything);

    // printf("OTA: 0x%llX. Man: 0x%llX, seed: %u, learning: %d\n", ota, man, seed, learning);

    keeloq_decrypt<LearningDecryptionMode::ForceAll>(enc, decryptor, full, decrypted);

    ret->value = decrypted.data[KeeloqLearning::Matrix::getIndex(type, mod)];
}


__global__ void Kernel_keeloq_main_one_input(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    Kernel_keeloq_main<OneEncInput>(KernelInputs, ret);
}

__global__ void Kernel_keeloq_main_two_inputs(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret, bool fixed_parts_match)
{
    if (fixed_parts_match)
    {
        Kernel_keeloq_main<TwoEncInputs, true>(KernelInputs, ret);
    }
    else
    {
        // Optimizations fallback
        // fixed parts not matched - cant use fast check
        Kernel_keeloq_main<TwoEncInputs, false>(KernelInputs, ret);
    }
}

__global__ void Kernel_keeloq_main_three_inputs_fast(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{

    Kernel_keeloq_main<ThreeEncInputs, true>(KernelInputs, ret);
}

__global__ void Kernel_keeloq_main_three_inputs_slow(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{

    Kernel_keeloq_main<ThreeEncInputs, false>(KernelInputs, ret);
}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock)
{
    KernelResult kernel_results;

    switch (mainInputs.NumInputs())
    {
    case 1:
    {
        Kernel_keeloq_main_one_input<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());
        break;
    }
    case 2:
    {
        Kernel_keeloq_main_two_inputs<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr(), mainInputs.InputsFixMatch());
        break;
    }
    case 3:
    default:
    {
        if (mainInputs.InputsFixMatch())
        {
            Kernel_keeloq_main_three_inputs_fast<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());
        }
        else
        {
            Kernel_keeloq_main_three_inputs_slow<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());
        }

        break;
    }
    }


    mainInputs.read();
    kernel_results.read();

    return kernel_results;
}

__host__ bool keeloq::kernels::cuda_is_working()
{
    KernelResult kernel_results;
    Kernel_keeloq_test<<<1, 1 >>>(kernel_results.ptr());
    kernel_results.read();

    return kernel_results.error == 0 && kernel_results.value != 0;
}


__host__ EncParcel keeloq::GetOTA(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count, KeeloqLearning::Type learningType)
{
    serial &= 0x0FFFFFFF;

    uint32_t unencrypted = button << 28 | ((serial & 0x3FF) << 16) | count;

    uint64_t n_key = key;
    uint32_t fix = (uint32_t)button << 28 | serial;

    switch (learningType)
    {
    case KeeloqLearning::Type::Normal:
        n_key = keeloq_common_normal_learning(fix, key);
        break;
    case KeeloqLearning::Type::Xor:
        n_key = keeloq_common_magic_xor_type1_learning(serial, key);
        break;
    case KeeloqLearning::Type::Faac:
        n_key = keeloq_common_faac_learning(seed, key);
        break;
    default:
        assert(false && "Unsupported learning type");
        break;
    }

    uint64_t detpyrcne = ((uint64_t)fix << 32) | keeloq_common_encrypt(unencrypted, n_key);
    auto encrypted =  misc::rev_bits(detpyrcne, sizeof(detpyrcne) * 8);

#if _DEBUG
    SingleResult::DecryptedArray results = {};
    KeeloqLearning::Matrix matrix;
    matrix.enable(learningType);
    matrix.enable(learningType, KeeloqLearning::Mod::Type::ReversedKey); // rev as well

    Decryptor fwd_dec(key, seed);
    Decryptor rev_dec(fwd_dec.nam(), seed);

    // CPU
    keeloq_decrypt<LearningDecryptionMode::ExplicitAll>(encrypted, fwd_dec, matrix, results);
    assert(results.data[KeeloqLearning::Matrix::getIndex(learningType, KeeloqLearning::Mod::Type::Regular)] == unencrypted);

    keeloq_decrypt<LearningDecryptionMode::ExplicitAll>(encrypted, rev_dec, matrix, results);
    assert(results.data[KeeloqLearning::Matrix::getIndex(learningType, KeeloqLearning::Mod::Type::ReversedKey)] == unencrypted);

    // GPU
    KernelResult kernel_results;
    Kenrel_keeloq_single_check<<<1, 1 >>>(encrypted, fwd_dec.man(), fwd_dec.seed(), learningType, KeeloqLearning::Mod::Type::Regular, kernel_results.ptr());
    kernel_results.read();
    assert(kernel_results.value == unencrypted);

    Kenrel_keeloq_single_check<<<1, 1 >>>(encrypted, rev_dec.man(), rev_dec.seed(), learningType, KeeloqLearning::Mod::Type::ReversedKey, kernel_results.ptr());
    kernel_results.read();
    assert(kernel_results.value == unencrypted);
#endif

    return encrypted;
}
