#include "device/cuda_common.h"
#include "device/cuda_span.h"

#include <cuda_runtime_api.h>

// For template functions - and compiler optimizations
constexpr bool EveryLearning = true;
constexpr bool MaskLearnings = false;

constexpr uint8_t OneEncInput = 1;
constexpr uint8_t TwoEncInputs = 2;
constexpr uint8_t ThreeEncInputs = 3;


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

    __device__ __host__ inline uint64_t keeloq_common_magic_xor_type1_learning(uint32_t data, uint64_t pxor)
    {
        data &= 0x0FFFFFFF;
        return (((uint64_t)data << 32) | data) ^ pxor;
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace debug
{
// Some problems with printf - looks like silently hit instruction limit
__device__ __noinline__ void assert_single_match_count(uint8_t match_count, bool has_match, KeeloqLearningType::Type curr, KeeloqLearningType::Type newone,
    const Span<SingleResult>& results)
{
    if (match_count > 0 && has_match)
    {
        auto ctx = CudaContext::Get();
        printf("get_match_learning(): %u, detected multiple match. Prev learning: %d Now learning: %d. Man: 0x%llX Inputs:[ 0x%llX; 0x%llX; 0x%llX]\n",
            ctx.thread_id,
            curr, newone,
            results[0].decryptor.man(),
            results[0].encrypted.ota, results[1].encrypted.ota, results[2].encrypted.ota);
    }
}
}


namespace
{

template<uint8_t NumInputs>
__device__ uint8_t is_cnt_match(const Span<SingleResult>& results, KeeloqLearningType::Type learning_type)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    uint8_t counter_maxdiff = NumInputs + 1;

    uint32_t expected_cnt = results[0].decrypted.cnt(learning_type);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        uint32_t cnt = results[item].decrypted.cnt(learning_type);
        lrn_matches += __usad(expected_cnt, cnt, 0) < counter_maxdiff;
    }

    return lrn_matches == NumInputs;
}

template<uint8_t NumInputs>
__device__ uint8_t is_btn_match(const Span<SingleResult>& results, KeeloqLearningType::Type learning_type)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    uint32_t expected_btn = results[0].decrypted.btn(learning_type);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        lrn_matches += results[item].decrypted.btn(learning_type) == expected_btn;
    }

    return lrn_matches == NumInputs;
}

template<uint8_t NumInputs>
__device__ uint8_t is_srl_match(const Span<SingleResult>& results, KeeloqLearningType::Type learning_type)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    uint32_t expected_srl = results[0].decrypted.srl(learning_type);
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        bool decryptor_valid = results[item].decryptor.is_valid();
        lrn_matches += decryptor_valid && results[item].decrypted.srl(learning_type) == expected_srl;
    }

    return lrn_matches == NumInputs && expected_srl != 0; // 0 check at the end to save instructions in loop
}

template<uint8_t NumInputs, bool AllLearnings>
__device__ KeeloqLearningType::Type get_match_learning(const Span<SingleResult>& results, const KeeloqLearningMask type_mask)
{
    uint8_t match_count = 0; // 0 or 1. if bigger - double match
    KeeloqLearningType::Type match_learning_type = KeeloqLearningType::INVALID;

    // outer loop - over all learning types
    UNROLL
    for (uint8_t lrn = 0; lrn < SingleResult::ResultsCount; ++lrn)
    {
        if (AllLearnings || type_mask[lrn])
        {
            bool has_match = is_srl_match<NumInputs>(results, lrn) &&
                is_btn_match<NumInputs>(results, lrn) &&
                is_cnt_match<NumInputs>(results, lrn);

        #if _DEBUG
            debug::assert_single_match_count(match_count, has_match, match_learning_type, lrn, results);
        #endif

            match_count += has_match;
            match_learning_type = (has_match * lrn + !has_match * match_learning_type);
        }
    }

    assert(match_count <= 1 && "analyze_results_srl() Multiple match - should not be possible!");
    return match_learning_type;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
template<uint8_t NumInputs, bool AllLearnings>
__device__ KeeloqLearningType::Type analyze_multiple_results(const CudaContext& ctx, Span<SingleResult>& results, const KeeloqLearningMask type_mask)
{
    KeeloqLearningType::Type learning_type =  get_match_learning<NumInputs, AllLearnings>(results, type_mask);

    UNROLL
    for (int i = 0; i < NumInputs; ++i)
    {
        // If no match will return INVALID - rewrite same value better than have if block
        results[i].match = learning_type;
    }

    return learning_type;
}

// In case of single input we checking fixed part of parcel's serial (28-bit serial | 4-bit button)
// with decoded serial
template<bool AllLearnings>
__device__ KeeloqLearningType::Type analyze_single_result(const SingleResult& result, uint32_t exp_srl, uint8_t exp_btn, const KeeloqLearningMask type_mask, uint8_t& match_count)
{
    match_count = 0; // 0 or 1. if bigger - double match
    KeeloqLearningType::Type match_learning_type = KeeloqLearningType::INVALID;

    // outer loop - over all learning types
    UNROLL
    for (uint8_t lrn = 0; lrn < SingleResult::ResultsCount; ++lrn)
    {
        // Allow compiler optimize it
        bool allowed = AllLearnings || type_mask[lrn] != 0;

        uint32_t srl = result.decrypted.srl(lrn);
        uint32_t btn = result.decrypted.btn(lrn);

        bool has_match = allowed && srl == exp_srl && srl != 0 && btn == exp_btn;

        match_count += has_match;
        match_learning_type = (has_match * lrn + !has_match * match_learning_type);
    }

    return match_learning_type;
}

}

namespace
{

template<KeeloqLearningType::Type type>
__device__ __host__ inline uint32_t keeloq_decrypt_single(uint32_t data, uint32_t fix, const Decryptor& decryptor)
{
    uint64_t key = decryptor.man();
    uint64_t key_rev = decryptor.nam();
    uint64_t n_key = 0;
    uint32_t seed = decryptor.seed();

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

template<bool AllLearnings>
__device__ __host__ inline void keeloq_decrypt_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const KeeloqLearningType::Type type_mask[], SingleResult::DecryptedArray& decrypted)
{
    uint64_t key = decryptor.man();
    uint64_t key_rev = decryptor.nam();
    uint64_t n_key = 0;
    uint32_t seed = decryptor.seed();

    // Simple Learning
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Simple])
        {
            decrypted.data[KeeloqLearningType::Simple] = keeloq_common_decrypt(data, key);
        }
        if (AllLearnings || type_mask[KeeloqLearningType::Simple_Rev])
        {
            decrypted.data[KeeloqLearningType::Simple_Rev] = keeloq_common_decrypt(data, key_rev);
        }
    }
    //###########################
    // Normal Learning
    // https://phreakerclub.com/forum/showpost.php?p=43557&postcount=37
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Normal])
        {
            n_key = keeloq_common_normal_learning(fix, key);
            decrypted.data[KeeloqLearningType::Normal] = keeloq_common_decrypt(data, n_key);
        }
        if (AllLearnings || type_mask[KeeloqLearningType::Normal_Rev])
        {
            n_key = keeloq_common_normal_learning(fix, key_rev);
            decrypted.data[KeeloqLearningType::Normal_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // Secure Learning
    {
        if ((AllLearnings && seed != 0) || type_mask[KeeloqLearningType::Secure])
        {
            n_key = keeloq_common_secure_learning(fix, seed, key);
            decrypted.data[KeeloqLearningType::Secure] = keeloq_common_decrypt(data, n_key);
        }

        if ((AllLearnings && seed != 0) || type_mask[KeeloqLearningType::Secure_Rev])
        {
            n_key = keeloq_common_secure_learning(fix, seed, key_rev);
            decrypted.data[KeeloqLearningType::Secure_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // Magic xor type1 learning
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Xor])
        {
            n_key = keeloq_common_magic_xor_type1_learning(fix, key);
            decrypted.data[KeeloqLearningType::Xor] = keeloq_common_decrypt(data, n_key);
        }

        if (AllLearnings || type_mask[KeeloqLearningType::Xor_Rev])
        {
            n_key = keeloq_common_magic_xor_type1_learning(fix, key_rev);
            decrypted.data[KeeloqLearningType::Xor_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // FAAC
    {
        if ((AllLearnings && seed != 0) || type_mask[KeeloqLearningType::Faac])
        {
            n_key = keeloq_common_faac_learning(seed, key);
            decrypted.data[KeeloqLearningType::Faac] = keeloq_common_decrypt(data, n_key);
        }

        if ((AllLearnings && seed != 0) || type_mask[KeeloqLearningType::Faac_Rev])
        {
            n_key = keeloq_common_faac_learning(seed, key_rev);
            decrypted.data[KeeloqLearningType::Faac_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // SERIAL_TYPE_1
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Serial1])
        {
            n_key = keeloq_common_magic_serial_type1_learning(fix, key);
            decrypted.data[KeeloqLearningType::Serial1] = keeloq_common_decrypt(data, n_key);
        }

        if (AllLearnings || type_mask[KeeloqLearningType::Serial1_Rev])
        {
            n_key = keeloq_common_magic_serial_type1_learning(fix, key_rev);
            decrypted.data[KeeloqLearningType::Serial1_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // SERIAL_TYPE_2
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Serial2])
        {
            n_key = keeloq_common_magic_serial_type2_learning(fix, key);
            decrypted.data[KeeloqLearningType::Serial2] = keeloq_common_decrypt(data, n_key);
        }

        if (AllLearnings || type_mask[KeeloqLearningType::Serial2_Rev])
        {
            n_key = keeloq_common_magic_serial_type2_learning(fix, key_rev);
            decrypted.data[KeeloqLearningType::Serial2_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }

    // SERIAL_TYPE_3
    {
        if (AllLearnings || type_mask[KeeloqLearningType::Serial3])
        {
            n_key = keeloq_common_magic_serial_type3_learning(fix, key);
            decrypted.data[KeeloqLearningType::Serial3] = keeloq_common_decrypt(data, n_key);
        }

        if (AllLearnings || type_mask[KeeloqLearningType::Serial3_Rev])
        {
            n_key = keeloq_common_magic_serial_type3_learning(fix, key_rev);
            decrypted.data[KeeloqLearningType::Serial3_Rev] = keeloq_common_decrypt(data, n_key);
        }
    }
}

template<bool AllLearnings>
__device__ __host__ inline void keeloq_decrypt(const EncParcel& enc, const Decryptor& decryptor, const KeeloqLearningMask type_mask, SingleResult::DecryptedArray& results)
{
    keeloq_decrypt_all<AllLearnings>(enc.hop(), enc.fix(), decryptor, type_mask, results);
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
template<uint8_t NumResults, bool AllLearnings, uint8_t FirstResult = 0>
__device__ uint8_t inline keeloq_decrypt_and_analyze(const CudaContext& ctx,
    const CudaArray<EncParcel>& encrypted, const Decryptor& decryptor, const KeeloqLearningMask learnings, Span<SingleResult>& results)
{
    if constexpr (NumResults == 1)
    {
        const EncParcel& enc = encrypted[0];

        uint8_t match_count = 0;
        SingleResult& result = results[0];

        result.decryptor = decryptor;
        result.encrypted = enc; // useless copy

        keeloq_decrypt<AllLearnings>(enc, decryptor, learnings, result.decrypted);

        result.match = analyze_single_result<AllLearnings>(result, enc.srl(), enc.btn(), learnings, match_count);

        // In case of single input that means that key has 2 (most probably) phantoms
        return match_count > 1;
    }
    else
    {
        assert(encrypted.num >= NumResults && "Encrypted array size doesn't match results array!");

        // inner loop for each input for decryptor - make decryption
        UNROLL
        for (uint32_t i = FirstResult; i < NumResults; ++i)
        {
            SingleResult& result = results[i];
            result.match = KeeloqLearningType::INVALID;

            result.decryptor = decryptor;
            result.encrypted = encrypted[i];

            keeloq_decrypt<AllLearnings>(result.encrypted, decryptor, learnings, result.decrypted);
        }

        // now check all decrypted results if they match somehow
        analyze_multiple_results<NumResults, AllLearnings>(ctx, results, learnings);
        return 0;
    }
}


// run decryption parallel per thread and find matches
template<uint8_t NumInputs, bool AllLearnings, bool UseFastCheck = true>
__device__ uint8_t inline keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelInput& input)
{
    static_assert(NumInputs > 0 && NumInputs <= 3, "Invalid inputs number!");

    constexpr bool UseFullCheck = NumInputs > 1 && !UseFastCheck;

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
            keeloq_decrypt_and_analyze<NumInputs, AllLearnings, 0>(ctx, encrypted, decryptor, input.learning_types, results);
        }
        else
        {
            // Single input
            auto multiple_match = keeloq_decrypt_and_analyze<1, AllLearnings>(ctx, encrypted, decryptor, input.learning_types, results);
            if constexpr (NumInputs == 1)
            {
                // Single mode only. in multiple - it's ok, we'll check just the rest
                result_error += multiple_match;
            }

            // Multiple input
            if constexpr (NumInputs >= 2)
            {
                assert(encrypted[0].fix() == encrypted[1].fix() && "Cannot `UseFastCheck` if fixed part of encrypted packets are not the same!");

                SingleResult& first_result = results[0];

                // Even if this is `if` - still it's faster
                if (first_result.match == KeeloqLearningType::INVALID)
                {
                    UNROLL
                    for (uint32_t i = 1; i < NumInputs; ++i)
                    {
                        results[i].match = KeeloqLearningType::INVALID;
                    }
                }
                else
                {
                    // since 1-st was already decrypted - starting index is 1
                    keeloq_decrypt_and_analyze<NumInputs, AllLearnings, 1>(
                        ctx, encrypted, decryptor, input.learning_types, results);
                }
            }
        }
    }

    return result_error;
}

// aggregate matches into count
template<uint8_t NumInputs>
__device__ uint8_t inline keeloq_analyze_results(const CudaContext& ctx, const CudaArray<SingleResult>& all_results, const KeeloqKernelInput::TCudaPtr KernelInputs)
{
    uint8_t num_matches = 0;

    // outer loop for thread decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, KernelInputs->decryptors->num)
    {
        if constexpr (NumInputs == 1)
        {
            num_matches += (all_results[decryptor_index].match != KeeloqLearningType::INVALID);
        }
        else
        {
            const SingleResult* decryptor_results = &all_results[decryptor_index * KernelInputs->encdata->num];

            // inner loop for each result of this decryptor
            UNROLL
            for (uint8_t r = 0; r < NumInputs; ++r)
            {
                num_matches += (decryptor_results[r].match != KeeloqLearningType::INVALID);
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

    bool all_learnings = KeeloqLearningType::AllEnabled(KernelInputs->learning_types);

    uint8_t num_errors = all_learnings ?
        keeloq_decryption_run<NumInputs, EveryLearning, UseFastCheck>(ctx, *KernelInputs) :
        keeloq_decryption_run<NumInputs, MaskLearnings, UseFastCheck>(ctx, *KernelInputs);

    uint8_t num_matches = keeloq_analyze_results<NumInputs>(ctx, results, KernelInputs);

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


__global__ void Kenrel_keeloq_single_check(uint64_t ota, uint64_t man, uint32_t seed, KeeloqLearningType::Type learning, KernelResult::TCudaPtr ret)
{
    SingleResult::DecryptedArray decrypted;

    EncParcel enc(ota);
    Decryptor decryptor(man, seed);
    KeeloqAllLearningsMask all;

    // printf("OTA: 0x%llX. Man: 0x%llX, seed: %u, learning: %d\n", ota, man, seed, learning);

    keeloq_decrypt<EveryLearning>(enc, decryptor, all.mask, decrypted);

    ret->value = decrypted.data[learning];
}

// Main kernel for keeloq decryptions if provided several enc parcels
__global__ void Kernel_keeloq_main_multi(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    const CudaArray<EncParcel>& encdata = *KernelInputs->encdata;
    uint32_t encrypted_num = (uint32_t)encdata.num;

    switch (encrypted_num)
    {
    case 1:
    {
        Kernel_keeloq_main<OneEncInput>(KernelInputs, ret);
        break;
    }
    case 2:
    {
        if (encdata[0].fix() == encdata[1].fix())
        {
            Kernel_keeloq_main<TwoEncInputs, true>(KernelInputs, ret);
        }
        else
        {
            // Optimizations fallback
            // fixed parts not matched - cant use fast check
            Kernel_keeloq_main<TwoEncInputs, false>(KernelInputs, ret);
        }
        break;
    }
    case 3:
    default:
    {
        if (encdata[0].fix() == encdata[1].fix() && encdata[1].fix() == encdata[2].fix())
        {
            Kernel_keeloq_main<ThreeEncInputs, true>(KernelInputs, ret);
        }
        else
        {
            // Optimizations fallback
            // fixed parts not matched - cant use fast check
            Kernel_keeloq_main<ThreeEncInputs, false>(KernelInputs, ret);
        }
        break;
    }
    }
}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock)
{
    KernelResult kernel_results;

    Kernel_keeloq_main_multi<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), kernel_results.ptr());

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


__host__ EncParcel keeloq::GetOTA(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count, KeeloqLearningType::Type learning)
{
    serial &= 0x0FFFFFFF;

    uint32_t unencrypted = button << 28 | ((serial & 0x3FF) << 16) | count;

    uint64_t n_key = key;
    uint32_t fix = (uint32_t)button << 28 | serial;

    switch (learning)
    {
    case KeeloqLearningType::Normal:
        n_key = keeloq_common_normal_learning(fix, key);
        break;
    case KeeloqLearningType::Xor:
        n_key = keeloq_common_magic_xor_type1_learning(serial, key);
        break;
    case KeeloqLearningType::Faac:
        n_key = keeloq_common_faac_learning(seed, key);
        break;
    default:
        break;
    }

    uint64_t detpyrcne = ((uint64_t)fix << 32) | keeloq_common_encrypt(unencrypted, n_key);
    auto encrypted =  misc::rev_bits(detpyrcne, sizeof(detpyrcne) * 8);

#if _DEBUG
    SingleResult::DecryptedArray results;
    uint8_t mask[KeeloqLearningType::ALL] = { 0 };
    mask[learning] = 1;
    mask[learning + 1] = 1; // rev as well

    Decryptor fwd_dec(key, seed);
    Decryptor rev_dec(fwd_dec.nam(), seed);

    // CPU
    keeloq_decrypt<MaskLearnings>(encrypted, fwd_dec, mask, results);
    assert(results.data[learning] == unencrypted);

    keeloq_decrypt<MaskLearnings>(encrypted, rev_dec, mask, results);
    assert(results.data[learning + 1] == unencrypted);

    // GPU
    KernelResult kernel_results;
    Kenrel_keeloq_single_check<<<1, 1 >>>(encrypted, fwd_dec.man(), fwd_dec.seed(), learning, kernel_results.ptr());
    kernel_results.read();
    assert(kernel_results.value == unencrypted);

    Kenrel_keeloq_single_check<<<1, 1 >>>(encrypted, rev_dec.man(), rev_dec.seed(), learning + 1, kernel_results.ptr());
    kernel_results.read();
    assert(kernel_results.value == unencrypted);
#endif

    return encrypted;
}
