#include "device/cuda_common.h"
#include "device/cuda_span.h"

#include "algorithm/keeloq/keeloq_single_result.h"

#include <cuda_runtime_api.h>

constexpr uint8_t OneEncInput = 1;
constexpr uint8_t TwoEncInputs = 2;
constexpr uint8_t ThreeEncInputs = 3;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        // No underflow in CUDA
        lrn_matches += __usad(expected_cnt, cnt, 0u) < counter_maxdiff;
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
    uint8_t lrn_matches = 1;

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
            match_res_index = (has_match * resIndex + !has_match * match_res_index);
        }

    }

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
__device__ KeeloqLearning::ResultIndex analyze_single_result(const SingleResult& result, uint32_t exp_srl, uint8_t exp_btn, const KeeloqLearning::Matrix& learnings_matrix)
{
    auto match_res_index = KeeloqLearning::NoMatch;

    // outer loop - over all learning types
    UNROLL
    for (uint8_t resIndex = 0; resIndex < KeeloqLearning::InvalidResultIndex; ++resIndex)
    {
        // No if block
        const bool allowed = learnings_matrix.isEnabled(resIndex);

        const uint32_t srl = result.decrypted.srl(resIndex);
        const uint32_t btn = result.decrypted.btn(resIndex);

        const bool has_match = allowed && srl == exp_srl && srl != 0 && btn == exp_btn;
        match_res_index = (has_match * resIndex + !has_match * match_res_index);
    }

    return match_res_index;
}

}

namespace
{

using namespace KeeloqLearning;

/**
 *  Single decryption call for specific learning type and modifier.
 * Returns decryption result.
 * Uses template parameters to optimize code and remove branches.
 * So for each learning type and modifier will be generated separate function.
 *
 *  call-stack:
 *   -> keeloq_encdec_single
 *      keeloq_encdec_multi
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, LearningType type, Modifier::Input IMod, Modifier::Algo AMod>
__device__ __host__ __forceinline__ uint32_t keeloq_encdec_single(uint32_t data, uint32_t fix, const Decryptor& decryptor)
{
    const uint64_t key = decryptor.template getKey<IMod == Modifier::Input::ReversedKey>();
    const uint32_t seed = decryptor.seed();

    static_assert(IMod == Modifier::Input::Normal || IMod == Modifier::Input::ReversedKey, "Unsupported/Unknown Keeloq Input Modifier");
    static_assert(AMod == Modifier::Algo::Normal || AMod == Modifier::Algo::Inverted, "Unsupported/Unknown Keeloq Algorithm Modifier");

    static_assert(
        type == LearningType::Simple    || type == LearningType::Normal || type == LearningType::Secure ||
        type == LearningType::Xor       || type == LearningType::Faac   || type == LearningType::Serial1 ||
        type == LearningType::Serial2   || type == LearningType::Serial3, "Unsupported KeeloqLearningType");

    static_assert(DecryptedResults::getIndex<type, IMod, AMod>() != KeeloqLearning::InvalidResultIndex, "Unsupported KeeloqLearningType/Mod combination");

    // Inverted logic for learning types that support it (Normal, Secure, Faac)
    if constexpr (AMod == Modifier::Algo::Inverted)
    {
        if constexpr (type == LearningType::Normal)
        {
            uint64_t n_key = keeloq::learning::normal<false>(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Secure)
        {
            uint64_t n_key = keeloq::learning::secure<false>(fix, seed, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Faac)
        {
            // Faac uses encrypt by default, Inverted logic here will be decrypt
            uint64_t n_key = keeloq::learning::faac<true>(seed, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
    }
    else if (AMod == Modifier::Algo::Normal)
    {
        if constexpr (type == LearningType::Simple)
        {
            return keeloq::common::encdec<IsDecrypt>(data, key);
        }
        else if constexpr (type == LearningType::Normal)
        {
            uint64_t n_key = keeloq::learning::normal(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Secure)
        {
            uint64_t n_key = keeloq::learning::secure(fix, seed, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Xor)
        {
            uint64_t n_key = keeloq::learning::magic_xor_type1(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Faac)
        {
            uint64_t n_key = keeloq::learning::faac(seed, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial1)
        {
            uint64_t n_key = keeloq::learning::serial_type1(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial2)
        {
            uint64_t n_key = keeloq::learning::serial_type2(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial3)
        {
            uint64_t n_key = keeloq::learning::serial_type3(fix, key);
            return keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
    }
}

/**
 *  Single decryption call wrapper for specific learning type and modification.
 * Result will be written to results array according to index from KeeloqLearning::DecryptedResults::getIndex() method.
 *
 *  call-stack:
 *   -> keeloq_encdec_single
 *      keeloq_encdec_multi
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, LearningType type, Modifier::Input IMod, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_single(uint32_t data, uint32_t fix, const Decryptor& decryptor, DecryptedResults& results)
{
    static constexpr auto index = DecryptedResults::getIndex<type, IMod, AMod>();
    if constexpr (index != KeeloqLearning::InvalidResultIndex)
    {
        results[index] = keeloq_encdec_single<IsDecrypt, type, IMod, AMod>(data, fix, decryptor);
    }
}

/**
 *  Unconditional param pack expansion call for single calls:
 *
 *  call-stack:
 *      keeloq_encdec_single
 *   -> keeloq_encdec_multi
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, Modifier::Input IMod, Modifier::Algo AMod, LearningType... LTypes>
__device__ __host__ __forceinline__ void keeloq_encdec_multi(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    DecryptedResults& results, LearningTypesSet<LTypes...>)
{
    ((keeloq_encdec_single<IsDecrypt, LTypes, IMod, AMod>(data, fix, decryptor, results)), ...);
}

/**
 *  Conditional param pack expansion call for single calls.
 * Uses if block, but in case of small number of learning types and modifications should be better than loop with if block inside.
 *
 *  call-stack:
 *      keeloq_encdec_single
 *   -> keeloq_encdec_multi_cond
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, Modifier::Input IMod, Modifier::Algo AMod, LearningType... LTypes>
__device__ __host__ __forceinline__ void keeloq_encdec_multi_cond(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, DecryptedResults& results, LearningTypesSet<LTypes...>)
{
    ((learnings_matrix.isEnabled(LTypes, IMod, AMod) ? keeloq_encdec_single<IsDecrypt, LTypes, IMod, AMod>(data, fix, decryptor, results) : void()), ...);
}

/**
 *  This function call unconditional or conditional multi call for seed learning types with specific modification.
 * In fact this is just magic wrapper for `keeloq_encdec_single` for Secure and Faac learning types
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, Modifier::Input IMod, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_seed_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, SingleResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, IMod, AMod>(data, fix, decryptor, decrypted.data, SeededTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, IMod, AMod>(data, fix, decryptor, learnings_matrix, decrypted.data, SeededTypes{});
    }
}

/**
 *  This function call unconditional or conditional multi call for non-seed (normal) learning types with specific modification.
 * In fact this is just magic wrapper for `keeloq_encdec_single` for Simple, Normal, Xor and Serial learning types
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, Modifier::Input IMod, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_normal_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, SingleResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, IMod, AMod>(data, fix, decryptor, decrypted.data, NormalTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, IMod, AMod>(data, fix, decryptor, learnings_matrix, decrypted.data, NormalTypes{});
    }
}

/**
 *  This function calls seeded and/or normal wrapper function for all modifications according to template parameters and learning matrix.
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *      keeloq_decrypt_[seed,normal]
 *   -> keeloq_encdec
 */
template<LearningDecryptionMode Mode, bool IsDecrypt = true>
__device__ __host__ inline void keeloq_encdec(const EncParcel& enc, const Decryptor& decryptor, const Matrix& learnings_matrix, SingleResult::LearningsArray& results)
{
    constexpr bool IgnoreMatrix     = !!(Mode & LearningDecryptionMode::Force);
    constexpr bool ExplicitDecrypt  = !!(Mode & LearningDecryptionMode::Explicit);

    static_assert(IgnoreMatrix != ExplicitDecrypt, "Can't be only Force or Explicit");

    // Modifiers that DISABLES specific calculations. Only used if not Force mode
    constexpr LearningDecryptionMode Modifiers = (Mode & (IgnoreMatrix ? static_cast<LearningDecryptionMode>(0) : LearningDecryptionMode::ModMask));

    // Normal learning types (NO SEED)
    if constexpr (!!(Mode & LearningDecryptionMode::Normal))
    {
        // If regular inputs allowed
        if constexpr (!(Modifiers & LearningDecryptionMode::NoReg))
        {
            keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, Modifier::Input::Normal, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

            // If inverted algo logic allowed
            if constexpr (!(Modifiers & LearningDecryptionMode::NoInv))
            {
                keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, Modifier::Input::Normal, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
            }
        }

        // If reversed inputs allowed
        if constexpr (!(Modifiers & LearningDecryptionMode::NoRev))
        {
            keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, Modifier::Input::ReversedKey, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

            // If inverted alog logic allowed
            if constexpr (!(Modifiers & LearningDecryptionMode::NoInv))
            {
                keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, Modifier::Input::ReversedKey, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
            }
        }

    }

    // Seeded learning types
    if constexpr (!!(Mode & LearningDecryptionMode::Seeded))
    {
        // If regular inputs allowed
        if constexpr (!(Modifiers & LearningDecryptionMode::NoReg))
        {
            keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, Modifier::Input::Normal, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

            // If inverted algo logic allowed
            if constexpr (!(Modifiers & LearningDecryptionMode::NoInv))
            {
                keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, Modifier::Input::Normal, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
            }
        }

        // If reversed inputs allowed
        if constexpr (!(Modifiers & LearningDecryptionMode::NoRev))
        {
            keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, Modifier::Input::ReversedKey, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

            // If inverted algo logic allowed
            if constexpr (!(Modifiers & LearningDecryptionMode::NoInv))
            {
                keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, Modifier::Input::ReversedKey, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
            }
        }
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
__device__ uint8_t inline keeloq_decrypt_and_analyze(const CudaContext& ctx, const Decryptor& decryptor, const Matrix& learning_matrix, Span<SingleResult>& results)
{
    static_assert(NumResults > 1, "Use `keeloq_decrypt_and_quick_analyze()` if results number is less than 2");
    static_assert(FirstResultIndex < NumResults, "Invalid template parameters!");

    // inner loop for each input for decryptor - make decryption
    UNROLL
    for (uint32_t i = FirstResultIndex; i < NumResults; ++i)
    {
        SingleResult& result = results[i];
        result.match = NoMatch;

        result.decryptor = decryptor;
        result.inputIndex = i;

        keeloq_encdec<LearningMode>(InputsCache[i], decryptor, learning_matrix, result.decrypted);
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
__device__ inline void keeloq_decrypt_and_quick_analyze(const CudaContext& ctx, const Decryptor& decryptor, const Matrix& learning_matrix, Span<SingleResult>& results)
{
    static_assert(InputIndex < 2, "You want Input index be 1 or 2 MOST LIKELY. Since last check should be the robust one");

    const EncParcel& enc = InputsCache[InputIndex];

    SingleResult& result = results[InputIndex];

    result.decryptor = decryptor;
    result.inputIndex = InputIndex;

    keeloq_encdec<LearningMode>(enc, decryptor, learning_matrix, result.decrypted);

    result.match = analyze_single_result(result, enc.srl(), enc.btn(), learning_matrix);
}


/**
 *  Run decryption parallel per thread and find matches
 * NumInputs - means how many inputs (encrypted OTA data) we have at all, single, two or three.
 *  depending on that num different optimization are applied deeper (like loop unrolling when writing results)
 * LearningMode - determines which learning calculation could be thrown away. (we could have no seed, that mean we don't need to even check ifs)
 * UseFastCheck - allows to early exit if first decrypted input doesn't match some bits from fixed part
 */
template<uint8_t NumInputs, LearningDecryptionMode LearningMode, bool UseFastCheck = true>
__device__ void inline keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelInput& input)
{
    static constexpr bool UseSlowCheck = NumInputs > 1 && !UseFastCheck;

    constexpr uint8_t First = 0;
    constexpr uint8_t Second = 1;
    constexpr uint8_t Third = 2;

    static_assert(NumInputs > 0 && NumInputs <= 3, "Invalid inputs number!");
    static_assert(First < Second && Second < Third, "Static assert just to get rid of warning");

    const auto inputsCount = input.inputsCount;
    const auto& decryptors = *input.decryptors;

    auto& all_results = *input.results;

    assert(inputsCount != 0 && "Number of encrypted inputs is ZERO!");
    assert(inputsCount >= NumInputs && "Number of encrypted inputs less than expected with template argument");
    assert(NumInputs == 1 || InputsCache[0].ota != InputsCache[1].ota && "Inputs are the same!");

    // outer loop for each thread's decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, decryptors.num)
    {
        const Decryptor& decryptor = decryptors[decryptor_index];

        Span<SingleResult> results(&all_results[decryptor_index * inputsCount], NumInputs);

        if constexpr (UseSlowCheck)
        {
            // only multiple input - decrypt all and then check
            keeloq_decrypt_and_analyze<NumInputs, LearningMode, First>(ctx, decryptor, input.GetLearningMatrix(), results);
        }
        else
        {
            // Single input
            keeloq_decrypt_and_quick_analyze<LearningMode, First>(ctx, decryptor, input.GetLearningMatrix(), results);

            // Multiple input
            if constexpr (NumInputs > 1)
            {
                assert(InputsCache[First].fix() == InputsCache[Second].fix() && "Cannot `UseFastCheck` if fixed part of encrypted packets are not the same!");

                // Even if this is `if` - still it's faster than calling decrypt for the rest of inputs
                if (results[First].match != KeeloqLearning::NoMatch)
                {
                    if constexpr (NumInputs > 2)
                    {
                        // Calling decrypt and fast check on next input
                        keeloq_decrypt_and_quick_analyze<LearningMode, Second>(ctx, decryptor, input.GetLearningMatrix(), results);

                        if (results[Second].match != KeeloqLearning::NoMatch)
                        {
                            // since 1-st and 2-nd was already decrypted - starting index is 2
                            // This will also reset the `result.match`
                            keeloq_decrypt_and_analyze<NumInputs, LearningMode, Third>(ctx, decryptor, input.GetLearningMatrix(), results);
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
                        keeloq_decrypt_and_analyze<NumInputs, LearningMode, Second>(ctx, decryptor, input.GetLearningMatrix(), results);
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
            const SingleResult* decryptor_results = &all_results[decryptor_index * KernelInputs->inputsCount];

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
NOINLINE __device__ void Kernel_keeloq_main(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();
    auto& results = *KernelInputs->results;

    const bool forceAll = KernelInputs->AllLearningsEnabled();

    const auto& config = KernelInputs->GetConfig();
    const auto& decryptors = *KernelInputs->decryptors;
    assert(decryptors.num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    const bool hasSeed = decryptors[ctx.thread_id].has_seed();
    const bool seedOnly = config.type == BruteforceType::Seed;

    if (seedOnly)
    {
        assert(hasSeed && "Seed must be present for seed-only mode, some problem in generator.");

        if (forceAll)
        {
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceSeeded, UseFastCheck>(ctx, *KernelInputs);
        }
        else
        {
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitSeeded, UseFastCheck>(ctx, *KernelInputs);
        }
    }
    else if (forceAll)
    {
        hasSeed ?
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceAll, UseFastCheck>(ctx, *KernelInputs) :
            keeloq_decryption_run<NumInputs, LearningDecryptionMode::ForceNormal, UseFastCheck>(ctx, *KernelInputs);
    }
    else if (hasSeed)
    {
        keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitAll, UseFastCheck>(ctx, *KernelInputs);
    }
    else
    {
        keeloq_decryption_run<NumInputs, LearningDecryptionMode::ExplicitNormal, UseFastCheck>(ctx, *KernelInputs);
    }


    uint8_t num_matches = keeloq_check_matches<NumInputs>(ctx, results, KernelInputs);
    ret->onKernelFinish(num_matches);
}

}

namespace
{
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_test(KernelResult::TCudaPtr ret)
{
    CudaContext ctx = CudaContext::Get();

    if (ctx.thread_id == 0)
    {
        const uint32_t pln_test = 0x11223344;
        const uint64_t key_test = 0xDEADBEEF00226688;
        const uint32_t enc_test = keeloq::common::encrypt(pln_test, key_test);

        const bool match = pln_test == keeloq::common::decrypt(enc_test, key_test);
        ret->onKernelFinish(match);
    }
}

__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_single_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, DecryptKernelResult::TCudaPtr res)
{
    if (isDecrypt)
    {
        keeloq_encdec<LearningDecryptionMode::ForceAll, true>(EncParcel(ota), Decryptor::Make(man, seed, true), KeeloqLearning::Matrix::Everything(), res->result.decrypted);
    }
    else
    {
        // specially prepared data for encryption, fix and hop
        EncParcel unencrypted(static_cast<uint32_t>(ota >> 32), static_cast<uint32_t>(ota));
        keeloq_encdec<LearningDecryptionMode::ForceAll, false>(unencrypted, Decryptor::Make(man, seed, true), KeeloqLearning::Matrix::Everything(), res->result.decrypted);
    }
}

__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_main_one_input(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    Kernel_keeloq_main<OneEncInput>(KernelInputs, ret);
}

__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_main_two_inputs(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret, bool fixed_parts_match)
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

__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_main_three_inputs_fast(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    Kernel_keeloq_main<ThreeEncInputs, true>(KernelInputs, ret);
}

__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_main_three_inputs_slow(KeeloqKernelInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    Kernel_keeloq_main<ThreeEncInputs, false>(KernelInputs, ret);
}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelInput& mainInputs, const CudaConfig& cuda)
{
    KernelResult kernel_results;

    switch (mainInputs.NumInputs())
    {
    case 1:
    {
        Kernel_keeloq_main_one_input<<<cuda.blocks, cuda.threads>>>(mainInputs.ptr(), kernel_results.ptr());
        break;
    }
    case 2:
    {
        Kernel_keeloq_main_two_inputs<<<cuda.blocks, cuda.threads >>>(mainInputs.ptr(), kernel_results.ptr(), mainInputs.InputsFixMatch());
        break;
    }
    case 3:
    default:
    {
        if (mainInputs.InputsFixMatch())
        {
            Kernel_keeloq_main_three_inputs_fast<<<cuda.blocks, cuda.threads >>>(mainInputs.ptr(), kernel_results.ptr());
        }
        else
        {
            Kernel_keeloq_main_three_inputs_slow<<<cuda.blocks, cuda.threads >>>(mainInputs.ptr(), kernel_results.ptr());
        }

        break;
    }
    }

    kernel_results.read();
    mainInputs.read();

    return kernel_results;
}

__host__ SingleResult keeloq::kernels::cuda_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt)
{
    DecryptKernelResult kernel_results;

    Kernel_keeloq_single_encdec<<<1, 1>>>(ota, man, seed, isDecrypt, kernel_results.ptr());

    kernel_results.read();
    return kernel_results.result;
}

__host__ bool keeloq::kernels::cuda_is_working()
{
    KernelResult kernel_results;
    Kernel_keeloq_test<<<1, 1>>>(kernel_results.ptr());
    kernel_results.read();

    return kernel_results.hasMatch() && kernel_results.threadsFinished() == 1;
}
