#include "device/cuda_common.h"
#include "device/cuda_span.h"

#include "algorithm/keeloq/keeloq_thread_result.h"

#include <array>

#include <cuda_runtime_api.h>

static constexpr uint8_t OneEncInput = 1;
static constexpr uint8_t TwoEncInputs = 2;
static constexpr uint8_t ThreeEncInputs = 3;

static constexpr uint8_t First = 0;
static constexpr uint8_t Second = 1;
static constexpr uint8_t Third = 2;


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

template<uint8_t NumInputs>
__device__ uint8_t is_cnt_match(const Span<ThreadResult::Multi>& results, KeeloqLearning::ResultIndex resIndex)
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
__device__ uint8_t is_btn_match(const Span<ThreadResult::Multi>& results, KeeloqLearning::ResultIndex resIndex)
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
__device__ uint8_t is_srl_match(const Span<ThreadResult::Multi>& results, KeeloqLearning::ResultIndex resIndex)
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

template<uint8_t NumInputs, KernelLearningMode LearningMode>
__device__ KeeloqLearning::ResultIndex get_match_index(const Span<ThreadResult::Multi>& results, const KeeloqLearning::Matrix& learnings_matrix)
{
    static constexpr bool IgnoreMatrix = !!(LearningMode & KernelLearningMode::Force);

    static constexpr bool NormalTypes = !!(LearningMode & KernelLearningMode::Normal);
    static constexpr bool SeedTypes = !!(LearningMode & KernelLearningMode::Seeded);

    KeeloqLearning::ResultIndex match_res_index = KeeloqLearning::NoMatch;

    if constexpr (SeedTypes)
    {
        static constexpr auto SeedIndicesSize = KeeloqLearning::DecryptedResults::SeededIndicesNum;
        static constexpr auto SeedIndicesCache = KeeloqLearning::DecryptedResults::GetSeededIndicesCache();

        UNROLL
        for (uint8_t lIndex = 0; lIndex < SeedIndicesSize; ++lIndex)
        {
            const uint8_t resIndex = SeedIndicesCache[lIndex];

            // Compiler should optimize this block and throw away `if` check in case of ForceAllLearningTypes = true
            const bool allowed = IgnoreMatrix || learnings_matrix.isEnabled(resIndex);

            if (allowed)
            {
                const bool has_match =
                    is_srl_match<NumInputs>(results, resIndex) &&
                    is_btn_match<NumInputs>(results, resIndex) &&
                    is_cnt_match<NumInputs>(results, resIndex);

                match_res_index = (has_match * resIndex + !has_match * match_res_index);
            }
        }
    }

    if constexpr (NormalTypes)
    {
        static constexpr auto NormalIndicesSize = KeeloqLearning::DecryptedResults::NormalIndicesNum;
        static constexpr auto NormalIndicesCache = KeeloqLearning::DecryptedResults::GetNormalIndicesCache();

        UNROLL
        for (uint8_t lIndex = 0; lIndex < NormalIndicesSize; ++lIndex)
        {
            const uint8_t resIndex = NormalIndicesCache[lIndex];

            // Compiler should optimize this block and throw away `if` check in case of ForceAllLearningTypes = true
            const bool allowed = IgnoreMatrix || learnings_matrix.isEnabled(resIndex);

            if (allowed)
            {
                const bool has_match =
                    is_srl_match<NumInputs>(results, resIndex) &&
                    is_btn_match<NumInputs>(results, resIndex) &&
                    is_cnt_match<NumInputs>(results, resIndex);

                match_res_index = (has_match * resIndex + !has_match * match_res_index);
            }
        }
    }

    return match_res_index;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// run from result[0] to result[num] tries to detect if there is a match (man key valid)
template<uint8_t NumInputs, KernelLearningMode LearningMode>
__device__ KeeloqLearning::ResultIndex analyze_multiple_results(const CudaContext& ctx, Span<ThreadResult::Multi>& results, const KeeloqLearning::Matrix& learnings_matrix)
{
    auto match_res_indx = get_match_index<NumInputs, LearningMode>(results, learnings_matrix);

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
template<KernelLearningMode LearningMode>
__device__ KeeloqLearning::ResultIndex analyze_single_result(const ThreadResult::Multi& result, uint32_t exp_srl, uint8_t exp_btn, const KeeloqLearning::Matrix& learnings_matrix)
{
    static constexpr bool IgnoreMatrix = !!(LearningMode & KernelLearningMode::Force);

    static constexpr bool NormalTypes = !!(LearningMode & KernelLearningMode::Normal);
    static constexpr bool SeedTypes = !!(LearningMode & KernelLearningMode::Seeded);

    auto match_res_index = KeeloqLearning::NoMatch;


    if constexpr (SeedTypes)
    {
        static constexpr auto SeedIndicesSize = KeeloqLearning::DecryptedResults::SeededIndicesNum;
        static constexpr auto SeedIndicesCache = KeeloqLearning::DecryptedResults::GetSeededIndicesCache();

        UNROLL
        for (uint8_t lIndex = 0; lIndex < SeedIndicesSize; ++lIndex)
        {
            const uint8_t resIndex = SeedIndicesCache[lIndex];

            // Since `allowed` is uniform across the warp, an if block is cheaper
            const bool allowed = IgnoreMatrix || learnings_matrix.isEnabled(resIndex);
            if (allowed)
            {
                const uint32_t srl = result.decrypted.srl(resIndex);
                const uint8_t btn = result.decrypted.btn(resIndex);

                const bool has_match = srl == exp_srl && srl != 0 && btn == exp_btn;
                match_res_index = (has_match * resIndex + !has_match * match_res_index);
            }
        }
    }

    if constexpr (NormalTypes)
    {
        static constexpr auto NormalIndicesSize = KeeloqLearning::DecryptedResults::NormalIndicesNum;
        static constexpr auto NormalIndicesCache = KeeloqLearning::DecryptedResults::GetNormalIndicesCache();

        UNROLL
        for (uint8_t lIndex = 0; lIndex < NormalIndicesSize; ++lIndex)
        {
            const uint8_t resIndex = NormalIndicesCache[lIndex];

            // Since `allowed` is uniform across the warp, an if block is cheaper
            const bool allowed = IgnoreMatrix || learnings_matrix.isEnabled(resIndex);
            if (allowed)
            {
                const uint32_t srl = result.decrypted.srl(resIndex);
                const uint8_t btn = result.decrypted.btn(resIndex);
                const bool has_match = srl == exp_srl && srl != 0 && btn == exp_btn;
                match_res_index = (has_match * resIndex + !has_match * match_res_index);
            }
        }
    }

    return match_res_index;
}

}

namespace
{

using namespace KeeloqLearning;

/**
 *  Helper templates to convert between InputTransform bitmask and KernelLearningMode bitmask.
 * This converts to Kernel's data
 */
template<InputTransform Mask>
struct InputTransformToKernelMode
{
    static constexpr KernelLearningMode value =
        (has_flag(Mask, InputTransform::RevKey) ? KernelLearningMode::RevKey : KernelLearningMode::NoInputTransform) |
        (has_flag(Mask, InputTransform::XorFix) ? KernelLearningMode::XorFix : KernelLearningMode::NoInputTransform);
};

/**
 *  Helper templates to convert between KernelLearningMode bitmask and InputTransform bitmask.
 * This converts from Kernel's data to Host's data
 */
template<KernelLearningMode Mode>
struct KernelModeToInputTransform
{
    static constexpr InputTransform value = static_cast<InputTransform>(
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::RevKey)) ? static_cast<uint8_t>(InputTransform::RevKey) : 0) |
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::XorFix)) ? static_cast<uint8_t>(InputTransform::XorFix) : 0));
};

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
template<bool IsDecrypt, InputTransform InputsMut, LearningType type, Modifier::Algo AMod>
__device__ __host__ __forceinline__ uint32_t keeloq_encdec_single(uint32_t hop, uint32_t fixed, const Decryptor& decryptor)
{
    const uint64_t key = decryptor.template getKey<!!(InputsMut & InputTransform::RevKey)>();
    const uint32_t fix = decryptor.template getXored<!!(InputsMut & InputTransform::XorFix)>(fixed);
    const uint32_t data = hop;
    const uint32_t seed = decryptor.seed();

    static_assert(AMod == Modifier::Algo::Normal || AMod == Modifier::Algo::Inverted, "Unsupported/Unknown Keeloq Algorithm Modifier");

    static_assert(
        type == LearningType::Simple    || type == LearningType::Normal || type == LearningType::Secure ||
        type == LearningType::Xor       || type == LearningType::Faac   || type == LearningType::Serial1 ||
        type == LearningType::Serial2   || type == LearningType::Serial3, "Unsupported KeeloqLearningType");

    static_assert(DecryptedResults::getIndex<type, AMod>() != KeeloqLearning::InvalidResultIndex, "Unsupported KeeloqLearningType/Mod combination");

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
template<bool IsDecrypt, InputTransform InputsMut, LearningType type, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_single(uint32_t data, uint32_t fix, const Decryptor& decryptor, DecryptedResults& results)
{
    static constexpr auto index = DecryptedResults::getIndex<type, AMod>();
    if constexpr (index != KeeloqLearning::InvalidResultIndex)
    {
        results[index] = keeloq_encdec_single<IsDecrypt, InputsMut, type, AMod>(data, fix, decryptor);
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
template<bool IsDecrypt, InputTransform InputsMut, Modifier::Algo AMod, LearningType... LTypes>
__device__ __host__ __forceinline__ void keeloq_encdec_multi(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    DecryptedResults& results, ValuesSet<LearningType, LTypes...>)
{
    ((keeloq_encdec_single<IsDecrypt, InputsMut, LTypes, AMod>(data, fix, decryptor, results)), ...);
}

/**
 *  Conditional param pack expansion call for single calls.
 * Uses if block, but in case of small number of learning types and modifications should be better than loop with if block inside.
 *
 *  call-stack:
 *      keeloq_encdec_single
 *   -> keeloq_encdec_multi_cond
 *      keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IsDecrypt, InputTransform InputsMut, Modifier::Algo AMod, LearningType... LTypes>
__device__ __host__ __forceinline__ void keeloq_encdec_multi_cond(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, DecryptedResults& results, ValuesSet<LearningType, LTypes...>)
{
    ((learnings_matrix.template isEnabled<LTypes, AMod>() ? keeloq_encdec_single<IsDecrypt, InputsMut, LTypes, AMod>(data, fix, decryptor, results) : void()), ...);
}

/**
 *  This function call unconditional or conditional multi call for seed learning types with specific modification.
 * In fact this is just magic wrapper for `keeloq_encdec_single` for Secure and Faac learning types
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, InputTransform InputsMut, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_seed_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, decrypted.data, SeededTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, learnings_matrix, decrypted.data, SeededTypes{});
    }
}

/**
 *  This function call unconditional or conditional multi call for all (seeded and normal) learning types with specific modification.
 * In fact this is just magic wrapper for `keeloq_encdec_single`
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, InputTransform InputsMut, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, decrypted.data, EveryLearningType{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, learnings_matrix, decrypted.data, EveryLearningType{});
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
template<bool IgnoreMatrix, bool IsDecrypt, InputTransform InputsMut, Modifier::Algo AMod>
__device__ __host__ __forceinline__ void keeloq_encdec_normal_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, decrypted.data, NormalTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AMod>(data, fix, decryptor, learnings_matrix, decrypted.data, NormalTypes{});
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
template<KernelLearningMode Mode, bool IsDecrypt = true>
__device__ __host__ inline void keeloq_encdec(const EncParcel& enc, const Decryptor& decryptor, const Matrix& learnings_matrix, ThreadResult::LearningsArray& results)
{
    static constexpr bool IgnoreMatrix     = !!(Mode & KernelLearningMode::Force);
    static constexpr bool ExplicitDecrypt  = !!(Mode & KernelLearningMode::Explicit);

    static_assert(IgnoreMatrix != ExplicitDecrypt, "Can't be only Force or Explicit");

    // Modifiers that DISABLES specific calculations. Only used if not Force mode
    constexpr KernelLearningMode Modifiers = (Mode & (IgnoreMatrix ? static_cast<KernelLearningMode>(0) : KernelLearningMode::ModMask));
    constexpr InputTransform InputsMut = KernelModeToInputTransform<Mode>::value;

    // Normal learning types (NO SEED)
    if constexpr (!!(Mode & KernelLearningMode::Normal))
    {
        keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, InputsMut, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

        // If inverted algo logic allowed
        if constexpr (!(Modifiers & KernelLearningMode::NoInv))
        {
            keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, InputsMut, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
        }
    }

    // Seeded learning types
    if constexpr (!!(Mode & KernelLearningMode::Seeded))
    {
        keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, InputsMut, Modifier::Algo::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

        // If inverted algo logic allowed
        if constexpr (!(Modifiers & KernelLearningMode::NoInv))
        {
            keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, InputsMut, Modifier::Algo::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
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
template<uint8_t NumResults, KernelLearningMode LearningMode, uint8_t FirstResultIndex>
__device__ uint8_t inline keeloq_decrypt_and_analyze(const CudaContext& ctx, const Decryptor& decryptor, const Matrix& learning_matrix, Span<ThreadResult::Multi>& results)
{
    static_assert(NumResults > 1, "Use `keeloq_decrypt_and_quick_analyze()` if results number is less than 2");
    static_assert(FirstResultIndex < NumResults, "Invalid template parameters!");

    // inner loop for each input for decryptor - make decryption
    UNROLL
    for (uint32_t i = FirstResultIndex; i < NumResults; ++i)
    {
        ThreadResult::Multi& result = results[i];
        result.match = NoMatch;

        result.decryptor = decryptor;
        result.setInputIndex(i);
        result.setInputTransform(KernelModeToInputTransform<LearningMode>::value);

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
template<KernelLearningMode LearningMode, uint8_t InputIndex>
__device__ inline void keeloq_decrypt_and_quick_analyze(const CudaContext& ctx, const Decryptor& decryptor, const Matrix& learning_matrix, Span<ThreadResult::Multi>& results)
{
    static_assert(InputIndex < 2, "You want Input index be 1 or 2 MOST LIKELY. Since last check should be the robust one");

    const EncParcel& enc = InputsCache[InputIndex];

    ThreadResult::Multi& result = results[InputIndex];

    result.decryptor = decryptor;
    result.setInputIndex(InputIndex);
    result.setInputTransform(KernelModeToInputTransform<LearningMode>::value);

    keeloq_encdec<LearningMode>(enc, decryptor, learning_matrix, result.decrypted);

    result.match = analyze_single_result<LearningMode>(result, enc.srl(), enc.btn(), learning_matrix);
}

template<uint8_t InputIndex, InputTransform InputMut, LearningType LType, Modifier::Algo AMod>
__device__ __forceinline__ bool keeloq_decrypt_single_learning(const CudaContext& ctx, const Decryptor& decryptor, ThreadResult::Single& result)
{
    const EncParcel& enc = InputsCache[InputIndex];

    result.decryptor = decryptor;
    result.setInputIndex(InputIndex);
    result.setInputTransform(InputMut);

    result.decryptor = decryptor;
    result.setInputIndex(InputIndex);
    result.setInputTransform(InputMut);

    static constexpr bool IsDecrypt = true;
    result.decrypted = keeloq_encdec_single<IsDecrypt, InputMut, LType, AMod>(enc.hop(), enc.fix(), decryptor);

    const bool match = result.srl() == enc.srl() && result.btn() == enc.btn();

    result.setHasMatch(match);
    return match;
}


/**
 *  Run decryption parallel per thread and find matches
 * NumInputs - means how many inputs (encrypted OTA data) we have at all, single, two or three.
 *  depending on that num different optimization are applied deeper (like loop unrolling when writing results)
 * LearningMode - determines which learning calculation could be thrown away. (we could have no seed, that mean we don't need to even check ifs)
 */
template<uint8_t NumInputs, KernelLearningMode LearningMode>
__device__ void inline keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelMultiLearningInput& input)
{
    static_assert(NumInputs > 0 && NumInputs <= 3, "Invalid inputs number!");
    static_assert(First < Second && Second < Third, "Static assert just to get rid of warning");

    const auto inputsCount = input.inputsCount;
    const auto& decryptors = *input.decryptors;
    const auto& learning_matrix = input.GetLearningMatrix();

    auto& all_results = *input.results;

    assert(inputsCount != 0 && "Number of encrypted inputs is ZERO!");
    assert(inputsCount >= NumInputs && "Number of encrypted inputs less than expected with template argument");
    assert(NumInputs == 1 || InputsCache[0].ota != InputsCache[1].ota && "Inputs are the same!");

    // outer loop for each thread's decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, decryptors.num)
    {
        const Decryptor& decryptor = decryptors[decryptor_index];

        Span<ThreadResult::Multi> results(&all_results[decryptor_index * inputsCount], NumInputs);

        // Single input
        keeloq_decrypt_and_quick_analyze<LearningMode, First>(ctx, decryptor, learning_matrix, results);

        // Multiple input
        if constexpr (NumInputs > 1)
        {
            if (results[First].match != KeeloqLearning::NoMatch)
            {
                if constexpr (NumInputs > 2)
                {
                    keeloq_decrypt_and_quick_analyze<LearningMode, Second>(ctx, decryptor, learning_matrix, results);

                    if (results[Second].match != KeeloqLearning::NoMatch)
                    {
                        keeloq_decrypt_and_analyze<NumInputs, LearningMode, Third>(ctx, decryptor, learning_matrix, results);
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
                    keeloq_decrypt_and_analyze<NumInputs, LearningMode, Second>(ctx, decryptor, learning_matrix, results);
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

// aggregate matches into count
template<uint8_t NumInputs>
__device__ uint8_t inline keeloq_check_matches(const CudaContext& ctx, const KeeloqKernelMultiLearningInput::TCudaPtr KernelInputs)
{
    uint8_t num_matches = 0;

    auto& results = *KernelInputs->results;

    // outer loop for thread decryptor
    KEELOQ_INNER_LOOP(ctx, decryptor_index, KernelInputs->decryptors->num)
    {
        if constexpr (NumInputs == 1)
        {
            num_matches += (results[decryptor_index].match != KeeloqLearning::NoMatch);
        }
        else
        {
            const ThreadResult::Multi* decryptor_results = &results[decryptor_index * KernelInputs->inputsCount];

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

template<InputTransform InputsMask>
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_single_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, DecryptKernelResult::TCudaPtr res)
{
    static_assert(is_valid(InputsMask), "Invalid input transform mask");
    constexpr KernelLearningMode InputsMode = InputTransformToKernelMode<InputsMask>::value;

    if (isDecrypt)
    {
        keeloq_encdec<KernelLearningMode::ForceAll | InputsMode, true>(EncParcel(ota), Decryptor::Make(man, seed, true), KeeloqLearning::Matrix::Everything(), res->result.decrypted);
    }
    else
    {
        // specially prepared data for encryption, fix and hop
        EncParcel unencrypted(static_cast<uint32_t>(ota >> 32), static_cast<uint32_t>(ota));
        keeloq_encdec<KernelLearningMode::ForceAll | InputsMode, false>(unencrypted, Decryptor::Make(man, seed, true), KeeloqLearning::Matrix::Everything(), res->result.decrypted);
    }

    res->result.setInputTransform(InputsMask);
}

template<InputTransform InputTransform, uint8_t NumInputs, bool SeedOnly, bool ForceAll>
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_bruteforce(KeeloqKernelMultiLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static_assert(is_valid(InputTransform), "Invalid input transform mask");
    static constexpr auto InputsMask = InputTransformToKernelMode<InputTransform>::value;

    CudaContext ctx = CudaContext::Get();
    assert(KernelInputs->decryptors->num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    if constexpr (SeedOnly)
    {
        assert((*KernelInputs->decryptors)[ctx.thread_id].has_seed() && "Seed must be present for seed-only mode, some problem in generator.");

        if constexpr (ForceAll)
        {
            keeloq_decryption_run<NumInputs, KernelLearningMode::ForceSeeded | InputsMask>(ctx, *KernelInputs);
        }
        else
        {
            keeloq_decryption_run<NumInputs, KernelLearningMode::ExplicitSeeded | InputsMask>(ctx, *KernelInputs);
        }
    }
    else
    {
        if constexpr (ForceAll)
        {
            // Dynamic per-thread branch
            if ((*KernelInputs->decryptors)[ctx.thread_id].has_seed())
            {
                keeloq_decryption_run<NumInputs, KernelLearningMode::ForceAll | InputsMask>(ctx, *KernelInputs);
            }
            else
            {
                keeloq_decryption_run<NumInputs, KernelLearningMode::ForceNormal | InputsMask>(ctx, *KernelInputs);
            }
        }
        else
        {
            // Dynamic per-thread branch
            if ((*KernelInputs->decryptors)[ctx.thread_id].has_seed())
            {
                keeloq_decryption_run<NumInputs, KernelLearningMode::ExplicitAll | InputsMask>(ctx, *KernelInputs);
            }
            else
            {
                keeloq_decryption_run<NumInputs, KernelLearningMode::ExplicitNormal | InputsMask>(ctx, *KernelInputs);
            }
        }
    }

    uint8_t num_matches = keeloq_check_matches<NumInputs>(ctx, KernelInputs);
    ret->onKernelFinish(num_matches);
}

template<uint8_t NumInputs, InputTransform InputMut, KeeloqLearning::LearningType LType, Modifier::Algo AMod>
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_single_learning(KeeloqKernelSingleLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{

    CudaContext ctx = CudaContext::Get();
    assert(KernelInputs->decryptors->num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    uint32_t num_full_matches = 0;

    KEELOQ_INNER_LOOP(ctx, decryptor_index, KernelInputs->decryptors->num)
    {
        const Decryptor& decryptor = KernelInputs->GetDecryptor(decryptor_index);

        // Single input
        uint8_t num_matches = keeloq_decrypt_single_learning<First, InputMut, LType, AMod>(
            ctx, decryptor, KernelInputs->template Result<First, NumInputs>(decryptor_index));

        // if more than 1 input
        if constexpr (NumInputs > 1)
        {
            // and first input has match - check next input
            if (num_matches == 1)
            {
                num_matches += keeloq_decrypt_single_learning<Second, InputMut, LType, AMod>(
                    ctx, decryptor, KernelInputs->template Result<Second, NumInputs>(decryptor_index));

                // if more than 2 inputs
                if constexpr (NumInputs > 2)
                {
                    // and second input also has match - check next input
                    if (num_matches == 2)
                    {
                        num_matches += keeloq_decrypt_single_learning<Third, InputMut, LType, AMod>(
                            ctx, decryptor, KernelInputs->template Result<Third, NumInputs>(decryptor_index));
                    }
                }
            }
        }

        if constexpr (NumInputs > 1)
        {
            static constexpr uint8_t MaxCounterDeviation = NumInputs + 1;

            const auto firstCnt = KernelInputs->template Result<First, NumInputs>(decryptor_index).cnt();

            const auto secndCnt = KernelInputs->template Result<Second, NumInputs>(decryptor_index).cnt();

            // We reduce number of matches if counter deviation is bigger than MaxCounterDeviation
            num_matches -= __usad(firstCnt, secndCnt, 0u) > MaxCounterDeviation;

            if constexpr (NumInputs > 2)
            {
                const auto thirdCnt = KernelInputs->template Result<Third, NumInputs>(decryptor_index).cnt();

                // We reduce number of matches if counter deviation is bigger than MaxCounterDeviation
                num_matches -= __usad(firstCnt, thirdCnt, 0u) > MaxCounterDeviation;
            }
        }

        num_full_matches += (num_matches == NumInputs);
    }

    ret->onKernelFinish(num_full_matches);
}

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{

using BruteforceKernelLauncherFunc = void(*)(const CudaConfig&, uint8_t numInputs, bool allLearnings, bool seedOnly, KeeloqKernelMultiLearningInput::TCudaPtr, KernelResult::TCudaPtr);

using SingleKernelLauncherFunc = void(*)(uint64_t, uint64_t, uint32_t, bool, DecryptKernelResult::TCudaPtr);

template<std::uint32_t RawInputModMask>
__host__ void LaunchBruteforceKernel(const CudaConfig& cuda, uint8_t numInputs, bool allLearnings, bool seedOnly, KeeloqKernelMultiLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static constexpr auto Mask = static_cast<InputTransform>(RawInputModMask);

    static constexpr auto ForceSeedOnly = true;
    static constexpr auto NotSeedOnly = false;

    static constexpr auto ForceAllLearnings = true;
    static constexpr auto UseLearningMatrix = false;

    #define CALL_KERNEL(NumInputs, al, so) \
        if (al && so)       { Kernel_keeloq_bruteforce<Mask, NumInputs, ForceSeedOnly,  ForceAllLearnings><<<cuda.blocks, cuda.threads>>> (KernelInputs, ret); } \
        else if (al && !so) { Kernel_keeloq_bruteforce<Mask, NumInputs, NotSeedOnly,    ForceAllLearnings><<<cuda.blocks, cuda.threads>>> (KernelInputs, ret); } \
        else if (!al && so) { Kernel_keeloq_bruteforce<Mask, NumInputs, ForceSeedOnly,  UseLearningMatrix><<<cuda.blocks, cuda.threads>>> (KernelInputs, ret); } \
        else                { Kernel_keeloq_bruteforce<Mask, NumInputs, NotSeedOnly,    UseLearningMatrix><<<cuda.blocks, cuda.threads>>> (KernelInputs, ret); } \

    switch (numInputs)
    {
    case 1:
    {
        CALL_KERNEL(1, allLearnings, seedOnly);
        break;
    }
    case 2:
    {
        CALL_KERNEL(2, allLearnings, seedOnly);
        break;
    }
    case 3:
    {
        CALL_KERNEL(3, allLearnings, seedOnly);
        break;
    }
    default:
    {
        assertf(false, "Invalid number of inputs for templated kernel launch: %d! CUDA launch skipped!\n", numInputs);
        break;
    }
    }
}

template<std::uint32_t RawInputModMask>
__host__ void LaunchSingleTemplatedKernel(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, DecryptKernelResult::TCudaPtr ret)
{
    static constexpr auto Mask = static_cast<InputTransform>(RawInputModMask);
    Kernel_keeloq_single_encdec<Mask> << <1, 1 >> > (ota, man, seed, isDecrypt, ret);
}

template<std::size_t... Is>
__host__ constexpr auto MakeLaunchBruteTable(std::index_sequence<Is...>)
{
    return std::array<BruteforceKernelLauncherFunc, sizeof...(Is)>
    {
        &LaunchBruteforceKernel<Is>...
    };
}

template<std::size_t... Is>
__host__ constexpr auto MakeLaunchSingleTable(std::index_sequence<Is...>)
{
    return std::array<SingleKernelLauncherFunc, sizeof...(Is)>
    {
        &LaunchSingleTemplatedKernel<Is>...
    };
}

} // namespace

namespace flat
{

using BruteforceKernelLauncherFunc = void(*)(const CudaConfig&, uint8_t numInputs, KeeloqKernelSingleLearningInput::TCudaPtr, KernelResult::TCudaPtr);

template<std::uint32_t RawInputModMask, std::uint32_t RawLearningType, std::uint32_t RawAlgoModifier>
__host__ void LaunchFlatBruteforceKernel(const CudaConfig& cuda, uint8_t numInputs, KeeloqKernelSingleLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static constexpr auto LearningType = static_cast<KeeloqLearning::LearningType>(RawLearningType);
    static constexpr auto AlgoModifier = static_cast<Modifier::Algo>(RawAlgoModifier);

    static constexpr bool IsValidCombination = DecryptedResults::getIndex<LearningType, AlgoModifier>() != KeeloqLearning::InvalidResultIndex;

    if constexpr (IsValidCombination)
    {
        static constexpr auto InputsMut = static_cast<InputTransform>(RawInputModMask);

        switch (numInputs)
        {
        case 1:
        {
            Kernel_keeloq_single_learning<1, InputsMut, LearningType, AlgoModifier> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
            break;
        }
        case 2:
        {
            Kernel_keeloq_single_learning<2, InputsMut, LearningType, AlgoModifier> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
            break;
        }
        case 3:
        {
            Kernel_keeloq_single_learning<3, InputsMut, LearningType, AlgoModifier> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
            break;
        }
        default:
        {
            assertf(false, "Invalid number of inputs for templated kernel launch: %d! CUDA launch skipped!\n", numInputs);
            break;
        }
        }
    }
}


template<typename ISeq, typename LSeq, typename MSeq>
struct KernelTableBuilder;

template<std::size_t... Is, std::size_t... Ls, std::size_t... Ms>
struct KernelTableBuilder<std::index_sequence<Is...>, std::index_sequence<Ls...>, std::index_sequence<Ms...>>
{
    static constexpr std::size_t NumI = sizeof...(Is);
    static constexpr std::size_t NumL = sizeof...(Ls);
    static constexpr std::size_t NumM = sizeof...(Ms);
    static constexpr std::size_t Total = NumI * NumL * NumM;

    static constexpr std::size_t IVals[] = { Is... };
    static constexpr std::size_t LVals[] = { Ls... };
    static constexpr std::size_t MVals[] = { Ms... };

    template<std::size_t Flat>
    static constexpr auto GetFunc()
    {
        return &LaunchFlatBruteforceKernel<
            static_cast<std::uint32_t>(IVals[Flat / (NumL * NumM)]),
            static_cast<std::uint32_t>(LVals[(Flat / NumM) % NumL]),
            static_cast<std::uint32_t>(MVals[Flat % NumM])>;
    }

    template<std::size_t... Flat>
    static constexpr auto Build(std::index_sequence<Flat...>)
    {
        return std::array<BruteforceKernelLauncherFunc, Total>{ GetFunc<Flat>()... };
    }
};

template<std::size_t... Is, std::size_t... Ls, std::size_t... Ms>
__host__ constexpr auto MakeKernelsLaunchTable(std::index_sequence<Is...>, std::index_sequence<Ls...>, std::index_sequence<Ms...>)
{
    using Builder = KernelTableBuilder<std::index_sequence<Is...>, std::index_sequence<Ls...>, std::index_sequence<Ms...>>;
    return Builder::Build(std::make_index_sequence<Builder::Total>{});
}

template<typename ISeq, typename LSeq, typename MSeq>
struct KernelTableIndexer;

template<std::size_t... Is, std::size_t... Ls, std::size_t... Ms>
struct KernelTableIndexer<std::index_sequence<Is...>, std::index_sequence<Ls...>, std::index_sequence<Ms...>>
{
    static constexpr std::size_t NumI = sizeof...(Is);
    static constexpr std::size_t NumL = sizeof...(Ls);
    static constexpr std::size_t NumM = sizeof...(Ms);

    static constexpr std::size_t IVals[] = { Is... };
    static constexpr std::size_t LVals[] = { Ls... };
    static constexpr std::size_t MVals[] = { Ms... };

    template<std::size_t N, std::size_t Val, std::size_t... Vals>
    static constexpr std::size_t IndexOf(const std::size_t(&arr)[N])
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            if (arr[i] == Val) return i;
        }
        return N; // not found
    }

    static constexpr std::size_t GetFlatIndex(std::size_t inputsMut, std::size_t learningType, std::size_t algoMod)
    {
        std::size_t iIdx = NumI, lIdx = NumL, mIdx = NumM;
        for (std::size_t i = 0; i < NumI; ++i) { if (IVals[i] == inputsMut) { iIdx = i; break; } }
        for (std::size_t i = 0; i < NumL; ++i) { if (LVals[i] == learningType) { lIdx = i; break; } }
        for (std::size_t i = 0; i < NumM; ++i) { if (MVals[i] == algoMod) { mIdx = i; break; } }
        return iIdx * (NumL * NumM) + lIdx * NumM + mIdx;
    }

    static constexpr std::size_t Total = NumI * NumL * NumM;
};

using KernelIndexer = KernelTableIndexer<
    std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>,
    KeeloqLearning::LearningTypesSequence,
    KeeloqLearning::Modifier::TypeSequence>;

} // namespace flat

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelMultiLearningInput& mainInputs, const CudaConfig& cuda)
{
    static constexpr auto LaunchTable = MakeLaunchBruteTable(std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>{});

    KernelResult kernel_results;

    if (!mainInputs.Ready())
    {
        assert(false && "Kernel inputs are not ready! Check your config and generator.");
        printf("Kernel inputs are not ready! CUDA launch skipped!\n");
        return kernel_results;
    }

    if (!mainInputs.GetLearningMatrix().isValid())
    {
        assert(false && "Invalid learning matrix! No learning type enabled. Check your config and generator.");
        printf("Invalid learning matrix! No learning type enabled. CUDA launch skipped!\n");
        return kernel_results;
    }

    if (mainInputs.InputsCount() > 1 && !mainInputs.InputsFixMatch())
    {
        assert(false && "Fixed parts of inputs do not match! Kernel launch skipped.");
        printf("Fixed parts of inputs do not match! CUDA launch skipped!\n");
        return kernel_results;
    }

    const auto inputTransform = mainInputs.GetInputTransform();
    auto launcherIndex = static_cast<std::size_t>(inputTransform);
    if (launcherIndex >= LaunchTable.size())
    {
        printf("Invalid input transform for templated kernel launch: %d! CUDA launch skipped!\n", static_cast<uint32_t>(inputTransform));
        assert(false && "Invalid input transform for templated kernel launch!");
        return kernel_results;
    }

    const bool allLearnings = mainInputs.GetLearningMatrix().isAllEnabled();
    const bool seedOnly = mainInputs.GetConfig().type == BruteforceType::Seed;

    LaunchTable[launcherIndex](cuda, mainInputs.InputsCount(), allLearnings, seedOnly, mainInputs.ptr(), kernel_results.ptr());

    kernel_results.read();
    mainInputs.read();

    return kernel_results;
}

__host__ KernelResult keeloq::kernels::cuda_brute(KeeloqKernelSingleLearningInput& flatInputs, const CudaConfig& cuda)
{
    static constexpr auto LaunchTable = flat::MakeKernelsLaunchTable(
        std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>{},
        KeeloqLearning::LearningTypesSequence{},
        KeeloqLearning::Modifier::TypeSequence{});

    KernelResult kernel_results;

    if (!flatInputs.Ready())
    {
        assert(false && "Kernel inputs are not ready! Did you forget to call prepare? Check your config and generator.");
        printf("Kernel inputs are not ready! CUDA launch skipped!\n");
        return kernel_results;
    }

    if (flatInputs.InputsCount() > 1 && !flatInputs.InputsFixMatch())
    {
        assert(false && "Fixed parts of inputs do not match! Kernel launch skipped.");
        printf("Fixed parts of inputs do not match! CUDA launch skipped!\n");
        return kernel_results;
    }

    const auto launcherIndex = flat::KernelIndexer::GetFlatIndex(
        static_cast<std::size_t>(flatInputs.inputTransform),
        static_cast<std::size_t>(flatInputs.learning),
        static_cast<std::size_t>(flatInputs.algorithModifier));

    LaunchTable[launcherIndex](cuda, flatInputs.inputsCount, flatInputs.ptr(), kernel_results.ptr());

    kernel_results.read();
    flatInputs.read();

    return kernel_results;
}

__host__ ThreadResult::Multi keeloq::kernels::cuda_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, InputTransform inputTransform)
{
    static constexpr auto LaunchTable = MakeLaunchSingleTable(std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>{});

    DecryptKernelResult kernel_results;

    const auto launcherIndex = static_cast<std::size_t>(inputTransform);
    if (launcherIndex >= LaunchTable.size())
    {
        printf("Invalid input transform for templated single enc/dec launch: %d! CUDA launch skipped!\n", static_cast<uint32_t>(inputTransform));
        assert(false && "Invalid input transform for templated single enc/dec launch!");
        return kernel_results.result;
    }

    LaunchTable[launcherIndex](ota, man, seed, isDecrypt, kernel_results.ptr());

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
