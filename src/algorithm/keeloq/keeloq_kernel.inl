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
__device__ uint8_t is_cnt_match(const Span<ThreadResult::Multi>& results)
{
    static_assert(NumInputs > 1, "This function is not supposed to be called in single input mode");

    const uint8_t counter_maxdiff = NumInputs + 1;

    const uint32_t expected_cnt = results[0].matchedCounter();
    uint32_t lrn_matches = 1;

    UNROLL
    for (uint8_t item = 1; item < NumInputs; ++item)
    {
        const uint32_t cnt = results[item].matchedCounter();
        // No underflow in CUDA
        lrn_matches += __usad(expected_cnt, cnt, 0u) < counter_maxdiff;
    }

    return lrn_matches == NumInputs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
template<InputsTransform Mask>
struct InputTransformToKernelMode
{
    static constexpr KernelLearningMode value =
        (has_flag(Mask, InputsTransform::RevKey) ? KernelLearningMode::RevKey : KernelLearningMode::NoInputTransform) |
        (has_flag(Mask, InputsTransform::XorFix) ? KernelLearningMode::XorFix : KernelLearningMode::NoInputTransform) |
        (has_flag(Mask, InputsTransform::XorHop) ? KernelLearningMode::XorHop : KernelLearningMode::NoInputTransform) |
        (has_flag(Mask, InputsTransform::XorDec) ? KernelLearningMode::XorDec : KernelLearningMode::NoInputTransform);
};

/**
 *  Helper templates to convert between KernelLearningMode bitmask and InputTransform bitmask.
 * This converts from Kernel's data to Host's data
 */
template<KernelLearningMode Mode>
struct KernelModeToInputTransform
{
    static constexpr InputsTransform value = static_cast<InputsTransform>(
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::RevKey)) ? static_cast<uint8_t>(InputsTransform::RevKey) : 0) |
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::XorFix)) ? static_cast<uint8_t>(InputsTransform::XorFix) : 0) |
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::XorHop)) ? static_cast<uint8_t>(InputsTransform::XorHop) : 0) |
        ((static_cast<int>(Mode) & static_cast<int>(KernelLearningMode::XorDec)) ? static_cast<uint8_t>(InputsTransform::XorDec) : 0));
};

/**
 *  Single decryption call for specific learning type and algorithm type.
 * Returns decryption result.
 * Uses template parameters to optimize code and remove branches.
 * So for each learning type and algorithm type will be generated separate function.
 *
 *  call-stack:
 *   -> keeloq_encdec_single
 *      keeloq_encdec_multi
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, InputsTransform InputsMut, LearningType type, AlgoType AType>
__device__ __forceinline__ uint32_t keeloq_encdec_single(uint32_t hop, uint32_t fixed, const Decryptor& decryptor)
{
    static constexpr bool IsEncrypt = !IsDecrypt;

    // In decrypt mode with XorHop transform - input is hop, and has to be xored
    // In encrypt mode with XorDec transform - `hop` is actual unencrypted value, and has to be xored before encryption
    static constexpr bool NeedPreXor = (IsDecrypt && !!(InputsMut & InputsTransform::XorHop)) || (IsEncrypt && !!(InputsMut & InputsTransform::XorDec));

    // In encrypt mode with XorHop transform - result is hop, and has to be xored
    // In decrypt mode with XorDec transform - we need to XOR decrypted resul
    static constexpr bool NeedPostXor = (IsEncrypt && !!(InputsMut & InputsTransform::XorHop)) || (IsDecrypt && !!(InputsMut & InputsTransform::XorDec));

    const uint64_t key = decryptor.template getKey<!!(InputsMut & InputsTransform::RevKey)>();
    const uint32_t fix = decryptor.template getXored<!!(InputsMut & InputsTransform::XorFix)>(fixed);
    const uint32_t data = decryptor.template getXored<NeedPreXor>(hop);
    const uint32_t seed = decryptor.seed();

    static_assert(AType == AlgoType::Normal || AType == AlgoType::Inverted, "Unsupported/Unknown Keeloq algorithm type");

    static_assert(
        type == LearningType::Simple    || type == LearningType::Normal || type == LearningType::Secure ||
        type == LearningType::Xor       || type == LearningType::Faac   || type == LearningType::Serial1 ||
        type == LearningType::Serial2   || type == LearningType::Serial3, "Unsupported KeeloqLearningType");

    static_assert(DecryptedResults::getIndex<type, AType>() != KeeloqLearning::InvalidResultIndex, "Unsupported KeeloqLearningType/AlgoType combination");

    uint32_t result = 0;

    // Inverted logic for learning types that support it (Normal, Secure, Faac)
    if constexpr (AType == AlgoType::Inverted)
    {
        if constexpr (type == LearningType::Normal)
        {
            const uint64_t n_key = keeloq::learning::normal<false>(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Secure)
        {
            const uint64_t n_key = keeloq::learning::secure<false>(fix, seed, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Faac)
        {
            // Faac uses encrypt by default, Inverted logic here will be decrypt
            const uint64_t n_key = keeloq::learning::faac<true>(seed, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
    }
    else if (AType == AlgoType::Normal)
    {
        if constexpr (type == LearningType::Simple)
        {
            result = keeloq::common::encdec<IsDecrypt>(data, key);
        }
        else if constexpr (type == LearningType::Normal)
        {
            const uint64_t n_key = keeloq::learning::normal(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Secure)
        {
            const uint64_t n_key = keeloq::learning::secure(fix, seed, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Xor)
        {
            const uint64_t n_key = keeloq::learning::magic_xor_type1(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Faac)
        {
            const uint64_t n_key = keeloq::learning::faac(seed, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial1)
        {
            const uint64_t n_key = keeloq::learning::serial_type1(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial2)
        {
            const uint64_t n_key = keeloq::learning::serial_type2(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
        else if constexpr (type == LearningType::Serial3)
        {
            const uint64_t n_key = keeloq::learning::serial_type3(fix, key);
            result = keeloq::common::encdec<IsDecrypt>(data, n_key);
        }
    }

    return decryptor.template getXored<NeedPostXor>(result);
}

/**
 *  Single decryption call wrapper for specific learning type and algorithm type.
 * Result will be written to results array according to index from KeeloqLearning::DecryptedResults::getIndex() method.
 *
 *  call-stack:
 *   -> keeloq_encdec_single
 *      keeloq_encdec_multi
 *      keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IsDecrypt, InputsTransform InputsMut, LearningType type, AlgoType AType>
__device__ __forceinline__ void keeloq_encdec_single(uint32_t data, uint32_t fix, const Decryptor& decryptor, DecryptedResults& results)
{
    static constexpr auto index = DecryptedResults::getIndex<type, AType>();
    if constexpr (index != KeeloqLearning::InvalidResultIndex)
    {
        results[index] = keeloq_encdec_single<IsDecrypt, InputsMut, type, AType>(data, fix, decryptor);
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
template<bool IsDecrypt, InputsTransform InputsMut, AlgoType AType, LearningType... LTypes>
__device__ __forceinline__ void keeloq_encdec_multi(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    DecryptedResults& results, helpers::ValuesSet<LearningType, LTypes...>)
{
    ((keeloq_encdec_single<IsDecrypt, InputsMut, LTypes, AType>(data, fix, decryptor, results)), ...);
}

/**
 *  Conditional param pack expansion call for single calls.
 * Uses if block, but in case of small number of learning types and algorithm types should be better than loop with if block inside.
 *
 *  call-stack:
 *      keeloq_encdec_single
 *   -> keeloq_encdec_multi_cond
 *      keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IsDecrypt, InputsTransform InputsMut, AlgoType AType, LearningType... LTypes>
__device__ __forceinline__ void keeloq_encdec_multi_cond(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, DecryptedResults& results, helpers::ValuesSet<LearningType, LTypes...>)
{
    ((learnings_matrix.template isEnabled<LTypes, AType, true>() ? keeloq_encdec_single<IsDecrypt, InputsMut, LTypes, AType>(data, fix, decryptor, results) : void()), ...);
}

/**
 *  This function call unconditional or conditional multi call for seed learning types with specific algorithm type.
 * In fact this is just magic wrapper for `keeloq_encdec_single` for Secure and Faac learning types
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, InputsTransform InputsMut, AlgoType AType>
__device__ __forceinline__ void keeloq_encdec_seed_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AType>(data, fix, decryptor, decrypted.data, SeededTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AType>(data, fix, decryptor, learnings_matrix, decrypted.data, SeededTypes{});
    }
}

/**
 *  This function call unconditional or conditional multi call for all (seeded and normal) learning types with specific algorithm type.
 * In fact this is just magic wrapper for `keeloq_encdec_single`
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal,all]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, InputsTransform InputsMut, AlgoType AType>
__device__ __forceinline__ void keeloq_encdec_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AType>(data, fix, decryptor, decrypted.data, EveryLearningType{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AType>(data, fix, decryptor, learnings_matrix, decrypted.data, EveryLearningType{});
    }
}

/**
 *  This function call unconditional or conditional multi call for non-seed (normal) learning types with specific algorithm type.
 * In fact this is just magic wrapper for `keeloq_encdec_single` for Simple, Normal, Xor and Serial learning types
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *   -> keeloq_decrypt_[seed,normal]
 *      keeloq_encdec
 */
template<bool IgnoreMatrix, bool IsDecrypt, InputsTransform InputsMut, AlgoType AType>
__device__ __forceinline__ void keeloq_encdec_normal_all(uint32_t data, uint32_t fix, const Decryptor& decryptor,
    const Matrix& learnings_matrix, ThreadResult::LearningsArray& decrypted)
{
    if constexpr (IgnoreMatrix)
    {
        keeloq_encdec_multi<IsDecrypt, InputsMut, AType>(data, fix, decryptor, decrypted.data, NormalTypes{});
    }
    else
    {
        keeloq_encdec_multi_cond<IsDecrypt, InputsMut, AType>(data, fix, decryptor, learnings_matrix, decrypted.data, NormalTypes{});
    }
}

/**
 *  This function calls seeded and/or normal wrapper function for all algorithm types according to template parameters and learning matrix.
 *
 *  call-stack:
 *      keeloq_encdec_single
 *      keeloq_encdec_multi_cond
 *      keeloq_decrypt_[seed,normal]
 *   -> keeloq_encdec
 */
template<KernelLearningMode Mode, bool IsDecrypt = true>
__device__ __forceinline__ void keeloq_encdec(const EncParcel& enc, const Decryptor& decryptor, const Matrix& learnings_matrix, ThreadResult::LearningsArray& results)
{
    static constexpr bool IgnoreMatrix     = !!(Mode & KernelLearningMode::Force);
    static constexpr bool ExplicitDecrypt  = !!(Mode & KernelLearningMode::Explicit);

    static_assert(IgnoreMatrix != ExplicitDecrypt, "Can't be only Force or Explicit");

    // Algorithm-type flags that disable specific calculations. Only used if not Force mode.
    constexpr KernelLearningMode disabledAlgoTypes = (Mode & (IgnoreMatrix ? static_cast<KernelLearningMode>(0) : KernelLearningMode::AlgoMask));
    constexpr InputsTransform InputsMut = KernelModeToInputTransform<Mode>::value;

    // Normal learning types (NO SEED)
    if constexpr (!!(Mode & KernelLearningMode::Normal))
    {
        keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, InputsMut, AlgoType::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

        // If inverted algo logic allowed
        if constexpr (!(disabledAlgoTypes & KernelLearningMode::NoInv))
        {
            keeloq_encdec_normal_all<IgnoreMatrix, IsDecrypt, InputsMut, AlgoType::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
        }
    }

    // Seeded learning types
    if constexpr (!!(Mode & KernelLearningMode::Seeded))
    {
        keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, InputsMut, AlgoType::Normal>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);

        // If inverted algo logic allowed
        if constexpr (!(disabledAlgoTypes & KernelLearningMode::NoInv))
        {
            keeloq_encdec_seed_all<IgnoreMatrix, IsDecrypt, InputsMut, AlgoType::Inverted>(enc.hop(), enc.fix(), decryptor, learnings_matrix, results);
        }
    }
}

/**
 *  This function decrypts input specified in template argument @InputIndex
 * And writes `result.match` if fixed part matches decrypted hopping.
 * This is used as first filter before robust check over all inputs.
 * There could be more than one match, in this method we do not care,
 * we want to have indication that this decryptor has a possibility to be the correct one.
 */
template<KernelLearningMode LearningMode, uint8_t InputIndex>
__device__ __forceinline__ void keeloq_decrypt_and_quick_analyze(const CudaContext& ctx, const Decryptor& decryptor, const Matrix& learning_matrix, Span<ThreadResult::Multi>& results)
{
    const EncParcel& enc = InputsCache[InputIndex];

    ThreadResult::Multi& result = results[InputIndex];

    result.decryptor = decryptor;
    result.setInputIndex(InputIndex);
    result.setInputTransform(KernelModeToInputTransform<LearningMode>::value);

    keeloq_encdec<LearningMode>(enc, decryptor, learning_matrix, result.decrypted);

    result.match = analyze_single_result<LearningMode>(result, enc.srl(), enc.btn(), learning_matrix);
}

template<uint8_t InputIndex, InputsTransform InputMut, LearningType LType, AlgoType AType>
__device__ __forceinline__ bool keeloq_decrypt_single_learning(const CudaContext& ctx, const Decryptor& decryptor, ThreadResult::Single& result)
{
    const EncParcel& enc = InputsCache[InputIndex];

    result.decryptor = decryptor;
    result.setInputIndex(InputIndex);
    result.setInputTransform(InputMut);

    static constexpr bool IsDecrypt = true;
    result.decrypted = keeloq_encdec_single<IsDecrypt, InputMut, LType, AType>(enc.hop(), enc.fix(), decryptor);

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
template<KernelLearningMode LearningMode>
__device__ __forceinline__ bool keeloq_decryption_run(const CudaContext& ctx, KeeloqKernelMultiLearningInput& input)
{
    static constexpr uint8_t NumInputs = 3;
    static_assert(First < Second && Second < Third, "Static assert just to get rid of warning");

    const auto& decryptors = *input.decryptors;
    const auto& learning_matrix = input.GetLearningMatrix();

    auto& all_results = *input.results;

    // outer loop for each thread's decryptor
    const auto decryptor_index = ctx.thread_id;

    Span<ThreadResult::Multi> results(&all_results[decryptor_index * NumInputs], NumInputs);

    const Decryptor& decryptor = decryptors[decryptor_index];

    // Single input
    keeloq_decrypt_and_quick_analyze<LearningMode, First>(ctx, decryptor, learning_matrix, results);

    if (results[First].match != KeeloqLearning::NoMatch)
    {
        keeloq_decrypt_and_quick_analyze<LearningMode, Second>(ctx, decryptor, learning_matrix, results);

        if (results[Second].match == results[First].match)
        {
            keeloq_decrypt_and_quick_analyze<LearningMode, Third>(ctx, decryptor, learning_matrix, results);

            if (results[Third].match == results[First].match)
            {
                // at this moment all 3 inputs have the same serial and button
                // now we check the counter deviation

                return is_cnt_match<NumInputs>(results);
            }
        }
    }

    // Always 3 inputs
    results[First].match = KeeloqLearning::NoMatch;
    results[Second].match = KeeloqLearning::NoMatch;
    results[Third].match = KeeloqLearning::NoMatch;

    return false;
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

template<InputsTransform InputsMask>
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

template<InputsTransform InputsTransform, bool SeedOnly, bool ForceAll>
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_bruteforce(KeeloqKernelMultiLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static_assert(is_valid(InputsTransform), "Invalid input transform mask");
    static constexpr auto InputsMask = InputTransformToKernelMode<InputsTransform>::value;

    CudaContext ctx = CudaContext::Get();
    assert(KernelInputs->decryptors->num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    bool hasMatch = false;

    if constexpr (SeedOnly)
    {
        assert((*KernelInputs->decryptors)[ctx.thread_id].has_seed() && "Seed must be present for seed-only mode, some problem in generator.");

        if constexpr (ForceAll)
        {
            hasMatch = keeloq_decryption_run<KernelLearningMode::ForceSeeded | InputsMask>(ctx, *KernelInputs);
        }
        else
        {
            hasMatch = keeloq_decryption_run<KernelLearningMode::ExplicitSeeded | InputsMask>(ctx, *KernelInputs);
        }
    }
    else
    {
        const bool hasSeed = (*KernelInputs->decryptors)[ctx.thread_id].has_seed();

        if constexpr (ForceAll)
        {
            // Dynamic per-thread branch
            if (hasSeed)
            {
                hasMatch = keeloq_decryption_run<KernelLearningMode::ForceAll | InputsMask>(ctx, *KernelInputs);
            }
            else
            {
                hasMatch = keeloq_decryption_run<KernelLearningMode::ForceNormal | InputsMask>(ctx, *KernelInputs);
            }
        }
        else
        {
            // Dynamic per-thread branch
            if (hasSeed)
            {
                hasMatch = keeloq_decryption_run<KernelLearningMode::ExplicitAll | InputsMask>(ctx, *KernelInputs);
            }
            else
            {
                hasMatch = keeloq_decryption_run<KernelLearningMode::ExplicitNormal | InputsMask>(ctx, *KernelInputs);
            }
        }
    }

    ret->onKernelFinish(hasMatch);
}

template<InputsTransform InputMut, KeeloqLearning::LearningType LType, AlgoType AType>
__global__ KERNEL_LAUNCH_BOUNDS void Kernel_keeloq_single_learning(KeeloqKernelSingleLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    // Now only 3 inputs supports in kernels
    static constexpr auto NumInputs = 3;

    CudaContext ctx = CudaContext::Get();
    assert(KernelInputs->decryptors->num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    uint32_t num_full_matches = 0;

    KEELOQ_INNER_LOOP(ctx, decryptor_index, KernelInputs->decryptors->num)
    {
        const Decryptor& decryptor = KernelInputs->GetDecryptor(decryptor_index);

        // Single input
        uint8_t num_matches = keeloq_decrypt_single_learning<First, InputMut, LType, AType>(ctx, decryptor, KernelInputs->template Result<First, NumInputs>(decryptor_index));

        static constexpr uint8_t MaxCounterDeviation = NumInputs + 1;

        // and first input has match - check next input
        if (num_matches == 1)
        {
            num_matches += keeloq_decrypt_single_learning<Second, InputMut, LType, AType>(
                ctx, decryptor, KernelInputs->template Result<Second, NumInputs>(decryptor_index));

            uint16_t counter0 = KernelInputs->template Result<First, NumInputs>(decryptor_index).cnt();
            uint16_t counter1 = KernelInputs->template Result<Second, NumInputs>(decryptor_index).cnt();

            // We reduce number of matches if counter deviation is bigger than MaxCounterDeviation
            num_matches -= __usad(counter0, counter1, 0u) > MaxCounterDeviation;

            // and second input also has match - check next input
            if (num_matches == 2)
            {
                num_matches += keeloq_decrypt_single_learning<Third, InputMut, LType, AType>(
                    ctx, decryptor, KernelInputs->template Result<Third, NumInputs>(decryptor_index));

                uint16_t counter2 = KernelInputs->template Result<Third, NumInputs>(decryptor_index).cnt();

                // We reduce number of matches if counter deviation is bigger than MaxCounterDeviation
                num_matches -= __usad(counter0, counter2, 0u) > MaxCounterDeviation;
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

using BruteforceKernelLauncherFunc = void(*)(const CudaConfig&, bool allLearnings, bool seedOnly, KeeloqKernelMultiLearningInput::TCudaPtr, KernelResult::TCudaPtr);

using SingleKernelLauncherFunc = void(*)(uint64_t, uint64_t, uint32_t, bool, DecryptKernelResult::TCudaPtr);

template<std::uint32_t RawInputTransform>
__host__ void LaunchBruteforceKernel(const CudaConfig& cuda, bool allLearnings, bool seedOnly, KeeloqKernelMultiLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static constexpr auto Mask = static_cast<InputsTransform>(RawInputTransform);

    static constexpr auto ForceSeedOnly = true;
    static constexpr auto NotSeedOnly = false;

    static constexpr auto ForceAllLearnings = true;
    static constexpr auto UseLearningMatrix = false;

    if (allLearnings && seedOnly)
    {
        Kernel_keeloq_bruteforce<Mask, ForceSeedOnly, ForceAllLearnings> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
    }
    else if (allLearnings && !seedOnly)
    {
        Kernel_keeloq_bruteforce<Mask, NotSeedOnly, ForceAllLearnings> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
    }
    else if (!allLearnings && seedOnly)
    {
        Kernel_keeloq_bruteforce<Mask, ForceSeedOnly, UseLearningMatrix> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
    }
    else
    {
        Kernel_keeloq_bruteforce<Mask, NotSeedOnly, UseLearningMatrix> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
    }
}

template<std::uint32_t RawInputTransform>
__host__ void LaunchSingleTemplatedKernel(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, DecryptKernelResult::TCudaPtr ret)
{
    static constexpr auto Mask = static_cast<InputsTransform>(RawInputTransform);
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

namespace SingleMode
{

using BruteforceKernelLauncherFunc = void(*)(const CudaConfig&, KeeloqKernelSingleLearningInput::TCudaPtr, KernelResult::TCudaPtr);

template<std::uint32_t RawInputTransform, std::uint32_t RawLearningType, std::uint32_t RawAlgoType>
__host__ void LaunchFlatBruteforceKernel(const CudaConfig& cuda, KeeloqKernelSingleLearningInput::TCudaPtr KernelInputs, KernelResult::TCudaPtr ret)
{
    static constexpr auto LearningType = static_cast<KeeloqLearning::LearningType>(RawLearningType);
    static constexpr auto AlgoTypeValue = static_cast<AlgoType>(RawAlgoType);

    static constexpr bool IsValidCombination = DecryptedResults::getIndex<LearningType, AlgoTypeValue>() != KeeloqLearning::InvalidResultIndex;

    if constexpr (IsValidCombination)
    {
        static constexpr auto InputsMut = static_cast<InputsTransform>(RawInputTransform);

        Kernel_keeloq_single_learning<InputsMut, LearningType, AlgoTypeValue> << <cuda.blocks, cuda.threads >> > (KernelInputs, ret);
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

    static constexpr std::size_t GetFlatIndex(std::size_t inputsMut, std::size_t learningType, std::size_t algoType)
    {
        std::size_t iIdx = NumI, lIdx = NumL, mIdx = NumM;
        for (std::size_t i = 0; i < NumI; ++i) { if (IVals[i] == inputsMut) { iIdx = i; break; } }
        for (std::size_t i = 0; i < NumL; ++i) { if (LVals[i] == learningType) { lIdx = i; break; } }
        for (std::size_t i = 0; i < NumM; ++i) { if (MVals[i] == algoType) { mIdx = i; break; } }
        return iIdx * (NumL * NumM) + lIdx * NumM + mIdx;
    }

    static constexpr std::size_t Total = NumI * NumL * NumM;
};

using KernelIndexer = KernelTableIndexer<
    std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>,
    KeeloqLearning::LearningTypesSequence,
    KeeloqLearning::AlgoTypeSequence>;

/** Launch table for CUDA kernels */
static constexpr auto LaunchTable = MakeKernelsLaunchTable(
    std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>{},
    KeeloqLearning::LearningTypesSequence{},
    KeeloqLearning::AlgoTypeSequence{});

} // namespace SingleMode

__host__ KernelResult keeloq::kernels::internal::cuda_brute(KeeloqKernelMultiLearningInput& mainInputs, const CudaConfig& cuda)
{
    static constexpr auto LaunchTable = MakeLaunchBruteTable(std::make_index_sequence<static_cast<std::size_t>(InputTransformVariantsCount)>{});

    KernelResult kernel_results;

    if (!mainInputs.GetLearningMatrix().isValid())
    {
        assert(false && "Invalid learning matrix! No learning type enabled. Check your config and generator.");
        printf("Invalid learning matrix! No learning type enabled. CUDA launch skipped!\n");
        return kernel_results;
    }

    const auto inputsTransform = mainInputs.GetInputsTransform();
    auto launcherIndex = static_cast<std::size_t>(inputsTransform);
    if (launcherIndex >= LaunchTable.size())
    {
        printf("Invalid input transform for templated kernel launch: %d! CUDA launch skipped!\n", static_cast<uint32_t>(inputsTransform));
        assert(false && "Invalid input transform for templated kernel launch!");
        return kernel_results;
    }

    const bool allLearnings = mainInputs.GetLearningMatrix().isAllEnabled();
    const bool seedOnly = mainInputs.GetConfig().type == BruteforceType::Seed;

    LaunchTable[launcherIndex](cuda, allLearnings, seedOnly, mainInputs.ptr(), kernel_results.ptr());

    kernel_results.read();
    mainInputs.read();

    return kernel_results;
}

__host__ KernelResult keeloq::kernels::internal::cuda_brute(KeeloqKernelSingleLearningInput& flatInputs, const CudaConfig& cuda)
{
    KernelResult kernel_results;

    const auto launcherIndex = SingleMode::KernelIndexer::GetFlatIndex(
        static_cast<std::size_t>(flatInputs.inputsTransform),
        static_cast<std::size_t>(flatInputs.learning),
        static_cast<std::size_t>(flatInputs.algoType));

    SingleMode::LaunchTable[launcherIndex](cuda, flatInputs.ptr(), kernel_results.ptr());

    kernel_results.read();
    flatInputs.read();

    return kernel_results;
}

__host__ ThreadResult::Multi keeloq::kernels::cuda_encdec(uint64_t ota, uint64_t man, uint32_t seed, bool isDecrypt, InputsTransform inputTransform)
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
