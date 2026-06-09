#include "algorithm/keeloq/keeloq_encryptor.h"

#include "algorithm/keeloq/keeloq_kernel.h"


EncParcel Encryptor::click(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType)
{
    const auto cpu_encrypted = cpuEncrypt(inTransform, ltype, algoType);
    assert(gpuEncrypt(inTransform, ltype, algoType) == cpu_encrypted && "GPU and CPU encryption results do not match");

    const uint64_t detpyrcne = ((uint64_t)fixed() << 32) | cpu_encrypted;
    const auto ota = misc::rev_bits(detpyrcne);

    const auto cpu_decrypted = cpuDecrypt(ota, inTransform, ltype, algoType);

    assert(cpu_decrypted == gpuDecrypt(ota, inTransform, ltype, algoType) && "GPU and CPU decryption results do not match");
    assert(cpu_decrypted == unencrypted() && "Decryption failed, decrypted result doesn't match initial unencrypted data");

    if (cpu_decrypted != unencrypted())
    {
        printf("Generation failed, decrypted result doesn't match initial unencrypted data. Expected: 0x%08X, got: 0x%08X\n",
            unencrypted(), cpu_decrypted);
        return EncParcel();
    }

    count++;
    return ota;
}

uint64_t Encryptor::man(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    const uint64_t use_key = has_flag(inTransform, InputsTransform::RevKey) ? misc::rev_bytes(key) : key;
    const uint32_t use_fixed = has_flag(inTransform, InputsTransform::XorFix) ? (fixed() ^ seed) : fixed();

    const bool useInv = (algoType == KeeloqLearning::AlgoType::Inverted);

    switch (ltype)
    {
    case KeeloqLearning::LearningType::Simple:
    {
       return use_key;
    }
    case KeeloqLearning::LearningType::Normal:
    {
        return useInv ? keeloq::learning::normal<false>(use_fixed, use_key) : keeloq::learning::normal<true>(use_fixed, use_key);
    }
    case KeeloqLearning::LearningType::Secure:
    {
        return useInv ? keeloq::learning::secure<false>(use_fixed, seed, use_key) : keeloq::learning::secure<true>(use_fixed, seed, use_key);
    }
    case KeeloqLearning::LearningType::Xor:
    {
        return keeloq::learning::magic_xor_type1(use_fixed, use_key);
    }
    case KeeloqLearning::LearningType::Faac:
    {
        // Faac uses encrypt by default (UseDecrypt=false); InvertedDec flips that to decrypt.
        return useInv ? keeloq::learning::faac<true>(seed, use_key) : keeloq::learning::faac<false>(seed, use_key);
    }
    case KeeloqLearning::LearningType::Serial1:
    {
        return keeloq::learning::serial_type1(use_fixed, use_key);
    }
    case KeeloqLearning::LearningType::Serial2:
    {
        return keeloq::learning::serial_type2(use_fixed, use_key);
    }
    case KeeloqLearning::LearningType::Serial3:
    {
        return keeloq::learning::serial_type3(use_fixed, use_key);
    }
    default:
        assert(false && "Unknown learning type");
        return 0;
    }
}

uint32_t Encryptor::cpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    // Pre-Xor unencrypted
    const auto use_unecrypted = has_flag(inTransform, InputsTransform::XorDec) ? (unencrypted() ^ seed) : unencrypted();

    const auto hop = keeloq::common::encrypt(use_unecrypted, man(inTransform, ltype, algoType));

    // Post-Xor hop
    const auto use_hop = has_flag(inTransform, InputsTransform::XorHop) ? (hop ^ seed) : hop;

    // This is not bit-reversed result
    return use_hop;
}

uint32_t Encryptor::cpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    auto reversed_enc = misc::rev_bits(enc);
    auto hopping = (uint32_t)reversed_enc;

    // Pre-Xor hop
    const auto use_hop = has_flag(inTransform, InputsTransform::XorHop) ? (hopping ^ seed) : hopping;

    const auto cpu_decrypted = keeloq::common::decrypt(use_hop, man(inTransform, ltype, algoType));

    // Post-Xor unencrypted
    const auto use_decrypted = has_flag(inTransform, InputsTransform::XorDec) ? (cpu_decrypted ^ seed) : cpu_decrypted;

    return use_decrypted;
}

uint32_t Encryptor::gpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, algoType);
    const auto results = keeloq::kernels::cuda_enc(((uint64_t)fixed() << 32) | unencrypted(), key, seed, inTransform);
    const auto gpu_encrypted = results.decrypted[resIndex];

    // This is not bit-reversed result
    return gpu_encrypted;
}

uint32_t Encryptor::gpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, algoType);

    ThreadResult::Multi result = keeloq::kernels::cuda_dec(enc, key, seed, inTransform);
    return result.decrypted.data[resIndex];
}
