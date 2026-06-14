#include "algorithm/keeloq/keeloq_encryptor.h"

#include "algorithm/keeloq/keeloq_kernel.h"


EncParcel Encryptor::click(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType)
{
    const auto cpu_encrypted = cpuEncrypt(inTransform, ltype, algoType);
    assert(gpuEncrypt(inTransform, ltype, algoType) == cpu_encrypted && "GPU and CPU encryption results do not match");

    const uint64_t detpyrcne = ((uint64_t)fixed(ltype) << 32) | cpu_encrypted;
    const auto ota = misc::rev_bits(detpyrcne);

    const auto cpu_decrypted = cpuDecrypt(ota, inTransform, ltype, algoType);

    assert(cpu_decrypted == gpuDecrypt(ota, inTransform, ltype, algoType) && "GPU and CPU decryption results do not match");
    assert(cpu_decrypted == unencrypted(ltype) && "Decryption failed, decrypted result doesn't match initial unencrypted data");

    if (cpu_decrypted != unencrypted(ltype))
    {
        printf("Generation failed, decrypted result doesn't match initial unencrypted data. Expected: 0x%08X, got: 0x%08X\n",
            unencrypted(ltype), cpu_decrypted);
        return EncParcel();
    }

    count++;
    return ota;
}

uint32_t Encryptor::fixed(KeeloqLearning::LearningType ltype) const
{
    if (ltype == KeeloqLearning::LearningType::Faac)
    {
        // FAAC SLH fixed code: 28-bit serial in the high bits, 4-bit button in the low nibble.
        return ((serial & 0x0FFFFFFFu) << 4) | (button & 0xFu);
    }

    // Standard KeeLoq fixed code: 4-bit button in the high nibble, 28-bit serial below.
    return (uint32_t)button << 28 | (serial & 0x0FFFFFFF);
}

uint32_t Encryptor::unencrypted(KeeloqLearning::LearningType ltype) const
{
    if (ltype != KeeloqLearning::LearningType::Faac)
    {
        // Standard KeeLoq hopping-code plaintext: 4-bit button | 10-bit serial | 16-bit counter.
        return (uint32_t)button << 28 | ((serial & 0x3FF) << 16) | (count & 0x0000FFFFu);
    }

    // FAAC SLH hopping-code plaintext: the low 20 bits hold the counter, the top 12 bits hold three
    // fixed-code nibbles selected by counter parity (matches Flipper's faac_slh gen_data).
    const uint32_t fix = fixed(ltype);

    // Nibbles of the fixed code, most-significant first:
    // nibble[0] = bits 28..31, ..., nibble[7] = bits 0..3 (the button).
    uint8_t nibble[8];
    for (uint32_t i = 0; i < 8; ++i)
    {
        nibble[i] = (fix >> (28u - i * 4u)) & 0xFu;
    }

    // Top 12 bits are three fixed-code nibbles chosen by counter parity; low 20 bits are the counter.
    uint32_t top12;
    if ((count & 1u) == 0u)
    {
        top12 = (uint32_t(nibble[6]) << 8) | (uint32_t(nibble[7]) << 4) | nibble[5];
    }
    else
    {
        top12 = (uint32_t(nibble[2]) << 8) | (uint32_t(nibble[3]) << 4) | nibble[4];
    }

    return (top12 << 20) | (count & 0xFFFFFu);
}

uint64_t Encryptor::man(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    const uint64_t use_key = has_flag(inTransform, InputsTransform::RevKey) ? misc::rev_bytes(key) : key;
    const uint32_t use_fixed = has_flag(inTransform, InputsTransform::XorFix) ? (fixed(ltype) ^ seed) : fixed(ltype);

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
    const auto plaintext = unencrypted(ltype);
    const auto use_unecrypted = has_flag(inTransform, InputsTransform::XorDec) ? (plaintext ^ seed) : plaintext;

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
    const auto result = keeloq::kernels::cuda_enc(((uint64_t)fixed(ltype) << 32) | unencrypted(ltype), key, seed, ltype, algoType, inTransform);

    // This is not bit-reversed result
    return result.decrypted;
}

uint32_t Encryptor::gpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const
{
    const auto result = keeloq::kernels::cuda_dec(enc, key, seed, ltype, algoType, inTransform);
    return result.decrypted;
}
