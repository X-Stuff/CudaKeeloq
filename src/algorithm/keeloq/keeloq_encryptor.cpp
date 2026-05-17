#include "algorithm/keeloq/keeloq_encryptor.h"

#include "algorithm/keeloq/keeloq_kernel.h"


EncParcel Encryptor::click(InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod)
{
    const auto cpu_encrypted = cpuEncrypt(inputTransform, ltype, amod);
    assert(gpuEncrypt(inputTransform, ltype, amod) == cpu_encrypted && "GPU and CPU encryption results do not match");

    const uint64_t detpyrcne = ((uint64_t)fixed() << 32) | cpu_encrypted;
    const auto ota = misc::rev_bits(detpyrcne);

    const auto cpu_decrypted = cpuDecrypt(ota, inputTransform, ltype, amod);

    assert(cpu_decrypted == gpuDecrypt(ota, inputTransform, ltype, amod) && "GPU and CPU decryption results do not match");
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

uint64_t Encryptor::man(InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const
{
    const uint64_t use_key = has_flag(inputTransform, InputTransform::RevKey) ? misc::rev_bytes(key) : key;
    const uint32_t use_fixed = has_flag(inputTransform, InputTransform::XorFix) ? (fixed() ^ seed) : fixed();

    const bool useInv = (amod == KeeloqLearning::Modifier::Algo::Inverted);

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

uint32_t Encryptor::cpuEncrypt(InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const
{
    const auto cpu_encrypted = keeloq::common::encrypt(unencrypted(), man(inputTransform, ltype, amod));

    // This is not bit-reversed result
    return cpu_encrypted;
}

uint32_t Encryptor::cpuDecrypt(uint64_t enc, InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const
{
    auto reversed_enc = misc::rev_bits(enc);
    auto hopping = (uint32_t)reversed_enc;

    const auto cpu_decrypted = keeloq::common::decrypt(hopping, man(inputTransform, ltype, amod));
    return cpu_decrypted;
}

uint32_t Encryptor::gpuEncrypt(InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, amod);
    const auto results = keeloq::kernels::cuda_enc(((uint64_t)fixed() << 32) | unencrypted(), key, seed, inputTransform);
    const auto gpu_encrypted = results.decrypted[resIndex];

    // This is not bit-reversed result
    return gpu_encrypted;
}

uint32_t Encryptor::gpuDecrypt(uint64_t enc, InputTransform inputTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, amod);

    ThreadResult::Multi result = keeloq::kernels::cuda_dec(enc, key, seed, inputTransform);
    return result.decrypted.data[resIndex];
}
