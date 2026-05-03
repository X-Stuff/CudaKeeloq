#include "keeloq_encryptor.h"

#include "keeloq_kernel.h"


EncParcel Encryptor::click(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod)
{
    const auto cpu_encrypted = cpu_encrypt(ltype, imod, amod);
    assert(gpu_encrypt(ltype, imod, amod) == cpu_encrypted && "GPU and CPU encryption results do not match");

    const uint64_t detpyrcne = ((uint64_t)fixed() << 32) | cpu_encrypted;
    const auto ota = misc::rev_bits(detpyrcne);

    const auto cpu_decrypted = cpu_decrypt(ota, ltype, imod, amod);

    assert(cpu_decrypted == gpu_decrypt(ota, ltype, imod, amod) && "GPU and CPU decryption results do not match");
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

uint64_t Encryptor::man(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod) const
{
    const uint64_t use_key = imod == KeeloqLearning::Modifier::Input::ReversedKey ? misc::rev_bytes(key) : key;

    const bool useInv = (amod == KeeloqLearning::Modifier::Algo::Inverted);

    switch (ltype)
    {
    case KeeloqLearning::LearningType::Simple:
    {
       return use_key;
    }
    case KeeloqLearning::LearningType::Normal:
    {
        return useInv ? keeloq::learning::normal<false>(fixed(), use_key) : keeloq::learning::normal<true>(fixed(), use_key);
    }
    case KeeloqLearning::LearningType::Secure:
    {
        return useInv ? keeloq::learning::secure<false>(fixed(), seed, use_key) : keeloq::learning::secure<true>(fixed(), seed, use_key);
    }
    case KeeloqLearning::LearningType::Xor:
    {
        return keeloq::learning::magic_xor_type1(fixed(), use_key);
    }
    case KeeloqLearning::LearningType::Faac:
    {
        // Faac uses encrypt by default (UseDecrypt=false); InvertedDec flips that to decrypt.
        return useInv ? keeloq::learning::faac<true>(seed, use_key) : keeloq::learning::faac<false>(seed, use_key);
    }
    case KeeloqLearning::LearningType::Serial1:
    {
        return keeloq::learning::serial_type1(fixed(), use_key);
    }
    case KeeloqLearning::LearningType::Serial2:
    {
        return keeloq::learning::serial_type2(fixed(), use_key);
    }
    case KeeloqLearning::LearningType::Serial3:
    {
        return keeloq::learning::serial_type3(fixed(), use_key);
    }
    default:
        assert(false && "Unknown learning type");
        return 0;
    }
}

uint32_t Encryptor::cpu_encrypt(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod) const
{
    const auto cpu_encrypted = keeloq::common::encrypt(unencrypted(), man(ltype, imod, amod));

    // This is not bit-reversed result
    return cpu_encrypted;
}

uint32_t Encryptor::cpu_decrypt(uint64_t enc, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod) const
{
    auto reversed_enc = misc::rev_bits(enc);
    auto hopping = (uint32_t)reversed_enc;

    const auto cpu_decrypted = keeloq::common::decrypt(hopping, man(ltype, imod, amod));
    return cpu_decrypted;
}

uint32_t Encryptor::gpu_encrypt(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, imod, amod);
    const auto results = keeloq::kernels::cuda_enc(((uint64_t)fixed() << 32) | unencrypted(), key, seed);
    const auto gpu_encrypted = results.decrypted[resIndex];

    // This is not bit-reversed result
    return gpu_encrypted;
}

uint32_t Encryptor::gpu_decrypt(uint64_t enc, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Input imod, KeeloqLearning::Modifier::Algo amod) const
{
    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, imod, amod);

    SingleResult result = keeloq::kernels::cuda_dec(enc, key, seed);
    return result.decrypted.data[resIndex];
}
