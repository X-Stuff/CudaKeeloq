#include "keeloq_encryptor.h"

#include "keeloq_kernel.h"


EncParcel Encryptor::click(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod)
{
    const uint32_t fix = fixed();

    // ReversedKey uses the byte-reversed key; Regular and InvertedDec use the original key.
    const uint64_t key_rev = misc::rev_bytes(key);
    const bool     useRevKey = (lmod == KeeloqLearning::Modifier::Type::ReversedKey);
    const bool     useInv = (lmod == KeeloqLearning::Modifier::Type::InvertedDec);
    const uint64_t base_key = useRevKey ? key_rev : key;

    uint64_t n_key = base_key;

    switch (ltype)
    {
    case KeeloqLearning::LearningType::Simple:
        n_key = base_key;
        break;
    case KeeloqLearning::LearningType::Normal:
        n_key = useInv ? keeloq::learning::normal<false>(fix, base_key) : keeloq::learning::normal<true>(fix, base_key);
        break;
    case KeeloqLearning::LearningType::Secure:
        assert(seed != 0);
        n_key = useInv ? keeloq::learning::secure<false>(fix, seed, base_key) : keeloq::learning::secure<true>(fix, seed, base_key);
        break;
    case KeeloqLearning::LearningType::Xor:
        n_key = keeloq::learning::magic_xor_type1(fix, base_key);
        break;
    case KeeloqLearning::LearningType::Faac:
        assert(seed != 0);
        // Faac uses encrypt by default (UseDecrypt=false); InvertedDec flips that to decrypt.
        n_key = useInv ? keeloq::learning::faac<true>(seed, base_key) : keeloq::learning::faac<false>(seed, base_key);
        break;
    case KeeloqLearning::LearningType::Serial1:
        n_key = keeloq::learning::serial_type1(fix, base_key);
        break;
    case KeeloqLearning::LearningType::Serial2:
        n_key = keeloq::learning::serial_type2(fix, base_key);
        break;
    case KeeloqLearning::LearningType::Serial3:
        n_key = keeloq::learning::serial_type3(fix, base_key);
        break;
    default:
        assert(false && "Unknown learning type");
        break;
    }

    const auto results = keeloq::kernels::cuda_encdec(((uint64_t)fix << 32) | unencrypted(), n_key, seed, false);
    const auto& hops = results.decrypted.data;

    const uint64_t detpyrcne = ((uint64_t)fix << 32) | keeloq::common::encrypt(unencrypted(), n_key);
    const auto encrypted = misc::rev_bits(detpyrcne, sizeof(detpyrcne) * 8);

    validate(encrypted, *this, ltype, lmod);

    count++;
    return encrypted;
}

bool Encryptor::validate(uint64_t ota, const Encryptor& enc, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod)
{
    KeeloqLearning::Matrix matrix;
    matrix.enable(ltype, lmod);

    const uint8_t resIndex = KeeloqLearning::DecryptedResults::getIndex(ltype, lmod);
    const bool useRev = (lmod == KeeloqLearning::Modifier::Type::ReversedKey);

    const uint64_t dec_key = useRev ? misc::rev_bytes(enc.key) : enc.key;

    SingleResult result = keeloq::kernels::cuda_encdec(ota, dec_key, enc.seed, true);
    const bool gpu_ok = result.decrypted.data[resIndex] == enc.unencrypted();

    assert(gpu_ok && "GPU calculation was invalid");
    return gpu_ok;
}
