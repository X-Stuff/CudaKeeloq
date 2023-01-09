#include "stdint.h"

#define NLF_LOOKUP_CONSTANT 0x3a5c742e

#define bit(x, n) (((x) >> (n)) & 1)
#define g5(x, a, b, c, d, e) \
    (bit(x, a) + bit(x, b) * 2 + bit(x, c) * 4 + bit(x, d) * 8 + bit(x, e) * 16)

#define KEELOQ_LEARNING_SIMPLE 0u
#define KEELOQ_LEARNING_SIMPLE_REV KEELOQ_LEARNING_SIMPLE + 1u

#define KEELOQ_LEARNING_NORMAL KEELOQ_LEARNING_SIMPLE_REV + 1u
#define KEELOQ_LEARNING_NORMAL_REV KEELOQ_LEARNING_NORMAL + 1u

#define KEELOQ_LEARNING_SECURE KEELOQ_LEARNING_NORMAL_REV + 1u
#define KEELOQ_LEARNING_SECURE_REV KEELOQ_LEARNING_SECURE + 1u

#define KEELOQ_LEARNING_MAGIC_XOR_TYPE_1 KEELOQ_LEARNING_SECURE_REV + 1u
#define KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV KEELOQ_LEARNING_MAGIC_XOR_TYPE_1 + 1u

#define KEELOQ_LEARNING_FAAC KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV + 1u
#define KEELOQ_LEARNING_FAAC_REV KEELOQ_LEARNING_FAAC + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1 KEELOQ_LEARNING_FAAC_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1 + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2 KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2 + 1u

#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3 KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV + 1u
#define KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3 + 1u


#define KEELOQ_LEARNING_LAST KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV + 1u
#define KEELOQ_LEARNING_INVALID 0xff

static const char* LearningNames[KEELOQ_LEARNING_LAST] = {
    "KEELOQ_LEARNING_SIMPLE",
    "KEELOQ_LEARNING_SIMPLE_REV",
    "KEELOQ_LEARNING_NORMAL",
    "KEELOQ_LEARNING_NORMAL_REV",
    "KEELOQ_LEARNING_SECURE",
    "KEELOQ_LEARNING_SECURE_REV",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV",
    "KEELOQ_LEARNING_FAAC",
    "KEELOQ_LEARNING_FAAC_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3",
    "KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV",
};


struct DecryptedArray
{
    uint32_t data[KEELOQ_LEARNING_LAST];
};


#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __host__
#define __host__
#endif

__device__ __host__ inline uint32_t keeloq_common_decrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x << 1) ^ bit(x, 31) ^ bit(x, 15) ^ (uint32_t)bit(key, (15 - r) & 63) ^
        bit(NLF_LOOKUP_CONSTANT, g5(x, 0, 8, 19, 25, 30));
    return x;
}

__device__ __host__ inline uint32_t keeloq_common_encrypt(const uint32_t data, const uint64_t key) {
    uint32_t x = data, r;
    for(r = 0; r < 528; r++)
        x = (x >> 1) ^ ((bit(x, 0) ^ bit(x, 16) ^ (uint32_t)bit(key, r & 63) ^
            bit(NLF_LOOKUP_CONSTANT, g5(x, 1, 9, 20, 26, 31)))
            << 31);
    return x;
}

__device__ __host__ inline bool decrypt_valid(uint32_t dectrypted, uint32_t check) {
    return false;
}

__device__ __host__ inline uint64_t keeloq_common_secure_learning(uint32_t data, uint32_t seed, const uint64_t key) {
    uint32_t k1, k2;

    data &= 0x0FFFFFFF;
    k1 = keeloq_common_decrypt(data, key);
    k2 = keeloq_common_decrypt(seed, key);

    return ((uint64_t)k1 << 32) | k2;
}

__device__ __host__ inline uint64_t keeloq_common_magic_xor_type1_learning(uint32_t data, uint64_t xor) {
    data &= 0x0FFFFFFF;
    return (((uint64_t)data << 32) | data) ^ xor;
}

__device__ __host__ inline uint64_t keeloq_common_normal_learning(uint32_t data, const uint64_t key) {
    uint32_t k1, k2;

    data &= 0x0FFFFFFF;
    data |= 0x20000000;
    k1 = keeloq_common_decrypt(data, key);

    data &= 0x0FFFFFFF;
    data |= 0x60000000;
    k2 = keeloq_common_decrypt(data, key);

    return ((uint64_t)k2 << 32) | k1; // key - shifrovanoya
}

__device__ __host__ inline uint64_t keeloq_common_faac_learning(const uint32_t seed, const uint64_t key) {
    uint16_t hs = seed >> 16;
    const uint16_t ending = 0x544D;
    uint32_t lsb = (uint32_t)hs << 16 | ending;
    uint64_t man = (uint64_t)keeloq_common_encrypt(seed, key) << 32 | keeloq_common_encrypt(lsb, key);
    return man;
}

__device__ __host__ inline uint64_t keeloq_common_magic_serial_type1_learning(uint32_t data, uint64_t man) {
    return (man & 0xFFFFFFFF) | ((uint64_t)data << 40) |
        ((uint64_t)(((data & 0xff) + ((data >> 8) & 0xFF)) & 0xFF) << 32);
}

__device__ __host__ inline uint64_t keeloq_common_magic_serial_type2_learning(uint32_t data, uint64_t man) {
    uint8_t* p = (uint8_t*)&data;
    uint8_t* m = (uint8_t*)&man;
    m[7] = p[0];
    m[6] = p[1];
    m[5] = p[2];
    m[4] = p[3];
    return man;
}

__device__ __host__ inline uint64_t keeloq_common_magic_serial_type3_learning(uint32_t data, uint64_t man) {
    return (man & 0xFFFFFFFFFF000000) | (data & 0xFFFFFF);
}

__device__ __host__ inline uint64_t keeloq_common_key_reverse(uint64_t key, uint8_t bit_count) {
    uint64_t reverse_key = 0;
    for(uint8_t i = 0; i < bit_count; i++) {
        reverse_key = reverse_key << 1 | bit(key, i);
    }
    return reverse_key;
}


__device__ __host__ struct DecryptedArray keeloq_decrypt_all(uint32_t data, uint32_t fix, const uint64_t key, const uint32_t seed) {

    struct DecryptedArray decrypted = {0};

    // Check for mirrored man
    uint64_t key_rev = 0;
    uint64_t key_rev_byte = 0;
    for(uint8_t i = 0; i < 64; i += 8) {
        key_rev_byte = (uint8_t)(key >> i);
        key_rev = key_rev | key_rev_byte << (56 - i);
    }

    uint64_t n_key = 0;

    // Simple Learning
    {
        decrypted.data[KEELOQ_LEARNING_SIMPLE] = keeloq_common_decrypt(data, key);
        decrypted.data[KEELOQ_LEARNING_SIMPLE_REV] = keeloq_common_decrypt(data, key_rev);
    }

    //###########################
    // Normal Learning
    // https://phreakerclub.com/forum/showpost.php?p=43557&postcount=37
    {
        n_key = keeloq_common_normal_learning(fix, key);
        decrypted.data[KEELOQ_LEARNING_NORMAL] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_normal_learning(fix, key_rev);
        decrypted.data[KEELOQ_LEARNING_NORMAL_REV] = keeloq_common_decrypt(data, n_key);
    }

    // Secure Learning
    if (seed != 0)
    {
        n_key = keeloq_common_secure_learning(fix, seed, key);
        decrypted.data[KEELOQ_LEARNING_SECURE] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_secure_learning(fix, seed, key_rev);
        decrypted.data[KEELOQ_LEARNING_SECURE_REV] = keeloq_common_decrypt(data, n_key);
    }

    // Magic xor type1 learning
    {
        n_key = keeloq_common_magic_xor_type1_learning(fix, key);
        decrypted.data[KEELOQ_LEARNING_MAGIC_XOR_TYPE_1] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_magic_xor_type1_learning(fix, key_rev);
        decrypted.data[KEELOQ_LEARNING_MAGIC_XOR_TYPE_1_REV] = keeloq_common_decrypt(data, n_key);
    }

    // FAAC
    if (seed != 0)
    {
        n_key = keeloq_common_faac_learning(seed, key);
        decrypted.data[KEELOQ_LEARNING_FAAC] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_faac_learning(seed, key_rev);
        decrypted.data[KEELOQ_LEARNING_FAAC_REV] = keeloq_common_decrypt(data, n_key);
    }

    // SERIAL_TYPE_1
    {
        n_key = keeloq_common_magic_serial_type1_learning(fix, key);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_magic_serial_type1_learning(fix, key_rev);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_1_REV] = keeloq_common_decrypt(data, n_key);
    }

    // SERIAL_TYPE_2
    {
        n_key = keeloq_common_magic_serial_type2_learning(fix, key);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_magic_serial_type2_learning(fix, key_rev);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_2_REV] = keeloq_common_decrypt(data, n_key);
    }

    // SERIAL_TYPE_3
    {
        n_key = keeloq_common_magic_serial_type3_learning(fix, key);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3] = keeloq_common_decrypt(data, n_key);

        n_key = keeloq_common_magic_serial_type3_learning(fix, key_rev);
        decrypted.data[KEELOQ_LEARNING_MAGIC_SERIAL_TYPE_3_REV] = keeloq_common_decrypt(data, n_key);
    }

    return decrypted;
}

__device__ __host__ struct DecryptedArray keeloq_decrypt(uint64_t ota, uint64_t man, uint32_t seed = 0) {

    uint64_t key = keeloq_common_key_reverse(ota, sizeof(ota) * 8);

    uint32_t fix = (uint32_t)(key >> 32);
    uint32_t hop = (uint32_t)(key);


    return keeloq_decrypt_all(hop, fix, man, seed);
}
