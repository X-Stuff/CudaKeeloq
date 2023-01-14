#pragma once

#include <string>
#include "CUDA_helpers.cuh"

// forward declaration
extern const char* LearningNames[];
extern const char* GeneratorTypeName[];


// Input encoded data (received over the air) - 16bytes
typedef uint64_t EncData;

enum KeeloqLearningType
{
    Simple = 0,
    Simple_Rev,

    Normal,
    Normal_Rev,

    Secure,
    Secure_Rev,

    Xor,
    Xor_Rev,

    Faac,
    Faac_Rev,

    Serial1,
    Serial1_Rev,

    Serial2,
    Serial2_Rev,

    Serial3,
    Serial3_Rev,

    LAST,

    INVALID = 0xffffffff,
};

enum class SmartFilterFlags : uint64_t
{
    //
    None = 0,

    // filter fuction return true if key has more than 6 consecutive 0 bits
    Max6ZerosInARow         = (1 << 0),

    // filter fuction return true if key has more than 6 consecutive 1 bits
    Max6OnesInARow          = (1 << 1),


    // filter fuction return true if key has patterns like 11:22:33:44.. or FF:EE:DD:CC
    // 6 bytes by default
    BytesIncremental    = (1 << 5),

    // filter fuction return true if key has repeating patterns like xx11:11:11:11xx or xxAA:AA:AA:AAxx
    BytesRepeat4        = (1 << 6),

    // filter fuction return true if key consist from only ascii numbers
    AsciiNumbers        = (1 << 11),

    // filter fuction return true if key consist from only letters 'a'-'z' 'A'-'Z'
    AsciiAlpha          = (1 << 12),

    // filter fuction return true if key consist from ascii letters and numbers
    AsciiAlphaNum       = AsciiAlpha | AsciiNumbers,

    // filter fuction return true if key consist from only ASCII special symbols like '^%#&*
    AsciiSpecial        = (1 << 13),

    // filter fuction return true if key consist from only ASCII typed characters
    AsciiAny      = AsciiAlphaNum | AsciiSpecial,

    //
    All = (uint64_t)-1,
};


// fixed side array for every learning type
struct DecryptedArray
{
    uint32_t data[KeeloqLearningType::LAST];
};


struct SingleResult
{
    DecryptedArray results;

    uint64_t man;
    uint32_t seed;

    uint64_t ota;

    KeeloqLearningType match;
};

// is test manufacture code with seed (default is 0)
struct Decryptor
{
    uint64_t man;
    uint32_t seed;

    Decryptor() = default;
    Decryptor(uint64_t key, uint32_t s = 0) :
        man(key), seed(s) {}
};


// Generator config - is setup for bruteforce
// In case of dictionary attack - does nothing
struct BruteforceConfig
{
    // CUDA bruteforce generator
    enum class Type : uint8_t
    {
        // Generation will be skipped
        None = 0,
        Dictionary = 0,

        // Simple +1 generator (vert fast in terms of genration of decryptors candidates)
        Simple,

        // Excluding obvious patterns
        Filtered,

        // Specify alphabet and brute over it
        Alphabet,

        // ASCII pattern like ss:01:??:3?:*
        Pattern,

        // Not for usage
        LAST,
    };

    struct Alphabet
    {
        static const uint16_t Size = 0xFF + 1; // 256

        // Actual size of alphabet
        uint8_t num = 0;

        Alphabet()
        {
        }

        Alphabet(const std::vector<uint8_t>& alphabet)
        {
            assert(alphabet.size() < Alphabet::Size && "Using all bytes values as alphabet is not efficient");

            num = (uint8_t)min(alphabet.size(), (size_t)Size);
            memcpy(alp, alphabet.data(), num * sizeof(uint8_t));

            memset(lut, 0xFF, num * sizeof(uint8_t));
            for (uint8_t i = 0; i < num; ++i)
            {
                lut[alp[i]] = i;
            }
        }

        const uint8_t* alphabet() { return alp; }



        __host__ __device__ inline bool add(uint8_t number[sizeof(uint64_t)], uint64_t value)
        {
            #pragma unroll
            for (int i = 0; i < sizeof(uint64_t); ++i)
            {
                uint8_t digit = value % num;                // 17 % 6 = 5
                uint16_t addition = number[i] + digit;      // 5 + 5 = 10 (6 + 4)
                number[i] = addition % num;                 // n[i] = 4

                value /= num;                               // 17 / 6 = 2

                uint8_t carry = addition > num;             // 10 > 6
                value += carry;                             // 2 + 1 = 3
            }
        }

        __host__ __device__ inline bool is_valid_index(uint8_t index) {
            return num == Alphabet::Size || lut[index] < num;
        }

        // return index of value
        __host__ __device__ inline uint8_t lookup(uint8_t value) {
            return lut[value];
        }

        // return value by index
        __host__ __device__ inline uint8_t operator[](uint8_t index) {
            return alp[index];
        }

    private:
        // The alphabet itself (256 bytes max)
        uint8_t alp[Size];

        // The alphabet lookup table
        uint8_t lut[Size];

    };

    struct Filters
    {
        // Filter for keys to include.
        // WARNINGL:
        //  Could be executed INFINITELY LONG TIME
        //  e.g. start: 0x00000000001 filter SmartFilterFlags::AsciiAny
        //  it will took around trillions and trillions operations just to get to the first valid with simple +1
        //  In case of specific input - use dictionary, pattern or alphabet
        SmartFilterFlags include = SmartFilterFlags::All;

        // Filter for keys to exclude
        SmartFilterFlags exclude = SmartFilterFlags::None;
    };

    // HOST SET. ONCE. How many generator rounds should be taken (in fact how many times cuda kernel will be called)
    size_t size;

    // HOST SET. ONCE. Decryption will start from this
    Decryptor start;

    // HOST SET. ONCE. Which generator to use.
    Type type;

    // HOST SET. ONCE. for filtered type.
    Filters filters;

    // HOST SET. ONCE. for alphabet type.
    Alphabet alphabet;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor next;

    BruteforceConfig() :
        BruteforceConfig(Type::None, 0) {

    }

    BruteforceConfig(Type gen, size_t num) :
        BruteforceConfig(0, gen, num)
    {
        assert(gen == Type::Dictionary && "This constructor is available only for dictionary type");
    }

    BruteforceConfig(Decryptor start, Type gen, size_t num) :
        start(start), type(gen), next(start), size(num)
    {
    }

    uint64_t dict_size() const {
        if (type == Type::Dictionary) {
            return size;
        }
        return 0;
    }

    uint64_t brute_size() const {
        if (type != Type::Dictionary) {
            return size;
        }
        return 0;
    }

    void update_decryptor() {
        if (type != Type::Dictionary) {
            start = next;
        }
    }

    std::string ToString() const
    {
        char tmp[128];
        if (type == Type::Dictionary) {
            sprintf_s(tmp, "Type: %s. size: %zd", GeneratorTypeName[(uint8_t)type % (uint8_t)Type::LAST], dict_size());
        }
        else {
            sprintf_s(tmp, "Type: %s. Initial: 0x%llX (seed:%ul). Brute size: %zd",
                GeneratorTypeName[(uint8_t)type % (uint8_t)Type::LAST], start.man, start.seed, brute_size());
        }
        return std::string(tmp);
    }
};

// Input data for main keeloq calculation kernel
struct KernelInput : TGenericGpuObject<KernelInput>
{
    // Constant per-run input data (captured encoded)
    CUDA_Array<EncData>* encdata;

    // Single-run set of dectryptors
    CUDA_Array<Decryptor>* decryptors;

    // Single-run results
    CUDA_Array<SingleResult>* results;

    // from this dectryptor generation will start
    BruteforceConfig generator;

    KernelInput() : KernelInput(nullptr, nullptr, nullptr, BruteforceConfig())
    {
    }

    KernelInput(CUDA_Array<EncData>* enc, CUDA_Array<Decryptor>* dec, CUDA_Array<SingleResult>* res, const BruteforceConfig& config)
        : TGenericGpuObject<KernelInput>(this), encdata(enc), decryptors(dec), results(res), generator(config)
    {
    }

    KernelInput(KernelInput&& other) : TGenericGpuObject<KernelInput>(this) {
        encdata = other.encdata;
        decryptors = other.decryptors;
        results = other.results;
        generator = other.generator;
    }

    KernelInput& operator=(KernelInput&& other) = delete;
    KernelInput& operator=(const KernelInput& other) = delete;

    inline void WriteDecryptors(const std::vector<Decryptor>& source, size_t from, size_t num)
    {
        if (decryptors != nullptr)
        {
            assert(generator.type == BruteforceConfig::Type::Dictionary);

            size_t copy_num = max(0ull, min(num, (source.size() - from)));
            decryptors->write(&source[from], copy_num);
        }
    }

    inline void UpdateInitialDecryptor()
    {
        assert(generator.type != BruteforceConfig::Type::Dictionary);
        generator.update_decryptor();
    }
};

//
//
struct KernelResult : TGenericGpuObject<KernelResult>
{
    // num errors. negative are kernel errors. positive - number of threads error
    int error = 0;

    // overall result
    int value = 0;

    KernelResult() : TGenericGpuObject<KernelResult>(this) {
    }

    KernelResult(KernelResult&& other) : TGenericGpuObject<KernelResult>(this) {
        error = other.error;
        value = other.value;
    }
};

static const char* LearningNames[KeeloqLearningType::LAST] = {
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

static const char* GeneratorTypeName[(int)BruteforceConfig::Type::LAST] = {
    "Dictionary",
    "Simple",
    "Filtered",
    "Alphabet",
    "Pattern"
};