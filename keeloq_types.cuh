#pragma once

#include <string>
#include <tuple>

#include "CUDA_helpers.cuh"

// forward declarations
enum class SmartFilterFlags : uint64_t;

extern const char* LearningNames[];
extern const char* GeneratorTypeName[];
extern const size_t GeneratorTypeNamesCount;
extern const std::vector<std::tuple<SmartFilterFlags, const char*>> FilterNames;


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


struct SingleResult
{
    static constexpr uint8_t ResultsCount = (uint8_t)KeeloqLearningType::LAST;

    struct DecryptedArray
    {
        // fixed side array for every learning type
        uint32_t data[ResultsCount];

        void print(uint8_t element, bool ismatch) const;

        void print() const;
    };

    // Per each learning type
    DecryptedArray results;

    // used manufactorer key and seed for this result
    uint64_t man;
    uint32_t seed;

    // Input data
    uint64_t ota;

    // Set by GPU after analysis if there was a match
    KeeloqLearningType match;

    void print(bool onlymatch = true) const;
};

// is test manufacture code with seed (default is 0)
struct Decryptor
{
    uint64_t man;
    uint32_t seed;

    Decryptor() = default;
    Decryptor(uint64_t key, uint32_t s = 0) : man(key), seed(s) {}

    __host__ __device__ inline bool operator==(const Decryptor& other) {
        return man == other.man && seed == other.seed;
    }
    __host__ __device__ inline bool operator<(const Decryptor& other) {
        return man < other.man;
    }
};


// Single run Configuration
struct BruteforceConfig
{
    // CUDA bruteforce generator
    enum class Type : uint8_t
    {
        // Generation will be skipped
        Dictionary = 0,
        None = Dictionary,

        // Simple +1 generator (vert fast in terms of genration of decryptors candidates)
        Simple,

        // Simple +1 bruteforce but with filters applied performance may degrage)
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
        // Actual size of alphabet
        uint8_t num = 0;

        Alphabet() = default;

        // Duplicates will be removed
        Alphabet(std::vector<uint8_t> alphabet);

        __host__ __device__ inline bool add(uint8_t number[sizeof(uint64_t)], uint64_t value)
        {
#ifdef __CUDA_ARCH__
            #pragma unroll
#endif
            for (int i = 0; i < sizeof(uint64_t); ++i)
            {
                uint8_t digit = value % num;                // 17 % 6 = 5
                uint16_t addition = number[i] + digit;      // 5 + 5 = 10 (6 + 4)
                number[i] = addition % num;                 // n[i] = 4

                value /= num;                               // 17 / 6 = 2

                uint8_t carry = addition >= num;            // 10 > 6
                value += carry;                             // 2 + 1 = 3
            }
        }

        // return index of value (cannot fail - if value not in a LUT - always return 0 index)
        __host__ __device__ inline uint8_t lookup(uint8_t value) const {
            return lut[value];
        }

        // lookup foreach byte
        __host__ __device__ inline uint64_t lookup(uint64_t value) const
        {
            uint64_t result = 0;
            uint8_t* pResult = (uint8_t*)&result;
            uint8_t* pValue = (uint8_t*)&value;
#ifdef __CUDA_ARCH__
            #pragma unroll
#endif
            for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
            {
                // Valid or 0 (first letter in alphabet)
                pResult[i] = lookup(pValue[i]);
            }
            return result;
        }

        // return value by index
        __host__ __device__ inline uint8_t operator[](uint8_t index) {
            return alp[index];
        }

        __host__ std::string toString() const;

    private:
        static const uint16_t Size = 0xFF + 1; // 256

        // The alphabet itself (256 bytes max)
        uint8_t alp[Size] = {0};

        // The alphabet lookup table
        uint8_t lut[Size] = {0};

    };

    struct Filters
    {
        // Filter for keys to include.
        // WARNING:
        //  Could be executed INFINITELY LONG TIME
        //  e.g. start: 0x00000000001 filter SmartFilterFlags::AsciiAny
        //  it will took around trillions and trillions operations just to get to the first valid with simple +1
        //  In case of specific input - use dictionary, pattern or alphabet
        SmartFilterFlags include = SmartFilterFlags::All;

        // Filter for keys to exclude
        SmartFilterFlags exclude = SmartFilterFlags::None;

        std::string toString(SmartFilterFlags flags) const;

        std::string toString() const;
    };

    // HOST SET. ONCE. Which generator to use.
    Type type;

    // HOST SET. UPDATING. PER BATCH. Decryption batch (or decrypters generation) will start from this
    Decryptor start;

    // HOST SET. ONCE. How many generator rounds should be taken (in fact how many times cuda kernel will be called)
    size_t size;

    // Dictionery - HOST SET. ONCE.
    // Brute -      GPU SET. UPDATING.
    std::vector<Decryptor> decryptors;

    // HOST SET. ONCE. for filtered type.
    Filters filters;

    // HOST SET. ONCE. for alphabet type.
    Alphabet alphabet;

    // GPU SET. UPDATING. Last generated decryptor (will be initial for next block run)
    Decryptor next;

    BruteforceConfig() : BruteforceConfig(0, Type::LAST, 0) {
    }

    static BruteforceConfig GetDictionary(const std::vector<Decryptor>&& dictionary)
    {
        BruteforceConfig result(0, Type::Dictionary, dictionary.size());
        result.decryptors = std::move(dictionary);
        return result;
    };

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size) { return BruteforceConfig(first, Type::Simple, size); }

    static BruteforceConfig GetBruteforce(Decryptor first, size_t size, const Filters& filters)
    {
        BruteforceConfig result(first, Type::Filtered, size);
        result.filters = filters;
        return result;
    }

    static BruteforceConfig GetAlphabet(Decryptor first, const Alphabet& alphabet, size_t num = (size_t)-1)
    {
        // max operation take to check all keys with alphabet of size
        num = min((uint64_t)pow(alphabet.num, sizeof(uint64_t)), num);

        BruteforceConfig result(first, Type::Alphabet, num);
        result.alphabet = alphabet;
        return result;
    }

    static BruteforceConfig GetPattern()
    {
        // not implemented
        return BruteforceConfig(0, Type::LAST, 0);
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

    std::string toString() const;

private:
    BruteforceConfig(Decryptor start, Type gen, size_t num) :
        start(start), type(gen), next(start), size(num)
    {
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

struct CommandLineArgs
{
    // Input recevied encrypted data
    std::vector<EncData> inputs;

    // How brute will be performed (may be several iterations)
    std::vector<BruteforceConfig> brute_configs;

    // Stop on forst match
    bool match_stop;

    uint16_t cuda_blocks;
    uint16_t cuda_threads;
    uint16_t cuda_loops;

    // Do not do all 16 calculations, use predefined one
    KeeloqLearningType selected_learning = KeeloqLearningType::INVALID;

    bool run_tests;

    inline void init_inputs(const std::vector<uint64_t> inp) {
        inputs = inp;
    }

    inline void init_cuda(uint16_t b, uint16_t t, uint16_t l) {
        cuda_blocks = b; cuda_threads = t; cuda_loops = l;
    }

    inline bool isValid() {
        return inputs.size() > 0 && brute_configs.size() > 0;
    }
};

inline std::vector<uint8_t> operator "" _b(const char* ascii, size_t num)
{
    return std::vector<uint8_t>(ascii, ascii + num);
}