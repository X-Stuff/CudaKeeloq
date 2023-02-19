#include "tests/test_alphabet.h"

#include "device/cuda_vector.h"
#include "kernels/kernel_result.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "algorithm/keeloq/keeloq_decryptor.h"


bool tests::alphabet_generation()
{
    // Filtered generator test itself
    constexpr auto NumBlocks = 64;
    constexpr auto NumThreads = 64;

    auto testConfig = BruteforceConfig::GetAlphabet(Decryptor(0,0), "abcd"_b, 0xFFFFFFFF);

    CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);

    KeeloqKernelInput generatorInputs(nullptr, decryptors.gpu(), nullptr, testConfig);

    for (int i = 0; i < 16; ++i)
    {
        GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);

        decryptors.read();

        assert((decryptors.cpu()[0].man() & 0x0000FFFFFFFFFFFF) == 0x616161616161);

        generatorInputs.NextDecryptor();
    }

    assert(decryptors.cpu()[4095].man() == 0x6464646464646464);

    return true;
}