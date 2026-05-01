#include "tests/test_alphabet.h"
#include "tests/test_keeloq.h"

#include "device/cuda_vector.h"
#include "kernels/kernel_result.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/generators/generator_bruteforce.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "algorithm/keeloq/keeloq_decryptor.h"


bool tests::alphabet_generation()
{
    // Filtered generator test itself
    constexpr CudaConfig Cuda { 64, 64, 1};

    auto testConfig = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 0, true), "abcd"_b, 0xFFFFFFFF);
    auto inputs = keeloq::gen_inputs(0x6161616161616161);

    CudaVector<Decryptor> decryptors(Cuda.blocks * Cuda.threads);

    KeeloqKernelInput generatorInputs;
    generatorInputs.decryptors = decryptors.gpu();
    generatorInputs.Initialize(testConfig, inputs, KeeloqLearning::Matrix(KeeloqLearning::Matrix::kEverything));

    for (int i = 0; i < 16; ++i)
    {
        GeneratorBruteforce::PrepareDecryptors(generatorInputs, Cuda);

        decryptors.read();

        assert((decryptors.cpu()[0].man() & 0x0000FFFFFFFFFFFF) == 0x616161616161);

        generatorInputs.NextDecryptor();
    }

    assert(decryptors.cpu()[4095].man() == 0x6464646464646464);

    return true;
}