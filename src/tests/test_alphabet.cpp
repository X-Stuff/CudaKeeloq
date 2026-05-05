#include "doctest/doctest.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "device/cuda_vector.h"
#include "kernels/kernel_result.h"

#include "tests/support/keeloq_inputs.h"


TEST_CASE("alphabet generator: produces expected decryptor sequence")
{
    const CudaConfig Cuda = CudaConfig::Tests();

    const auto pattern = "abcd"_b;

    // 4^8 since we have 8 bytes and each byte can be 4 values
    const auto fullTurn = static_cast<uint32_t>(std::pow(pattern.size(), 8)); // 65536

    auto testConfig = BruteforceConfig::GetAlphabet(Decryptor::Make(0, 0, true), pattern, 0xFFFFFFFF);
    auto inputs = tests::keeloq::genInputs(0x6161616161616161);

    CudaVector<Decryptor> decryptors(Cuda.total());

    KeeloqKernelInput generatorInputs;
    generatorInputs.decryptors = decryptors.gpu();
    generatorInputs.Initialize(testConfig, inputs, KeeloqLearning::Matrix(KeeloqLearning::Matrix::kEverything));

    const auto fullCyclesNum = fullTurn / Cuda.total();

    for (uint32_t i = 0; i < fullCyclesNum; ++i)
    {
        GeneratorBruteforce::PrepareDecryptors(generatorInputs, Cuda);
        decryptors.read();

        CHECK((decryptors.cpu()[0].man() & 0x000000FFFFFFFFFF) == 0x6161616161);

        generatorInputs.NextDecryptor();
    }

    CHECK(decryptors.cpu()[Cuda.total() - 1].man() == 0x6464646464646464);
}
