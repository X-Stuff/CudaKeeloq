#include "kernels/kernel_input_multi_learning.h"

#include "common.h"

size_t KeeloqKernelMultiLearningInput::BytesAllocated() const
{
    return IKeeloqKernelInputBase::BytesAllocated() + (results ? results->allocated() : 0);
}

void KeeloqKernelMultiLearningInput::BruteforcePrepare(const KeeloqLearning::Matrix& inLearnings, InputsMutation mutations)
{
    assert(is_valid(mutations) && "Invalid input mutation mask");

    assert(GetConfig().type != BruteforceType::XorFix || !!(mutations & InputsMutation::XorFix) &&
        "In XorFix bruteforce you should have always XorFix mutation enabled");

    learnings = inLearnings;
    allLearnings = learnings.isAllEnabled();
    mutationsMask = mutations;

    SetReady(true);
}
