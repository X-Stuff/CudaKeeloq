#include "kernels/kernel_input_single_learning.h"

#include "bruteforce/bruteforce_config.h"


size_t KeeloqKernelSingleLearningInput::BytesAllocated() const
{
    return TKeeloqKernelInputBase::BytesAllocated() + (results ? results->allocated() : 0);
}

void KeeloqKernelSingleLearningInput::BruteforcePrepare(InputsMutation mutations, KeeloqLearning::LearningType learningType, KeeloqLearning::Modifier::Algo algorithModifier)
{
    inputsMutation = mutations;
    learning = learningType;
    algorithModifier = algorithModifier;

    SetReady(true);
}