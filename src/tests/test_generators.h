#pragma once

#include "common.h"

/**
 * GPU-side decryptor-generator tests.
 */
namespace tests
{

namespace generators
{
/** Validates that the pattern generator produces the expected first decryptor. */
bool pattern();

/** Validates that the seed generator produces a contiguous, monotonically increasing seed sequence. */
bool seed();

/** Runs every generator test and returns true if all passed. */
bool all();

}

}
