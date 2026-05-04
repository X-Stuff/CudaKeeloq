#pragma once

struct CommandLineArgs;

/**
 * Command-line parser smoke tests.
 */
namespace tests
{
namespace console
{
    /** Parse a battery of example command lines and assert the expected args are set. */
    void run();
}
}
