#include "command_line_args.h"

#include <cuda_runtime_api.h>


bool CommandLineArgs::can_bruteforce()
{
	return inputs.size() > 0 && brute_configs.size() > 0;
}

void CommandLineArgs::init_inputs(const std::vector<uint64_t>& inp)
 {
	 inputs = inp;
 }

 void CommandLineArgs::init_cuda(uint16_t b, uint16_t t, uint16_t l)
 {
	 cuda_blocks	= b;
	 assert(cuda_blocks < max_cuda_blocks() && "This GPU cannot use this much blocks!");

	 cuda_threads	= t;
	 cuda_loops		= l;

	 if (cuda_threads == 0)
	 {
		 cuda_threads = (uint16_t)max_cuda_threads();
	 }
 }

 uint32_t CommandLineArgs::max_cuda_threads()
 {
	 cudaDeviceProp prop;
	 cudaGetDeviceProperties(&prop, 0);

	 return prop.maxThreadsPerBlock;
 }

 uint32_t CommandLineArgs::max_cuda_blocks()
 {
	 cudaDeviceProp prop;
	 cudaGetDeviceProperties(&prop, 0);

	 return prop.maxGridSize[0];
 }
