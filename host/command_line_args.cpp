#include "command_line_args.h"

#include <cuda_runtime_api.h>


bool CommandLineArgs::isValid()
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
	 cuda_threads	= t;
	 cuda_loops		= l;

	 if (cuda_threads == 0)
	 {
		 cudaDeviceProp prop;
		 cudaGetDeviceProperties(&prop, 0);
		 cuda_threads = prop.maxThreadsPerBlock;
	 }
 }
