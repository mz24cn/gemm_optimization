//k*ernel void gemm(global float* out, const global float* in, const global float* weight, const global float* bias, 
//		/*local float* tmp, */const int dim_hidden, const int dim_in)
//{
//	const int GID = get_global_id(0);
//	const int n = GID / dim_hidden;
//	const int hidden = GID % dim_hidden;
//	const int weight_offset = hidden * dim_in;
//	
//	const int in_offset = n * dim_in;
////	const int batch_size = get_global_size(0) / dim_hidden;
//	float z = bias != NULL? bias[hidden] : 0;
//
//	for (int i = 0; i < dim_in; i++)
//		z += weight[weight_offset + i] * in[in_offset + i];
//	out[GID] = z;
//}

kernel void gemm_version2(global float* out, const global float* in, const global float* weight, const global float* bias, 
		/*local float* tmp, */const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int weight_offset = hidden * dim_in;
	
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

	for (int i = 0; i < dim_in; i++)
		z += weight[dim_hidden * i + hidden] * in[in_offset + i];
	out[GID] = z;
}