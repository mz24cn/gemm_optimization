kernel void gemm_base(global float* out, const global float* in, const global float* weight, const global float* bias, 
		const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

	for (int i = 0; i < dim_in; i++)
		z += weight[dim_hidden * i + hidden] * in[in_offset + i];
	out[GID] = z;
}