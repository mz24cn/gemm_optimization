//k*ernel void gemm_base(global float* out, const global float16* in, const global float16* weight, const global float16* bias, 
//		const int dim_hidden, const int dim_in)
//{
//	const int GID = get_global_id(0);
//	const int n = GID / dim_hidden;
//	const int hidden = GID % dim_hidden;
//	
//	weight += hidden * dim_in / 16;
//	in += n * dim_in / 16;
//	float16 z = bias != NULL? bias[hidden] : 0;
//
//	const int loop = dim_in / 16;
//	for (int i = 0; i < loop; i++)
//		z += weight[i] * in[i];
////		z += weight[dim_hidden * i + hidden] * in[i];
//	float8 sum8 = z.lo + z.hi;
//	float4 sum4 = sum8.lo + sum8.hi;
//	float2 sum2 = sum4.lo + sum4.hi;
//	float sum = sum2.lo + sum2.hi;
//	out[GID] = sum;
//}

kernel void gemm_base(global float* out, const global float* in, const global float* weight, const global float* bias,
		const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

#pragma unroll
	for (int i = 0; i < dim_in; i++)
		z += weight[dim_hidden * i + hidden] * in[in_offset + i];
	out[GID] = z;
}