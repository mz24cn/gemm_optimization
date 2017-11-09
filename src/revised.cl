//#define LoadM 8
//#define LoadN 8
//#define GroupSizeM 16
//#define GroupSizeN 16
//#define TileK 16
__attribute__((reqd_work_group_size(GroupSizeM, GroupSizeN, 1)))
kernel void gemm_first(global float* C, const global float* B, const global float* A, const int N, const int M, const int K)
{
	const int ml = get_local_id(0);
	const int nl = get_local_id(1);
	const int mg = get_global_id(0);
	const int ng = get_global_id(1);
	const int ML = get_local_size(0);
	const int NL = get_local_size(1);
	const int MG = get_global_size(0);
	const int NG = get_global_size(1);
	
	local float TileA[TileK][LoadM * GroupSizeM + 3]; //bank conflict
	local float TileB[LoadN * GroupSizeN][TileK];
	
	float PrivateA;
	float PrivateB[LoadN];
	float sum[LoadM][LoadN];
#pragma unroll
	for (int i = 0; i < LoadM; i++)
#pragma unroll
		for (int j = 0; j < LoadN; j++)
			sum[i][j] = 0;

	int k = 0;
	do {
//#pragma unroll
//		for (int j = nl; j < TileK; j += NL)
//#pragma unroll
//			for (int i = 0; i < LoadM; i++)
////				TileA[j][LoadM * ml + i] = A[MG * i + mg + M * (k + j)];
//				TileA[j][LoadM * ml + i] = A[i + LoadM * mg + M * (k + j)];
#pragma unroll
		for (int j = nl; j < TileK; j += NL) {
			float8 vectorA = ((const global float8*) A)[(LoadM * mg + M * (k + j)) / 8];
			TileA[j][LoadM * ml + 0] = vectorA.s0;
			TileA[j][LoadM * ml + 1] = vectorA.s1;
			TileA[j][LoadM * ml + 2] = vectorA.s2;
			TileA[j][LoadM * ml + 3] = vectorA.s3;
			TileA[j][LoadM * ml + 4] = vectorA.s4;
			TileA[j][LoadM * ml + 5] = vectorA.s5;
			TileA[j][LoadM * ml + 6] = vectorA.s6;
			TileA[j][LoadM * ml + 7] = vectorA.s7;
		}

#pragma unroll
		for (int j = ml; j < TileK; j += ML)
#pragma unroll
			for (int i = 0; i < LoadN; i++)
				TileB[LoadN * nl + i][j] = B[K * (NG * i + ng) + k + j];

		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		for (int j = 0; j < TileK; j++) {
#pragma unroll
			for (int i = 0; i < LoadN; i++)
				PrivateB[i] = TileB[LoadN * nl + i][j];
#pragma unroll
			for (int i = 0; i < LoadM; i++) {
				PrivateA = TileA[j][LoadM * ml + i];
#pragma unroll
				for (int r = 0; r < LoadN; r++)
					sum[i][r] += PrivateA * PrivateB[r];
			}
		}
		
		k += TileK;
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	} while (k < K);

#pragma unroll
	for (int i = 0; i < LoadM; i++) {
		int row = LoadM * mg + i;
#pragma unroll
		for (int j = 0; j < LoadN; j++) {
			int column = NG * j + ng;
			C[row + column * M] = sum[i][j];
		}
	}
}
