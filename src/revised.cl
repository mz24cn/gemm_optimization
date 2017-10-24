kernel void gemm_unroll(global float* out, const global float* in, const global float* weight, const global float* bias,
		const int dim_hidden, const int dim_in)
{
	const int out_row = get_global_id(0);
	const int out_column = get_global_id(1);
	const int in_offset = dim_in * out_column;
	float z = bias != NULL? bias[out_row] : 0;

//#pragma unroll 8
	for (int i = 0; i < dim_in; i++)
		z += weight[dim_hidden * i + out_row] * in[in_offset + i];
	out[get_global_size(0) * out_column + out_row] = z;
}

//#define TSM 128                // The tile-size in dimension M
//#define TSN 128                // The tile-size in dimension N
//#define TSK 16                 // The tile-size in dimension K
//#define WPTM 8                 // The work-per-thread in dimension M
//#define WPTN 8                 // The work-per-thread in dimension N
//#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
//#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
//#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
//#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B
//kernel void gemm_version3(global float* out, const global float* in, const global float* weight, const global float* bias, 
//		/*local float* tmp, */const int dim_hidden, const int dim_in)
//{
//	// Thread identifiers
//	const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM)
//	const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN)
//	const int offsetM = TSM*get_group_id(0); // Work-group offset
//	const int offsetN = TSN*get_group_id(1); // Work-group offset
//
//	// Local memory to fit a tile of A and B
//	local float Asub[TSK][TSM];
//	local float Bsub[TSK][TSN];
//
//	// Allocate register space
//	float Areg;
//	float Breg[WPTN];
//	float acc[WPTM][WPTN];
//
//	// Initialise the accumulation registers
//	for (int wm=0; wm<WPTM; wm++) {
//		for (int wn=0; wn<WPTN; wn++) {
//			acc[wm][wn] = 0.0f;
//		}
//	}
//	
//	// Loop over all tiles
//	int numTiles = K/TSK;
//	for (int t=0; t<numTiles; t++) {
//
//		// Load one tile of A and B into local memory
//		for (int la=0; la<LPTA/WIDTH; la++) {
//			int tid = tidn*RTSM + tidm;
//			int id = la*RTSN*RTSM + tid;
//			int row = id % (TSM/WIDTH);
//			int col = id / (TSM/WIDTH);
//
//			// Load the values (wide vector load)
//			int tiledIndex = TSK*t + col;
//			floatX vecA = A[tiledIndex*(M/WIDTH) + offsetM/WIDTH + row];
//			floatX vecB = B[tiledIndex*(N/WIDTH) + offsetN/WIDTH + row];
//
//			// Store the loaded vectors into local memory
//			#if WIDTH == 1
//				Asub[col][row] = vecA;
//				Asub[col][row] = vecA;
//			#elif WIDTH == 2
//				Asub[col][WIDTH*row + 0] = vecA.x;
//				Asub[col][WIDTH*row + 1] = vecA.y;
//			#elif WIDTH == 4
//				Asub[col][WIDTH*row + 0] = vecA.x;
//				Asub[col][WIDTH*row + 1] = vecA.y;
//				Asub[col][WIDTH*row + 2] = vecA.z;
//				Asub[col][WIDTH*row + 3] = vecA.w;
//			#endif
//			#if WIDTH == 1
//				Bsub[col][row] = vecB;
//				Bsub[col][row] = vecB;
//			#elif WIDTH == 2
//				Bsub[col][WIDTH*row + 0] = vecB.x;
//				Bsub[col][WIDTH*row + 1] = vecB.y;
//			#elif WIDTH == 4
//				Bsub[col][WIDTH*row + 0] = vecB.x;
//				Bsub[col][WIDTH*row + 1] = vecB.y;
//				Bsub[col][WIDTH*row + 2] = vecB.z;
//				Bsub[col][WIDTH*row + 3] = vecB.w;
//			#endif
//		}
//		
//		// Synchronise to make sure the tile is loaded
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		// Loop over the values of a single tile
//		for (int k=0; k<TSK; k++) {
//
//			// Cache the values of Bsub in registers
//			for (int wn=0; wn<WPTN; wn++) {
//				int col = tidn + wn*RTSN;
//				Breg[wn] = Bsub[k][col];
//			}
//
//			// Perform the computation
//			for (int wm=0; wm<WPTM; wm++) {
//				int row = tidm + wm*RTSM;
//				Areg = Asub[k][row];
//				for (int wn=0; wn<WPTN; wn++) {
//					acc[wm][wn] += Areg * Breg[wn];
//				}
//			}
//		}
//
//		// Synchronise before loading the next tile
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//
//	// Store the final results in C
//	for (int wm=0; wm<WPTM; wm++) {
//		int globalRow = offsetM + tidm + wm*RTSM;
//		for (int wn=0; wn<WPTN; wn++) {
//			int globalCol = offsetN + tidn + wn*RTSN;
//			C[globalCol*M + globalRow] = acc[wm][wn];
//		}
//	}
//}

//#define TS 32
//kernel void gemm_version3(global float* out, const global float* in, const global float* weight, const global float* bias, 
//		/*local float* tmp, */const int dim_hidden, const int dim_in)
//{
//	const int row = get_local_id(0); // Local row ID (max: TS)
//	const int col = get_local_id(1); // Local col ID (max: TS)
//	const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..dim_hidden)
//	const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
//
//	// Local memory to fit a tile of TS*TS elements of A and B
//	local float Asub[TS][TS];
//	local float Bsub[TS][TS];
//
//	// Initialise the accumulation register
//	float acc = 0.0f;
//
//	// Loop over all tiles
//	const int numTiles = dim_in / TS;
//	for (int t=0; t<numTiles; t++) {
//		// Load one tile of A and B into local memory
//		const int tiledRow = TS*t + row;
//		const int tiledCol = TS*t + col;
//		Asub[col][row] = weight[tiledCol * dim_hidden + globalRow];
//		Bsub[col][row] = in[globalCol * dim_in + tiledRow];
//
//		// Synchronise to make sure the tile is loaded
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		// Perform the computation for a single tile
//		for (int k=0; k<TS; k++) {
//			acc += Asub[k][row] * Bsub[col][k];
//		}
//
//		// Synchronise before loading the next tile
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}
//
//	// Store the final result in C
//	out[globalCol * dim_hidden + globalRow] = acc;
//}

//kernel void gemm_version3(global float* out, const global float* in, const global float* weight, const global float* bias, 
//		/*local float* tmp, */const int dim_hidden, const int dim_in)
//{
//	const int GID = get_global_id(0);
//	const int n = GID / dim_hidden;
//	const int hidden = GID % dim_hidden;
//	const int weight_offset = hidden * dim_in;
//	
//	const int in_offset = n * dim_in;
//	float z = bias != NULL? bias[hidden] : 0;
//
////	local float weight_tile[dim_in];
//	local float in_tile[2048];
//	for (int i = get_local_id(0); i < dim_in; i += get_local_size(0)) {
////		weight_tile[i] = weight[weight_offset + i];
//		in_tile[i] = in[in_offset + i];
//	}
//	barrier(CLK_LOCAL_MEM_FENCE);
////	const global float* in_tile = in + in_offset;
//	
//	for (int i = 0; i < dim_in; i++)
//		z += weight[dim_hidden * i + hidden] * in_tile[i];
//	out[GID] = z;
//}

//kernel void gemm_tiling_dim_in(global float* out, const global float* in, const global float* weight, const global float* bias,
//		/*local float* tmp, */const int dim_hidden)
//{
//	const int GID = get_global_id(0);
//	const int N = get_global_size(0) / dim_hidden;
//	const int n = GID / dim_hidden;
//	const int hidden = GID % dim_hidden;
//	const int weight_offset = hidden * dim_in;
//	const int in_offset = n * dim_in;
//	float z = bias != NULL? bias[hidden] : 0;
//
//	local float weight_tile[dim_in];
//	local float in_tile[dim_in];
//	for (int i = get_local_id(0); i < dim_in; i += get_local_size(0)) {
//		weight_tile[i] = weight[weight_offset + i];
//		in_tile[i] = in[in_offset + i];
//	}
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//#pragma unroll
//	for (int i = 0; i < dim_in; i++)
//		z += weight_tile[i] * in_tile[i];
//	out[GID] = z;
//}
