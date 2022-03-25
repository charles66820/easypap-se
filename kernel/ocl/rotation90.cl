#include "kernel/ocl/common.cl"


__kernel void rotation90_ocl (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[GPU_TILE_H][GPU_TILE_W];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xLoc = get_local_id(0);
  int yLoc = get_local_id(1);

  tile[GPU_TILE_W - xLoc - 1][yLoc] = in[y * DIM + x];

  barrier(CLK_LOCAL_MEM_FENCE);

  out[(DIM - x + xLoc - GPU_TILE_W) * DIM + y - yLoc + xLoc + yLoc * DIM] = tile[yLoc][xLoc];
}

