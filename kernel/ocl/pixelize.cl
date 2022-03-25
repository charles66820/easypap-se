#include "kernel/ocl/common.cl"

#ifdef PARAM
#define PIX_BLOC PARAM
#else
#define PIX_BLOC 16
#endif

// In this over-simplified kernel, all the pixels of a bloc adopt the color
// on the top-left pixel (i.e. we do not compute the average color).
__kernel void pixelize_ocl (__global unsigned *in)
{
  __local int4 tile[GPU_TILE_H][GPU_TILE_W];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile[yloc][xloc] = color_to_int4(in[y * DIM + x]);

  for (int d = PIX_BLOC >> 1; d > 0; d >>= 1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (xloc % PIX_BLOC < d)
      tile[yloc][xloc] += tile[yloc][xloc + d];

    barrier(CLK_LOCAL_MEM_FENCE);
    if (yloc % PIX_BLOC < d)
      tile[yloc][xloc] += tile[yloc + d][xloc];
  }

  barrier (CLK_LOCAL_MEM_FENCE);

  in[y * DIM + x] = int4_to_color(tile[yloc - (yloc % PIX_BLOC)][xloc - (xloc % PIX_BLOC)] / (int4) (PIX_BLOC * PIX_BLOC));
}
