#include <python.h>

extern "C"
{
	__declspec(dllexport) void imonesrank(int *dim, int *mask, int *maskdim, int *result);

	__declspec(dllexport) void threshold_contrast(unsigned short *img, int *dim, int *grid, int *thre, unsigned char *output);

	__declspec(dllexport) void imrank(unsigned short *imp,unsigned short *oop,int *inds,int indlen, int *rankinds, int ranklen, unsigned short *temp_out);

	__declspec(dllexport) void local_average(float *img, int*dim, int *aveinds, int avelen, int *candidates, int indlen, float *out);

	__declspec(dllexport) void local_max(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out);

	__declspec(dllexport) void local_min(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out);
}