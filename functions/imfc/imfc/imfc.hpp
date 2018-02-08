#if defined _WIN32
	#define DLL_PUBLIC __declspec(dllexport)
#else
	#define DLL_PUBLIC __attribute__ ((visibility ("default")))
#endif

#include <Python.h>

extern "C"
{
	DLL_PUBLIC void imonesrank(int *dim, int *mask, int *maskdim, int *result);

	DLL_PUBLIC void threshold_contrast(unsigned short *img, int *dim, int *grid, int *thre, unsigned char *output);

	DLL_PUBLIC void imrank(unsigned short *imp,unsigned short *oop,int *inds,int indlen, int *rankinds, int ranklen, unsigned short *temp_out);

	DLL_PUBLIC void local_average(float *img, int*dim, int *aveinds, int avelen, int *candidates, int indlen, float *out);

	DLL_PUBLIC void local_max(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out);

	DLL_PUBLIC void local_min(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out);
}
