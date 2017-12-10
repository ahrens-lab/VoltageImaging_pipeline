
import numpy as np
import fast_ffts
import multiprocessing as mp
import time
import matplotlib.pyplot as plt




def register_multiple_images_subpix_cuda(stack, template):
    
    
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import pycuda.cumath as cumath
    import skcuda.fft as cu_fft
    import skcuda.linalg as lin
    import skcuda.cublas as cub
    from numpy import pi,newaxis,floor
    import cmath
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.compiler import SourceModule
    
    from numpy import conj,abs,arctan2,sqrt,real,imag,shape,zeros,trunc,ceil,floor,fix
    from numpy.fft import fftshift,ifftshift
    fft2,ifft2 = fftn,ifftn = fast_ffts.get_ffts(nthreads=1, use_numpy_fft=False)
    
    
    mod = SourceModule("""
   #include <pycuda-complex.hpp>"
   
    __global__ void load_convert(unsigned short *a, float *b,int f, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        int offset = f * imlen;
        if (idx <imlen)
        {
            b[idx] = (float)a[offset+idx];
        }
    }
        
    __global__ void convert_export(float *a, unsigned short *b,int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            b[idx] = (unsigned short)(a[idx]>0 ? a[idx] : 0) ;
        }
    }
        
    __global__ void multiply_comp_float(pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *z, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            z[idx] = x[idx] * y[idx];
        }
    }
        
    __global__ void calc_conj(pycuda::complex<float> *x, pycuda::complex<float> *y, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            y[idx]._M_re = x[idx]._M_re;
            y[idx]._M_im = -x[idx]._M_im;
        }
    }
        
        
    __global__ void convert_multiply(float *x, pycuda::complex<float> *y, float sx, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            y[idx]._M_re = 0;
            y[idx]._M_im = x[idx] * sx;
        }
    }
        
    __global__ void transfer_array(pycuda::complex<float> *x, pycuda::complex<float> *y, int imlenl, int imlen,  int nlargeh, int nh)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        int offset = imlenl*3/4;
        if (idx<imlen)
        {
            int target_ind = (offset+(idx/nh)*nlargeh + (idx % nh))%imlenl;
            x[target_ind] = y[idx];
        }      
    
    }    
        
    __global__ void calc_shiftmatrix(float *x, float *y, pycuda::complex<float> *z, float sx, float sy,float dg, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            z[idx]._M_re = 0;
            z[idx]._M_im = x[idx] * sx + y[idx] * sy + dg;
        }
    }
        
    __global__ void sub_float(float *x, float *y, float sv,  int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            x[idx] = y[idx]-sv;
        }
    }
        

    """)
    
    load_convert_kernel = mod.get_function('load_convert')
    convert_export_kernel = mod.get_function('convert_export')
    convert_multiply_kernel = mod.get_function('convert_multiply')
    multiply_float_kernel = mod.get_function('multiply_comp_float')
    transfer_array_kernel=mod.get_function('transfer_array')
    calc_shiftmatrix_kernel=mod.get_function('calc_shiftmatrix')
    conj_kernel=mod.get_function('calc_conj')
    sub_float_kernel=mod.get_function('sub_float')
    
    
    Z=stack.shape[0]
    M=stack.shape[1]
    N=stack.shape[2]
    max_memsize=4200000000
    
    imlen=M*N
    half_imlen=M*(N//2+1)
    grid_dim=(64,int(imlen/(512*64))+1,1)
    block_dim=(512,1,1) #512 threads per block
    
    
    stack_bin=int(max_memsize/(M*N*stack.itemsize))
    stack_ite=int(Z/stack_bin)+1
    
    
    usfac=100 ## needs to be bigger than 10
    
    if not template.shape == stack.shape[1:]:
        raise ValueError("Images must have same shape.")

   

    if np.any(np.isnan(template)):
        template = template.copy()
        template[template!=template] = 0
    if np.any(np.isnan(stack)):
        stack = stack.copy()
        stack[stack!=stack] = 0
        
        
    mlarge=M*2;
    nlarge=N*2;
  


    t = time.time()

    plan_forward = cu_fft.Plan((M, N), np.float32, np.complex64)
    plan_inverse = cu_fft.Plan((M, N), np.complex64, np.float32)
    plan_inverse_big = cu_fft.Plan((mlarge, nlarge), np.complex64, np.float32)
    cub_h=cub.cublasCreate()

    template_gpu=gpuarray.to_gpu(template.astype('float32'))
    source_gpu=gpuarray.empty((M, N), np.float32)
    ifft_gpu=gpuarray.empty((M, N), np.float32)
    result_gpu=gpuarray.empty((M, N), np.uint16)
    
    
    templatef_gpu=gpuarray.empty((M, N//2+1), np.complex64)
    sourcef_gpu=gpuarray.empty((M, N//2+1), np.complex64)
    prod_gpu1=gpuarray.empty((M, N//2+1), np.complex64)
    prod_gpu2=gpuarray.empty((M, N//2+1), np.complex64)
    shiftmatrix=gpuarray.empty((M, N//2+1), np.complex64)
    
    cu_fft.fft(template_gpu,templatef_gpu,plan_forward, scale=True)
    templatef_gpu=templatef_gpu.conj()
    
    move_list=np.zeros((Z,2))
    
    
    
    largearray1_gpu=gpuarray.zeros((mlarge, nlarge//2+1), np.complex64)
    largearray2_gpu=gpuarray.empty((mlarge, nlarge), np.float32)
    imlenl=mlarge*(nlarge//2+1)
    
    
    zoom_factor=1.5
    dftshiftG = trunc(ceil(usfac*zoom_factor)/2); #% Center of output array at dftshift+1
    upsample_dim=int(ceil(usfac*zoom_factor))
    
    term1c = ( ifftshift(np.arange(N,dtype='float') - floor(N/2)).T[:,newaxis] )/N # fftfreq  # output points
    term2c = (( np.arange(upsample_dim,dtype='float') )/usfac)[newaxis,:]   
    term1r = ( np.arange(upsample_dim,dtype='float').T )[:,newaxis]        
    term2r = ( ifftshift(np.arange(M,dtype='float')) - floor(M/2) )[newaxis,:] # fftfreq
    term1c_gpu=gpuarray.to_gpu(term1c[:int(floor(N/2)+1),:].astype('float32'))
    term2c_gpu=gpuarray.to_gpu(term2c.astype('float32'))
    term1r_gpu=gpuarray.to_gpu(term1r.astype('float32'))
    term2r_gpu=gpuarray.to_gpu(term2r.astype('float32'))
    term2c_gpu_ori=gpuarray.to_gpu(term2c.astype('float32'))
    term1r_gpu_ori=gpuarray.to_gpu(term1r.astype('float32'))
    
    kernc_gpu=gpuarray.zeros((N//2+1,upsample_dim),np.float32)
    kernr_gpu=gpuarray.zeros((upsample_dim,M),np.float32)
    kernc_gpuc=gpuarray.zeros((N//2+1,upsample_dim),np.complex64)
    kernr_gpuc=gpuarray.zeros((upsample_dim,M),np.complex64)
    
    
    Nr = np.fft.ifftshift(np.linspace(-np.fix(M/2),np.ceil(M/2)-1,M))
    Nc = np.fft.ifftshift(np.linspace(-np.fix(N/2),np.ceil(N/2)-1,N))
    [Nc,Nr] = np.meshgrid(Nc,Nr);
    Nc_gpu=gpuarray.to_gpu((Nc[:,:N//2+1]/N).astype('float32'))
    Nr_gpu=gpuarray.to_gpu((Nr[:,:N//2+1]/M).astype('float32'))


    upsampled1=gpuarray.empty((upsample_dim,N//2+1),np.complex64)
    upsampled2=gpuarray.empty((upsample_dim,upsample_dim),np.complex64)
    
    source_stack=gpuarray.empty((stack_bin,M,N),dtype=stack.dtype)
    copy=drv.Memcpy3D()
    copy.set_src_host(stack.data)
    copy.set_dst_device(source_stack.gpudata)
    copy.width_in_bytes = copy.src_pitch = stack.strides[1]
    copy.src_height = copy.height = M
    
    
    for zb in range(stack_ite):
        
        zrange=np.arange(zb*stack_bin,min((stack_bin*(zb+1)),Z))
        copy.depth=len(zrange)
        copy.src_z=int(zrange[0])
        copy()
        
        for i in range(len(zrange)):
            
            t=zb*stack_bin+i
            load_convert_kernel(source_stack, source_gpu.gpudata, np.int32(i), np.int32(imlen), block=block_dim, grid=grid_dim)
            cu_fft.fft(source_gpu,sourcef_gpu,plan_forward, scale=True)
            
            multiply_float_kernel(sourcef_gpu,templatef_gpu,prod_gpu1,np.int32(half_imlen), block=block_dim, grid=grid_dim)
            transfer_array_kernel(largearray1_gpu,prod_gpu1, np.int32(imlenl),np.int32(half_imlen),np.int32(nlarge//2+1),np.int32(N//2+1),block=block_dim, grid=grid_dim)
            cu_fft.ifft(largearray1_gpu,largearray2_gpu,plan_inverse_big, scale=True)
            peakind=cub.cublasIsamax(cub_h,largearray2_gpu.size, largearray2_gpu.gpudata,1)
            rloc,cloc = np.unravel_index(peakind,largearray2_gpu.shape)    
            
            md2 = trunc(mlarge/2); nd2 = trunc(nlarge/2);
            if rloc > md2 :
                row_shift2 = rloc - mlarge;
            else:
                row_shift2 = rloc;
            if cloc > nd2:
                col_shift2 = cloc - nlarge;
            else:
                col_shift2 = cloc;
            row_shiftG=row_shift2/2.;
            col_shiftG=col_shift2/2.;
    
            # Initial shift estimate in upsampled grid
            
            row_shiftG0 = round(row_shiftG*usfac)/usfac; 
            col_shiftG0 = round(col_shiftG*usfac)/usfac;     
            # Matrix multiply DFT around the current shift estimate
            roffG = dftshiftG-row_shiftG0*usfac
            coffG = dftshiftG-col_shiftG0*usfac
            
            sub_float_kernel(term2c_gpu,term2c_gpu_ori,np.float32(coffG/usfac), np.int32(term2c_gpu.size), block=block_dim, grid=grid_dim)
            sub_float_kernel(term1r_gpu,term1r_gpu_ori,np.float32(roffG), np.int32(term1r_gpu.size), block=block_dim, grid=grid_dim)
            
            lin.dot(term1c_gpu,term2c_gpu,handle=cub_h,out=kernc_gpu)
            lin.dot(term1r_gpu,term2r_gpu,handle=cub_h,out=kernr_gpu)
            convert_multiply_kernel(kernc_gpu,kernc_gpuc,np.float32(-2*pi),np.int32(kernc_gpu.size), block=block_dim, grid=grid_dim)
            convert_multiply_kernel(kernr_gpu,kernr_gpuc,np.float32(-2*pi/(M*usfac)),np.int32(kernr_gpu.size), block=block_dim, grid=grid_dim)
            cumath.exp(kernc_gpuc, out=kernc_gpuc)
            cumath.exp(kernr_gpuc, out=kernr_gpuc)
            
            conj_kernel(prod_gpu1,prod_gpu2,np.int32(half_imlen), block=block_dim, grid=grid_dim)
            
            lin.dot(kernr_gpuc,prod_gpu2,handle=cub_h,out=upsampled1)
            lin.dot(upsampled1,kernc_gpuc,handle=cub_h,out=upsampled2)
    
            
            CCG = conj(upsampled2.get())/(md2*nd2*usfac**2);
            rlocG,clocG = np.unravel_index(abs(CCG).argmax(), CCG.shape) 
            CCGmax = CCG[rlocG,clocG]
        
    
            
            rlocG = rlocG - dftshiftG #+ 1 # +1 # questionable/failed hack + 1;
            clocG = clocG - dftshiftG #+ 1 # -1 # questionable/failed hack - 1;
            row_shiftG = row_shiftG0 + rlocG/usfac;
            col_shiftG = col_shiftG0 + clocG/usfac;    
    
            diffphaseG=arctan2(imag(CCGmax),real(CCGmax));
            
            # Compute registered version of source stack
            calc_shiftmatrix_kernel(Nr_gpu, Nc_gpu, shiftmatrix,np.float32(row_shiftG*2*np.pi), np.float32(col_shiftG*2*np.pi),np.float32(diffphaseG),np.int32(half_imlen), block=block_dim, grid=grid_dim)
            cumath.exp(shiftmatrix, out=shiftmatrix);
            multiply_float_kernel(sourcef_gpu,shiftmatrix,prod_gpu1,np.int32(half_imlen), block=block_dim, grid=grid_dim)
            cu_fft.ifft(prod_gpu1,ifft_gpu,plan_inverse)
            convert_export_kernel(ifft_gpu,result_gpu,np.int32(imlen), block=block_dim, grid=grid_dim)
   
            move_list[t,:]=( row_shiftG, col_shiftG)
            stack[t,:,:] = result_gpu.get()
								 
    
    cub.cublasDestroy(cub_h)
    return (stack,move_list)






