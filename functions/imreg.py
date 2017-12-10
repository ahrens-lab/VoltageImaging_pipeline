
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


def stackRegister_simple_cuda(stack, template):
    
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as drv
    import skcuda.fft as cu_fft
    import skcuda.linalg as lin
    import skcuda.cublas as cub
    from pycuda.elementwise import ElementwiseKernel
    from pycuda.compiler import SourceModule
    

    
    mod = SourceModule("""
   #include <pycuda-complex.hpp>"
   
    __global__ void load_convert(unsigned short *a, double *b,int f, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        int offset = f * imlen;
        if (idx <imlen)
        {
            b[idx] = (double)a[offset+idx];
        }
    }
        
    __global__ void multiply_comp(pycuda::complex<double> *x, pycuda::complex<double> *y, pycuda::complex<double> *z, int imlen)
    {
        int idx = (int) gridDim.x*blockDim.x*blockIdx.y+blockIdx.x *  blockDim.x +  threadIdx.x ;
        if (idx <imlen)
        {
            z[idx] = x[idx]*y[idx];
        }
    }
    """)
    
    load_convert_kernel = mod.get_function('load_convert')
    multiply_kernel = mod.get_function('multiply_comp')
    
    
    block_dim=(512,1,1) #512 threads per block
    
    
    Z=stack.shape[0]
    M=stack.shape[1]
    N=stack.shape[2]
    max_memsize=4200000000
    imlen=M*N
    grid_dim=(64,int(imlen/(512*64))+1,1)
    
    
    
    stack_bin=int(max_memsize/(M*N*stack.itemsize))
    stack_ite=int(Z/stack_bin)+1
    
    #print(Z)
    #print(stack_bin)
    #print(stack_ite)

    plan_forward = cu_fft.Plan((M, N), np.float64, np.complex128)
    plan_inverse = cu_fft.Plan((M, N), np.complex128, np.float64)
    #plan_inverse = cu_fft.Plan((M, N), np.complex128, np.complex128)
    
    
    y = np.empty((Z, M, N), dtype=stack.dtype)
    
    template_gpu=gpuarray.to_gpu(template.astype('float64'))
    templatef_gpu=gpuarray.empty((M, N//2+1), np.complex128)
    cu_fft.fft(template_gpu,templatef_gpu,plan_forward)
    templatef_gpu=templatef_gpu.conj()
    
    inv_gpu = gpuarray.empty_like(template_gpu)
    #inv_gpu = gpuarray.empty((M, N), np.complex128)
    prod_gpu = gpuarray.empty_like(templatef_gpu)
    xf_gpu = gpuarray.empty_like(templatef_gpu)
    xz_gpu = gpuarray.empty_like(template_gpu)
    
    shiftlist=np.zeros((Z,2))
    h=cub.cublasCreate()
    x_gpu=gpuarray.empty((stack_bin,M,N),dtype=stack.dtype)
    copy=drv.Memcpy3D()
    copy.set_src_host(stack.data)
    copy.set_dst_device(x_gpu.gpudata)
    copy.width_in_bytes = copy.src_pitch = stack.strides[1]
    copy.src_height = copy.height = M
    
    for zb in range(stack_ite):
        
        zrange=np.arange(zb*stack_bin,min((stack_bin*(zb+1)),Z))
        copy.depth=len(zrange)
        copy.src_z=int(zrange[0])
        copy()
        #x_gpu[:len(zrange),:,:].set(stack[zb*stack_bin:min((stack_bin*(zb+1)),Z),:,:])
        
        for i in range(len(zrange)):
            
            t=zb*stack_bin+i
            load_convert_kernel(x_gpu, xz_gpu.gpudata, np.int32(i), np.int32(imlen), block=block_dim, grid=grid_dim)
            cu_fft.fft(xz_gpu,xf_gpu,plan_forward)
            multiply_kernel(xf_gpu,templatef_gpu,prod_gpu,np.int32(imlen), block=block_dim, grid=grid_dim)
            cu_fft.ifft(prod_gpu,inv_gpu,plan_inverse)
            peakind=cub.cublasIdamax(h,inv_gpu.size, inv_gpu.gpudata,1)
            
            col_shift = peakind % N
            row_shift = np.floor(peakind / N)
        
            if (row_shift > (M/2)):
                row_shift -=  M
        
            if (col_shift > (N/2)):
                col_shift -=  N
                
            shiftlist[t,:]=[row_shift,col_shift]
            shift = int(row_shift*N+col_shift)
            y[t, :, :] = np.roll(stack[t,:,:],-shift)
        
    cub.cublasDestroy(h)
    
    return y,shiftlist


def register_multiple_images_subpix(im2,im1):
    
    usfac=100 ## needs to be bigger than 10
    
    print(__name__)
    if not im1.shape == im2.shape[1:]:
        raise ValueError("Images must have same shape.")

   

    if np.any(np.isnan(im1)):
        im1 = im1.copy()
        im1[im1!=im1] = 0
    if np.any(np.isnan(im2)):
        im2 = im2.copy()
        im2[im2!=im2] = 0
        

    def dftups(inp,nor=None,noc=None,usfac=1,roff=0,coff=0):
     
        from numpy.fft import ifftshift,fftfreq
        from numpy import pi,newaxis,floor
    
        nr,nc=np.shape(inp);
        # Set defaults
        if noc is None: noc=nc;
        if nor is None: nor=nr;
        # Compute kernels and obtain DFT by matrix products
        term1c = ( ifftshift(np.arange(nc,dtype='float') - floor(nc/2)).T[:,newaxis] )/nc # fftfreq
        term2c = (( np.arange(noc,dtype='float') - coff  )/usfac)[newaxis,:]              # output points
        kernc=np.exp((-1j*2*pi)*term1c*term2c);
    
        term1r = ( np.arange(nor,dtype='float').T - roff )[:,newaxis]                # output points
        term2r = ( ifftshift(np.arange(nr,dtype='float')) - floor(nr/2) )[newaxis,:] # fftfreq
        kernr=np.exp((-1j*2*pi/(nr*usfac))*term1r*term2r);
        out=np.dot(np.dot(kernr,inp),kernc);
        return out 
        

    def dftregistration(buf1ft,buf2ft,usfac):
       
    
        # this function is translated from matlab, so I'm just going to pretend
        # it is matlab/pylab
        from numpy import conj,abs,arctan2,sqrt,real,imag,shape,zeros,trunc,ceil,floor,fix
        from numpy.fft import fftshift,ifftshift
        fft2,ifft2 = fftn,ifftn = fast_ffts.get_ffts(nthreads=1, use_numpy_fft=False)
    
    
        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        [m,n]=shape(buf1ft);
        mlarge=m*2;
        nlarge=n*2;
        CClarge=zeros([mlarge,nlarge], dtype='complex');
        #CClarge[m-fix(m/2):m+fix((m-1)/2)+1,n-fix(n/2):n+fix((n-1)/2)+1] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        CClarge[round(mlarge/4.):round(mlarge/4.*3),round(nlarge/4.):round(nlarge/4.*3)] = fftshift(buf1ft) * conj(fftshift(buf2ft));
        # note that matlab uses fix which is trunc... ?
      
        # Compute crosscorrelation and locate the peak 
        CC = ifft2(ifftshift(CClarge)); # Calculate cross-correlation
    
        rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape)
        CCmax=CC[rloc,cloc]; 
 

        
        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak 
        [m,n] = shape(CC); md2 = trunc(m/2); nd2 = trunc(n/2);
        if rloc > md2 :
            row_shift2 = rloc - m;
        else:
            row_shift2 = rloc;
        if cloc > nd2:
            col_shift2 = cloc - n;
        else:
            col_shift2 = cloc;
        row_shift2=row_shift2/2.;
        col_shift2=col_shift2/2.;

        
        # Initial shift estimate in upsampled grid
        zoom_factor=1.5
        row_shift0 = round(row_shift2*usfac)/usfac; 
        col_shift0 = round(col_shift2*usfac)/usfac;     
        dftshift = trunc(ceil(usfac*zoom_factor)/2); #% Center of output array at dftshift+1
        # Matrix multiply DFT around the current shift estimate
        roff = dftshift-row_shift0*usfac
        coff = dftshift-col_shift0*usfac
        upsampled = dftups(
                (buf2ft * conj(buf1ft)),
                ceil(usfac*zoom_factor),
                ceil(usfac*zoom_factor), 
                usfac, 
                roff,
                coff)
        
        CC = conj(upsampled)/(md2*nd2*usfac**2);
                       # Locate maximum and map back to original pixel grid 
        rloc,cloc = np.unravel_index(abs(CC).argmax(), CC.shape) 
        rloc0,cloc0 = np.unravel_index(abs(CC).argmax(), CC.shape) 
        CCmax = CC[rloc,cloc]
    
        rg00 = dftups(buf1ft * conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2);
        rf00 = dftups(buf2ft * conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2);  
        #if DEBUG: print rloc,row_shift,cloc,col_shift,dftshift
        rloc = rloc - dftshift #+ 1 # +1 # questionable/failed hack + 1;
        cloc = cloc - dftshift #+ 1 # -1 # questionable/failed hack - 1;
        row_shift = row_shift0 + rloc/usfac;
        col_shift = col_shift0 + cloc/usfac;    


        error = 1.0 - CCmax * conj(CCmax)/(rg00*rf00);
        error = sqrt(abs(error));
        diffphase=arctan2(imag(CCmax),real(CCmax));
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0;
        if nd2 == 1:
            col_shift = 0;
        #output=[error,diffphase,row_shift,col_shift];
        output=[row_shift,col_shift, diffphase]
    
    
        # Compute registered version of buf2ft
        nr,nc=shape(buf2ft);
        Nr = np.fft.ifftshift(np.linspace(-np.fix(nr/2),np.ceil(nr/2)-1,nr))
        Nc = np.fft.ifftshift(np.linspace(-np.fix(nc/2),np.ceil(nc/2)-1,nc))
        [Nc,Nr] = np.meshgrid(Nc,Nr);
        Greg = buf2ft * np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
        Greg = Greg*np.exp(1j*diffphase);
        output.append(Greg)
    
        return output



	


    fft2,ifft2 = fftn,ifftn = fast_ffts.get_ffts(nthreads=1, use_numpy_fft=False)
    
    t = time.time()

    siz=im2.shape
    im1fft = fft2(im1.astype('float32'))
    result_im=np.zeros(siz)
    move_list=np.zeros((siz[0],2))
    
    
    
    for z in range(siz[0]):
        tmp=im2[z,:,:].squeeze()
        im2fft = fft2(tmp.astype('float32'))
        if ((z %500) == 0): print(z, 'frames registered')

        output = dftregistration(im1fft,im2fft,usfac)
        #print(output[:-1])
								 
        move_list[z,:]=(-output[0],-output[1])
        result_im[z,:,:] = (np.abs(ifft2(output[-1]))).reshape((1,siz[1],siz[2]))
            

    
    tt=time.time()-t
    return (result_im,move_list,tt)


def stackRegister_simple(stack, template):
    batch_size=stack.shape[0]
    M=stack.shape[1]
    N=stack.shape[2]
    y = np.empty((batch_size, M, N), dtype=stack.dtype)
    template_f=np.conj(np.fft.fft2(template.astype('float64')))
    shiftlist=np.zeros((batch_size,2))
    
    for i in range(batch_size):
        
        if ((i%100)==0):
            print('Registering frame %d'%i)
        tmp = np.fft.fft2(stack[i, :, :].astype('float64'))
        peakind=np.argmax(np.abs(np.fft.ifft2(tmp*template_f)))
        
        col_shift = peakind % N
        row_shift = np.floor(peakind / N)
    
        if (row_shift > (M/2)):
            row_shift -=  M
    
        if (col_shift > (N/2)):
            col_shift -=  N
            
        shiftlist[i,:]=[row_shift,col_shift]
        shift = int(row_shift*N+col_shift)
        y[i, :, :] = np.roll(stack[i,:,:],-shift)
    
    return y,shiftlist




