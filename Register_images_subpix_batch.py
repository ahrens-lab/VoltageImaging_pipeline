
import imreg
import time
import glob

clear_all()

pathname = r"D:\Takashi\SPIM\12072017"
precision=100;
frequency=300

subdir=glob.glob(pathname+"\\Fish*\\")

for d in range(len(subdir)):
    pathname=subdir[d]
    plt.close("all")

    print(pathname)
    outdir = pathname+"\\Registered"
    oname1 = 'raw.tif'
    oname2 = 'registered.tif'
    oname3 = 'ave.tif'
    
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    
    def remove_hot_pix(imarray, inds):
        
        tmp=np.ones((3,3))
        tmp[1,1]=0
        move_inds=np.where(tmp)
        move_inds[0][:]=move_inds[0][:]-1
        move_inds[1][:]=move_inds[1][:]-1
    
        dim=imarray.shape
        keep_ind=np.where((inds[0]>0) & (inds[0]<(dim[1]-1)) & (inds[1]>0) & (inds[1]<(dim[2]-1)))
        ind_ys=inds[0][keep_ind]
        ind_xs=inds[1][keep_ind]
        numpix=len(ind_ys)
        for i in range(numpix):
            ind_y=ind_ys[i]
            ind_x=ind_xs[i]
            tcourse=imarray[:,ind_y+move_inds[0],ind_x+move_inds[1]]
            imarray[:,ind_y,ind_x]=tcourse.mean(axis=1).astype(imarray.dtype)
        print(str(numpix),' hot pixels removed')
        return imarray
    
    
    if (not os.path.isdir(outdir)): 
        os.mkdir(outdir)
        
    list = glob.glob(pathname+'\\TM*')
    
    if (os.path.isfile(outdir+"\\"+oname2) and list):
    
        for f in range(len(list)):
            os.remove(list[f])
    
    if ((not os.path.isfile(outdir+"\\"+oname2)) and list):
        
        siz = np.fromfile(pathname+'\\Stack dimensions.log',dtype='uint32')
        siz[2]=50
        stack_siz = np.copy(siz)
        
        #%%
        stack_siz[2] = len(list)*siz[2]
        stack_siz = np.flipud(stack_siz)
        imarray = np.zeros(stack_siz,'uint16')
        
        if (list[0].endswith("tif")):
            zl=0;
            for f in list:
                stack = imread(f)
                tmp = stack.shape
                step = 1 if(len(tmp)==2)else tmp[0]
                imarray[zl:zl+step,:,:]=stack.astype('uint16')
                zl=zl+step
        elif (list[0].endswith("stack")):
            zl=0;
            for f in list:
                stack = (np.fromfile(f,dtype='uint16')).reshape(np.flipud(siz))
                tmp = stack.shape
                step = 1 if(len(tmp)==2)else tmp[0]
                imarray[zl:zl+step,:,:]=stack.astype('uint16')
                zl=zl+step
        print(stack_siz[0],' frames loaded')
        
        imsave(outdir+"\\"+oname1,imarray.astype('uint16'))
        sample = imarray[stack_siz[0]-100:,:,:].squeeze()
        error = sample.std(axis=0)/sample.mean(axis=0)
        hist , bins=np.histogram(error.flatten()**2, range=(0,0.01), bins=30)
        
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        d1=(error.flatten()**2).mean()
        d2=(error.flatten()**2).std()
        dist=gaussian(center,d1,d2)
        dist=dist*(hist.max()/dist.max())
        plt.figure(1)
        #plt.subplot(121).imshow(imarray.mean(axis=0))
        plt.subplot(121).imshow(error)
        plt.subplot(122).bar(center,hist,align='center', width=width)
        #plt.subplot(122).plot(center,dist)
        plt.savefig(outdir+"\\hotpix_before.png")
        
        noisy_pix=np.where(error**2>(d1+d2*2))
        imarray = remove_hot_pix(imarray, noisy_pix)
        np.save(outdir+"\\noisy_pix",noisy_pix);
        
        
        
        sample = imarray[stack_siz[0]-100:,:,:].squeeze()
        error = sample.std(axis=0)/sample.mean(axis=0)
        hist , bins=np.histogram(error.flatten()**2, range=(0,0.01), bins=30)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        d1=(error.flatten()**2).mean()
        d2=(error.flatten()**2).std()
        dist=gaussian(center,d1,d2)
        dist=dist*(hist.max()/dist.max())
        plt.figure(2)
        plt.subplot(121).imshow(error)
        plt.subplot(122).bar(center,hist,align='center', width=width)
        #plt.subplot(122).plot(center,dist)
        plt.savefig(outdir+"\\hotpix_after.png")
        
        
        im_ave=imarray.mean(axis=0).astype('float32')
        
        
        
        t = time.time()
        (imarray,shiftlist) = imreg.register_multiple_images_subpix_cuda(imarray.astype('uint16'),im_ave)
        print(time.time()-t)
        
        im_ave2=imarray.mean(axis=0).astype('float32')
        #%%%
        tlen=imarray.shape[0]
        
        f2=plt.figure(3,figsize=(16,5))
        plt.plot(np.linspace(1,tlen,tlen)/frequency,shiftlist[:,0]*406)
        plt.plot(np.linspace(1,tlen,tlen)/frequency,shiftlist[:,1]*406)
        plt.title(str((shiftlist[0,:]*406).std()))
        plt.ylim(-400,400)
        plt.savefig(outdir+"\\shift_graph.png")
        np.save(outdir+"\\shiftlist",shiftlist);
        
        imsave(outdir+"\\"+oname2,imarray.astype('uint16'))
        imsave(outdir+"\\"+oname3,im_ave2.squeeze().astype('uint16'))

        if (os.path.isfile(outdir+"\\"+oname2) and list):
            
            for f in range(len(list)):
                os.remove(list[f])