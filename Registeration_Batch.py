
import Volt_Registration as reg
import Volt_Imfunctions as im
import time
import glob

clear_all()

pathname = r"D:\Takashi\SPIM\Maarten\test"
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
        dist=im.gaussian(center,d1,d2)
        dist=dist*(hist.max()/dist.max())
        plt.figure(1)
        #plt.subplot(121).imshow(imarray.mean(axis=0))
        plt.subplot(121).imshow(error)
        plt.subplot(122).bar(center,hist,align='center', width=width)
        #plt.subplot(122).plot(center,dist)
        plt.savefig(outdir+"\\hotpix_before.png")
        
        noisy_pix=np.where(error**2>(d1+d2*2))
        imarray = im.remove_hot_pix(imarray, noisy_pix)
        np.save(outdir+"\\noisy_pix",noisy_pix);
        
        
        
        sample = imarray[stack_siz[0]-100:,:,:].squeeze()
        error = sample.std(axis=0)/sample.mean(axis=0)
        hist , bins=np.histogram(error.flatten()**2, range=(0,0.01), bins=30)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        d1=(error.flatten()**2).mean()
        d2=(error.flatten()**2).std()
        dist=im.gaussian(center,d1,d2)
        dist=dist*(hist.max()/dist.max())
        plt.figure(2)
        plt.subplot(121).imshow(error)
        plt.subplot(122).bar(center,hist,align='center', width=width)
        #plt.subplot(122).plot(center,dist)
        plt.savefig(outdir+"\\hotpix_after.png")
        
        
        im_ave=imarray.mean(axis=0).astype('float32')
        
        
        
        t = time.time()
        (imarray,shiftlist) = reg.register_multiple_images_subpix_cuda(imarray.astype('uint16'),im_ave)
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