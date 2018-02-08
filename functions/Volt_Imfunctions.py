
import numpy as np
import math



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



def makeDisk(r):
    
    array=np.zeros((2*r+1,2*r+1))
    ii=0;
    while ii < (2*r+1):
        jj=0;
        while jj < (2*r+1):
            if math.sqrt((ii-r)**2 + (jj-r)**2) <= r:
                array[ii,jj]=1            
            jj=jj+1
        ii=ii+1
    
    return array
          

    
    
def imNormalize(array,fr):
    
    div=100/(100-fr)
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/div)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2    
    
  
def LassoSelection(img):
   
    from matplotlib.widgets import LassoSelector
    from matplotlib import path
    import matplotlib.pyplot as plt
    global nROI,ROI, ROI_color,points
    
    dims=img.shape
    
    nROI=0;
    ROI=np.zeros((dims[0],dims[1]));
    ROI_color=np.zeros((dims[0],dims[1],3));
    points=np.fliplr(np.array(np.nonzero(np.ones((dims[0],dims[1])))).T)
    
    fig = plt.figure(1,figsize=(14,7))
    ax1 = plt.subplot(121)
    f1=ax1.imshow(imNormalize(img,99),cmap=plt.get_cmap('gray'));
    ax2 = plt.subplot(122)
    f2=ax2.imshow(ROI_color,cmap=plt.get_cmap('gray'));
    
    points_poly=[]
    
        
    def onselectf(verts):
        
        global nROI, ROI, ROI_color,points
        nROI+=1
        
        p=path.Path(verts)
        points_poly=np.where(p.contains_points(points, radius=1))[0]
        
        tmp=np.zeros((dims[0],dims[1]))
        tmp[points[points_poly,1],points[points_poly,0]]=1
        tmp[np.where(ROI>0)]=0
        
        ROI=ROI+tmp*nROI
        ROI_color=ROI_color+np.tile(tmp[:,:,None],(1,1,3))
        f2.set_data(ROI_color)        
        fig.canvas.draw()
        print('ROI',nROI,'drawn')
    
    lasso=LassoSelector(ax1,onselect=onselectf)
        
    
    plt.show()
    
    cnum=0
    
    while True:
        select=fig.ginput(1)
        
        if not select:
            break
        else:
            xy=(np.asarray(select)[0]).astype(int)
            if xy.min()>=1 and xy[1]<dims[0] and xy[0]<dims[1]:
                cnum=1
    
   
    
    print('ROI selection done')
    
    return nROI,ROI
    


def imrank_dll(image,r):
    
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    import sys

    if sys.platform == 'darwin':  # Mac
        imfc = ctypes.cdll.LoadLibrary(
            os.getcwd() + r"/functions/imfc.cpython-36m-darwin.so")    
    elif sys.platform == 'linux':
        imfc = ctypes.cdll.LoadLibrary(
            os.getcwd() + r"/functions/imfc.cpython-36m-x86_64-linux-gnu.so")        
    else:  # Windows
        imfc = ctypes.cdll.LoadLibrary(os.getcwd()+r"\\functions\\imfc.dll")

    func1 = imfc.imrank
    func1.restype = None
    func1.argtypes = [ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ctypes.c_int,
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ctypes.c_int,
                     ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS")]
    
    func2 = imfc.imonesrank
    func2.restype = None
    func2.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    
    image=image.astype('uint16')
    image_pad=np.zeros((image.shape[0]+2*r,image.shape[1]+2*r),'uint16')
    image_pad[r:-r,r:-r]=image
    ones_pad=np.zeros((image.shape[0]+2*r,image.shape[1]+2*r),'uint16')
    ones_pad[r:-r,r:-r]=1
    rankout=np.zeros((image.shape[0],image.shape[1]),'uint16')
    onesout=np.zeros((image.shape[0],image.shape[1]),'int32')
    
    indslist = np.where(ones_pad>0)
    inds=(indslist[0]*image_pad.shape[1]+indslist[1]).astype('int32')
    
    disk=makeDisk(r)
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    rankinds=(r_inds*image_pad.shape[1]+c_inds).astype('int32')
    
    func1(image_pad,ones_pad,inds,int(inds.size),rankinds,int(rankinds.size),rankout)
    func2(np.array(onesout.shape).astype('int32'),disk.astype('int32'),np.array(disk.shape).astype('int32'),onesout)
    
    return rankout.astype('float')/onesout.astype('float')



def local_minima_dll(image,r):
    
    
    import ctypes
    from numpy.ctypeslib import ndpointer
    import os
    import sys

    if sys.platform == 'darwin':  # Mac
        imfc = ctypes.cdll.LoadLibrary(
            os.getcwd() + r"/functions/imfc.cpython-36m-darwin.so")    
    elif sys.platform == 'linux':
        imfc = ctypes.cdll.LoadLibrary(
            os.getcwd() + r"/functions/imfc.cpython-36m-x86_64-linux-gnu.so")        
    else:  # Windows
        imfc = ctypes.cdll.LoadLibrary(os.getcwd()+r"\\functions\\imfc.dll")
    
    func1 = imfc.local_min
    func1.restype = None
    func1.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ctypes.c_int,
                     ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                     ctypes.c_int,
                     ndpointer(ctypes.c_ubyte, flags="C_CONTIGUOUS")]
    
  
    
    image=image.astype('float32')
    image_pad=np.ones((image.shape[0]+2*r,image.shape[1]+2*r),'float32')*1000
    image_pad[r:-r,r:-r]=image
    ones_pad=np.zeros((image.shape[0]+2*r,image.shape[1]+2*r),'float32')
    ones_pad[r:-r,r:-r]=1
    minout=np.zeros((image.shape[0]+2*r,image.shape[1]+2*r),'uint8')
    
    indslist = np.where(ones_pad>0)
    inds=(indslist[0]*image_pad.shape[1]+indslist[1]).astype('int32')
    
    disk=makeDisk(r)   
    disk[r,r]=0
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    mininds=(r_inds*image_pad.shape[1]+c_inds).astype('int32')
    
    func1(image_pad,np.array(image_pad.shape).astype('int32'),mininds,int(mininds.size),inds,int(inds.size),minout)
    
    
    return minout[r:-r,r:-r]


def draw_periphery_single(image,y,x,r):
    
    dangle=np.pi/6
    base_angle=(np.pi*2)/20
    
    v1=image.mean()
    v2=image.std()
    image_pad=(np.random.randn(image.shape[0]+r*2,image.shape[1]+r*2)*np.sqrt(v2))+v1

    image_pad[r:-r,r:-r]=image
    image_pad_ones=np.zeros(image_pad.shape)
    image_pad_ones[r:-r,r:-r]=1
    
    def angular_map(maxradius):

        mask=makeDisk(maxradius)
        mask[maxradius,maxradius]=0
        (disk_r, disk_c)=np.where(mask==1)
        disk_r=disk_r-maxradius
        disk_c=disk_c-maxradius
        
        
        #return disk_r
        pixel_num=np.zeros((maxradius,));
        disk_d=np.zeros((disk_r.size,));
        disk_th=np.zeros((disk_r.size,));
        
        for a1 in range(len(disk_r)):
               dist=int(np.sqrt(disk_r[a1]**2+disk_c[a1]**2));
               pixel_num[dist-1] +=1;
               disk_d[a1]=dist;
               disk_th[a1]=np.angle(disk_r[a1]*1j+disk_c[a1])
        return (disk_r,disk_c,disk_d,disk_th)

    angle_map=angular_map(int(r))
    
    line=np.where(makeDisk(1)>0)
    
    distances1=np.zeros((80,))
    
    for a in range(80):
        base_angle=(np.pi/40)*a-np.pi
        search_inds=np.where((angle_map[3]>(base_angle-dangle/2)) & (angle_map[3]<(base_angle+dangle/2)))
        angular_inds=(angle_map[0][search_inds],angle_map[1][search_inds])
        vlist1=image_pad[y+angular_inds[0]+r,x+angular_inds[1]+r]
        dlist=angle_map[2][search_inds]
        new_x=np.arange(1,int(r),0.5)
        sortinds=np.argsort(dlist)
        vlist2=np.interp(new_x,dlist[sortinds],vlist1[sortinds])
        distances1[a]=new_x[vlist2.argmax()]+1
    
    distances_pad=np.zeros((90,))
    distances_pad[5:85]=distances1
    distances_pad[:5]=distances1[-5:]
    distances_pad[-5:]=distances1[:5]
    distances2=np.convolve(distances_pad,np.array([1,1,1])/3,'same')[5:85]
    
    tmp=np.zeros((r*4+1,r*4+1))
    tmp_ones=np.zeros((r*4+1,r*4+1))
    for a in range(80):
    
        base_angle=(np.pi/40)*a-np.pi
        cent=(int(np.ceil(distances2[a]*np.sin(base_angle))),int(np.ceil(distances2[a]*np.cos(base_angle))))
        
        x_min=max(0,x+r-2*r)
        x_max=min(image_pad.shape[1],x+r+2*r+1)
        y_min=max(0,y+r-2*r)
        y_max=min(image_pad.shape[0],y+r+2*r+1)
        
        tmp_ones[2*r-(y+r-y_min):2*r+(y_max-(y+r)),2*r-(x+r-x_min):2*r+(x_max-(x+r))]=image_pad_ones[y_min:y_max,x_min:x_max]
        
        tmp[r*2+cent[0]+line[0]-1,r*2+cent[1]+line[1]-1]=1
    
    inds=np.where((tmp*tmp_ones)==1)
    
    return (inds[0]-r*2+y,inds[1]-r*2+x)

def draw_periphery(image,ys,xs,r):
    
    dangle=np.pi/6
    base_angle=(np.pi*2)/20
    
    v1=image.mean()
    v2=image.std()
    image_pad=(np.random.randn(image.shape[0]+r*2,image.shape[1]+r*2)*np.sqrt(v2))+v1

    image_pad[r:-r,r:-r]=image
    image_pad_ones=np.zeros(image_pad.shape)
    image_pad_ones[r:-r,r:-r]=1
    
    def angular_map(maxradius):

        mask=makeDisk(maxradius)
        mask[maxradius,maxradius]=0
        (disk_r, disk_c)=np.where(mask==1)
        disk_r=disk_r-maxradius
        disk_c=disk_c-maxradius
        
        
        #return disk_r
        pixel_num=np.zeros((maxradius,));
        disk_d=np.zeros((disk_r.size,));
        disk_th=np.zeros((disk_r.size,));
        
        for a1 in range(len(disk_r)):
               dist=int(np.sqrt(disk_r[a1]**2+disk_c[a1]**2));
               pixel_num[dist-1] +=1;
               disk_d[a1]=dist;
               disk_th[a1]=np.angle(disk_r[a1]*1j+disk_c[a1])
        return (disk_r,disk_c,disk_d,disk_th)

    angle_map=angular_map(int(r))
    
    ncell=len(ys)
    ROI_inds=[]
    line=np.where(makeDisk(1)>0)
    
    for i in range(ncell):
    
        x=xs[i]
        y=ys[i]
        
        distances1=np.zeros((80,))
        
        for a in range(80):
            base_angle=(np.pi/40)*a-np.pi
            search_inds=np.where((angle_map[3]>(base_angle-dangle/2)) & (angle_map[3]<(base_angle+dangle/2)))
            angular_inds=(angle_map[0][search_inds],angle_map[1][search_inds])
            vlist1=image_pad[y+angular_inds[0]+r,x+angular_inds[1]+r]
            dlist=angle_map[2][search_inds]
            new_x=np.arange(1,int(r),0.5)
            sortinds=np.argsort(dlist)
            vlist2=np.interp(new_x,dlist[sortinds],vlist1[sortinds])
            distances1[a]=new_x[vlist2.argmax()]+1
        
        distances_pad=np.zeros((90,))
        distances_pad[5:85]=distances1
        distances_pad[:5]=distances1[-5:]
        distances_pad[-5:]=distances1[:5]
        distances2=np.convolve(distances_pad,np.array([1,1,1])/3,'same')[5:85]
        
        tmp=np.zeros((r*4+1,r*4+1))
        tmp_ones=np.zeros((r*4+1,r*4+1))
        for a in range(80):
        
            base_angle=(np.pi/40)*a-np.pi
            cent=(int(np.ceil(distances2[a]*np.sin(base_angle))),int(np.ceil(distances2[a]*np.cos(base_angle))))
            
            x_min=max(0,x+r-2*r)
            x_max=min(image_pad.shape[1],x+r+2*r+1)
            y_min=max(0,y+r-2*r)
            y_max=min(image_pad.shape[0],y+r+2*r+1)
            
            tmp_ones[2*r-(y+r-y_min):2*r+(y_max-(y+r)),2*r-(x+r-x_min):2*r+(x_max-(x+r))]=image_pad_ones[y_min:y_max,x_min:x_max]
            
            tmp[r*2+cent[0]+line[0]-1,r*2+cent[1]+line[1]-1]=1
        
        inds=np.where((tmp*tmp_ones)==1)
        ROI_inds.append((inds[0]-r*2+y,inds[1]-r*2+x))
    
    return ROI_inds

def proof_images(imlist,inds):
    
    from numpy.random import permutation
    import matplotlib.patches as patches
    import imfunctions as im
    import matplotlib.pyplot as plt
    
    image_size=imlist.shape[2]
    image_radius=int((imlist.shape[2]-1)/2)

    block_size=100
    proof=True
    block_num=np.floor(len(inds)/block_size).astype('int')+1
    
    permute_inds=permutation(len(inds))
    remove_list=[]
    
    for b in range(block_num):
        
        p1=np.zeros((image_size*10,image_size*10,3))
        proof_size=min(block_size,len(inds)-(block_size*b))
        
        for i in range(proof_size):
            image=imlist[inds[permute_inds[block_size*b+i]],:,:,:].squeeze()
            plot_position=(int(i/10),int(np.mod(i,10)))
            tmp=np.tile(im.imNormalize90(image)[:,:,None],(1,1,3))
            tmp[image_radius,image_radius,0]=1
            tmp[image_radius,image_radius,1]=0
            tmp[image_radius,image_radius,2]=0
            x=plot_position[0]*image_size
            y=plot_position[1]*image_size
            p1[x:(x+image_size),y:(y+image_size),:]=tmp
        
        
        fig=plt.figure(1,figsize=(12,12))
        fig.clear()
        ax=plt.axes()
        plt.imshow(p1)
        plt.axis('off')
        cell_list=[]
        while True:
            
            fig.canvas.draw() 
            xy=fig.ginput(1)
            
            if not xy:
                break
            else:
                xpos=int(np.floor(xy[0][0]/image_size))
                ypos=int(np.floor(xy[0][1]/image_size))
                
                x=xpos*image_size
                y=ypos*image_size
                rec=patches.Rectangle((x, y), image_size,image_size, fill=False,color='r',linewidth=2)
                label=ypos*10+xpos
                cell_list.append(inds[permute_inds[b*block_size+label]])
                remove_list.append(inds[permute_inds[b*block_size+label]])
                ax.add_patch(rec)
                ax.set_title(cell_list)
                
    return remove_list
