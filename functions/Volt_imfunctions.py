
import numpy as np
import math

def getStackDims(inDir):
    """parse xml file to get dimension information of experiment.
    Returns [x,y,z] dimensions as a list of ints
    """
    import struct
    f = open(inDir+"Stack dimensions.log", "rb")

    s = f.read(12)
    dims = struct.unpack("<lll", s)
        
    return dims


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

def makeEllipse(r1,r2):
    
    array=np.zeros((2*r1+1,2*r2+1))
    ii=0;
    while ii < (2*r1+1):
        jj=0;
        while jj < (2*r2+1):
            if math.sqrt(((ii-r1)/r1)**2 + ((jj-r2)/r2)**2) <= 1:
                array[ii,jj]=1            
            jj=jj+1
        ii=ii+1
    
    return array

def makeGaussEllipse(r1,r2):
    
    array=np.zeros((2*r1+1,2*r2+1))
    ii=0;
    while ii < (2*r1+1):
        jj=0;
        while jj < (2*r2+1):
            if math.sqrt(((ii-r1)/r1)**2 + ((jj-r2)/r2)**2) <= 1:
                array[ii,jj]=1            
            jj=jj+1
        ii=ii+1
    
    return array

def makeGaussEllipse(height, center_x, center_y, width_x, width_y,rotation):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)

        rotation = np.deg2rad(rotation)
        center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
        center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

        def rotgauss(x,y):
            xp = x * np.cos(rotation) - y * np.sin(rotation)
            yp = x * np.sin(rotation) + y * np.cos(rotation)
            g = height*np.exp(
                -(((center_x-xp)/width_x)**2+
                  ((center_y-yp)/width_y)**2)/2.)
            return g
        
        out=np.zeros((height,height))
        
        for i in range(height):
            out[:,i]=rotgauss(np.arange(height),i)
        
        return out
    


def makeDiskInd(r,dims):
    
    array=np.zeros((2*r+1,2*r+1))
    ii=0.;
    while ii < (2*r+1):
        jj=0.;
        while jj < (2*r+1):
            if math.sqrt((ii-r)**2 + (jj-r)**2) <= r:
                array[ii,jj]=1            
            jj=jj+1
        ii=ii+1
    
    ind=np.nonzero(array);
    return (ind[0]-r,ind[1]-r)
    

def makeDiskInd2(r,dims):
    
    array=np.zeros((2*r+1,2*r+1))
    ii=0.;
    while ii < (2*r+1):
        jj=0.;
        while jj < (2*r+1):
            if math.sqrt((ii-r)**2 + (jj-r)**2) <= r:
                array[ii,jj]=1            
            jj=jj+1
        ii=ii+1
    
    ind=np.nonzero(array);
    return np.int32(array), np.int32((ind[0]-r)*dims[1]+(ind[1]-r)) ,np.int32((2*r+1,2*r+1))  
 

def ones_rank(shape,r):
    
    small_ones=np.ones((2*r+1,2*r+1))
    small_ones_pad=np.pad(small_ones,(r,r),'constant', constant_values=0)
    
    disk=makeDisk(r)
    disk[r,r]=0
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    
    out_small_pad=np.zeros((small_ones_pad.shape))
    (one_r_inds,one_c_inds)=np.where(small_ones_pad>0)
    
    for i in range(len(one_r_inds)):
        out_small_pad[one_r_inds[i],one_c_inds[i]]=np.sum(small_ones_pad[one_r_inds[i]+r_inds,one_c_inds[i]+c_inds])
    
    out_small=out_small_pad[r:-r,r:-r]
    out=np.zeros(shape)
    out[:r,:r]=out_small[:r,:r]
    out[-r:,:r]=out_small[-r:,:r]
    out[:r,-r:]=out_small[:r,-r:]
    out[-r:,-r:]=out_small[-r:,-r:]
    out[:r,r:-r]=out_small[:r,r][:,None]
    out[-r:,r:-r]=out_small[-r:,r][:,None]
    out[r:-r,:r]=out_small[r,:r]
    out[r:-r,-r:]=out_small[r,-r:]
    out[r:-r,r:-r]=out_small[r,r]
    
    
    return out

    
def imrank(image,r):
    
    disk=makeDisk(r)
    disk[r,r]=0
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    
    image_pad=np.pad(image,(r,r),'constant', constant_values=0)
    (one_r_inds,one_c_inds)=np.where(image_pad>0)
    out_pad=np.zeros(image_pad.shape)
    
    for i in range(len(one_r_inds)):
        series=image_pad[one_r_inds[i]+r_inds,one_c_inds[i]+c_inds]
        v=image_pad[one_r_inds[i],one_c_inds[i]]
        out_pad[one_r_inds[i],one_c_inds[i]]=np.sum((series>0)*(series<v))
    
    return out_pad[r:-r,r:-r]


def local_minima(image,r):
    
    disk=makeDisk(r)
    disk[r,r]=0
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    
    image_pad=np.pad(image,(r,r),'constant', constant_values=2)
    (one_r_inds,one_c_inds)=np.where(image_pad<2)
    out_pad=np.zeros(image_pad.shape)
    
    for i in range(len(one_r_inds)):
        series=image_pad[one_r_inds[i]+r_inds,one_c_inds[i]+c_inds]
        v=image_pad[one_r_inds[i],one_c_inds[i]]
        out_pad[one_r_inds[i],one_c_inds[i]]=(v<=series.min())
    
    
    return out_pad[r:-r,r:-r]
    
              
def imNormalize90(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/10)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2      


def imNormalize95(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/20)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2    
    
    
def imNormalize99(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/100)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2    

    
def imNormalize999(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/1000)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2    
    
def imNormalize9999(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/10000)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    
    return array2    

    
def imNormalize99_color(array):
    
    nlen=array.size
    t=np.sort(array.reshape((1,nlen)));
    
    vmax=t[0,-round(nlen/100)]
    vmin=array.min()
    array2=(array-vmin)/(vmax-vmin)
    array2=np.clip(array2,0,1)
    array2=np.transpose(np.tile(array2,[3,1,1]),axes=(1,2,0))
    
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
    f1=ax1.imshow(imNormalize99(img),cmap=plt.get_cmap('gray'));
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
    

def calc_mov_baseline(trace,span,fraction,mode='bottom'):

    totlen=trace.size
    halfspan=np.floor(span/2)
    moves=np.arange(-halfspan,halfspan+1).astype('int32')
    calc_inds=(halfspan+np.arange(0,totlen)).astype('int32')
    
    
    tmp=np.matlib.repmat(np.append(np.arange(0,totlen)+1,np.zeros(span+1,)),1,len(moves));
    move_matrix=np.reshape(tmp[:,:-span-1],(len(moves),totlen+span))
    move_matrix=move_matrix[:,calc_inds]
    
    dim=move_matrix.shape
    
    
    avefraction=np.ceil((move_matrix!=0).sum(axis=0)*fraction).astype('int')
    one_matrix=np.zeros(move_matrix.shape)
    
    for i in range(totlen):
        one_matrix[0:avefraction[i],i]=1
                  
    out=np.zeros((1,totlen))
    
    transfer_pos=np.where((move_matrix.T)>0);
    transfer_inds=(move_matrix.T)[transfer_pos].astype('int32')
    
    
    if mode=='bottom':
        calc_matrix=np.ones(dim)*66000
        calc_matrix[transfer_pos[1],transfer_pos[0]]=trace[transfer_inds-1]
        tmp=np.sort(calc_matrix,axis=0)
    else:
        calc_matrix=np.ones(dim)*(-1)
        calc_matrix[transfer_pos[1],transfer_pos[0]]=trace[transfer_inds-1]
        tmp=np.flipud(np.sort(calc_matrix,axis=0))
        
    out=(tmp*one_matrix).sum(axis=0)/(avefraction.astype('single'))
    return out

def calcMovBaseline_parallel(trace,span,fraction):
    
    from multiprocessing import Process, Queue
    
    trace=trace.T
    
    ncell=trace.shape[0]
    totlen=trace.shape[1]
    halfspan=np.floor(span/2);
    
    moves=np.arange(-halfspan,halfspan+1)
    calc_inds=halfspan+np.arange(0,totlen)
    
    tmp=np.zeros((totlen+span+1,1))
    tmp[:totlen,0]=np.arange(1,totlen+1)
    tmp=np.tile(tmp,(len(moves),1))
    move_matrix=np.reshape(tmp[0:-span-1,0],(len(moves),totlen+span),order='C')
    move_matrix=move_matrix[:,np.int32(calc_inds)]
    
    dim=move_matrix.shape
    
    avefraction=np.float32(np.ceil(np.sum(move_matrix>0,axis=0)*fraction))
    
    one_matrix=np.zeros(move_matrix.shape);
    for i in range(totlen):
        one_matrix[0:avefraction[i],i]=1
        
    
    transfer_pos=np.where(move_matrix.T>0)
    transfer_inds=np.int32(move_matrix[transfer_pos[1],transfer_pos[0]]-1)
    calc_matrix=np.ones(dim,dtype=np.float32)*70000;
    
    baseline=np.zeros((ncell,totlen));
    
    
    def pnormalize(keys,trace_chunk,out_q):
        
        nrep=trace_chunk.shape[0]
        outdict={}
        for j in range(nrep):
            tmp=trace_chunk[j,:]
            calc_matrix[transfer_pos[1],transfer_pos[0]]=tmp[transfer_inds]
            outdict[keys[j]]=np.sum(one_matrix*np.sort(calc_matrix, axis=0),axis=0)/avefraction
            
        out_q.put(outdict)
    
    nprocs=12
    out_q=Queue()
    chunksize=int(np.ceil(float(ncell)/float(nprocs)))
    procs=[]
    
    for i in range(nprocs):
        p=Process(target=pnormalize,args=(range(chunksize*i,chunksize*(i+1)),trace[chunksize*i:chunksize*(i+1),:],out_q))
        procs.append(p)
        p.start()
    
    resultdict={}
    for i in range(nprocs):
        resultdict.update(out_q.get())
        
    for p in procs:
        p.join()
    
    for i in range(ncell):
        baseline[i,:]=resultdict[i].T;
    
    return baseline.T


def imrank_dll(image,r):
    
    import ctypes
    from numpy.ctypeslib import ndpointer
    
    imfc = ctypes.cdll.LoadLibrary(r"C:\Users\kawashimat\Documents\Spyder\functions\imfc\x64\Release\imfc.dll")
    
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
    onesout=np.zeros((image.shape[0],image.shape[1]),'int')
    
    indslist = np.where(ones_pad>0)
    inds=(indslist[0]*image_pad.shape[1]+indslist[1]).astype('int')
    
    disk=makeDisk(r)
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    rankinds=(r_inds*image_pad.shape[1]+c_inds).astype('int')
    
    func1(image_pad,ones_pad,inds,int(inds.size),rankinds,int(rankinds.size),rankout)
    func2(np.array(onesout.shape).astype('int'),disk.astype('int'),np.array(disk.shape).astype('int'),onesout)
    
    return rankout.astype('float')/onesout.astype('float')

def local_minima_dll(image,r):
    
    
    import ctypes
    from numpy.ctypeslib import ndpointer
    
    imfc = ctypes.cdll.LoadLibrary(r"C:\Users\kawashimat\Documents\Spyder\functions\imfc\x64\Release\imfc.dll")
    
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
    inds=(indslist[0]*image_pad.shape[1]+indslist[1]).astype('int')
    
    disk=makeDisk(r)   
    disk[r,r]=0
    (r_inds,c_inds)=np.where(disk>0)
    r_inds -= r
    c_inds -= r
    mininds=(r_inds*image_pad.shape[1]+c_inds).astype('int')
    
    func1(image_pad,np.array(image_pad.shape).astype('int'),mininds,int(mininds.size),inds,int(inds.size),minout)
    
    
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