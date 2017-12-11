# -*- coding: utf-8 -*-


plt.close("all")
clear_all()

import Volt_Imfunctions as im
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes


from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th') # color channel first



pathname=r"D:\Takashi\SPIM\06152017";
fname="Fish3-3"


br_threshold1=120
cell_radius=5
rank_radius=int(cell_radius*1.5)
model = load_model(os.getcwd()+r'\\functions\\cell_model_32.hdf5')

imdir=pathname+"\\"+fname+"\\Registered\\"
ave=imread(imdir+"\\ave.tif")

image_radius=14 ## fixed value for deep learning datasets
mask=binary_fill_holes(ave>br_threshold1)
rank_image=gaussian_filter(im.imrank_dll(gaussian_filter(ave,cell_radius*0.2),rank_radius),cell_radius*0.2)
minima_image=im.local_minima_dll(rank_image,int(cell_radius*1.5))
overlay=np.tile(im.imNormalize(ave,95)[:,:,None],(1,1,3))/2
cell_centers=np.where((minima_image>0) & mask)
cell_centers_yx=np.array([cell_centers[0],cell_centers[1]]).T
    
overlay[cell_centers[0],cell_centers[1],0]=1
overlay[cell_centers[0],cell_centers[1],1]=0
overlay[cell_centers[0],cell_centers[1],2]=0


plt.figure(1)
plt.imshow(overlay)

v1=ave.mean()
v2=ave.std()
ave_pad=(np.random.randn(ave.shape[0]+image_radius*2,ave.shape[1]+image_radius*2)*np.sqrt(v2))+v1
ave_pad[image_radius:-image_radius,image_radius:-image_radius]=ave
rank_pad=np.zeros((ave.shape[0]+image_radius*2,ave.shape[1]+image_radius*2))
rank_pad[image_radius:-image_radius,image_radius:-image_radius]=rank_image

cell_r=cell_centers[0][np.where(cell_centers[0]>cell_radius*2)]
cell_r=cell_centers[0]
cell_c=cell_centers[1]

imlist=np.zeros((len(cell_r),1,image_radius*2+1,image_radius*2+1))
classlist=np.zeros((len(cell_r),))
for i in range(len(cell_r)):
    imlist[i,0,:,:]=im.imNormalize(ave_pad[cell_r[i]:(cell_r[i]+image_radius*2+1), \
          cell_c[i]:(cell_c[i]+image_radius*2+1)],99)[None,None,:,:]



scores=model.predict(imlist)
label=(scores[:,0]>0.5).astype('int')
label[np.where(label==0)]=2

inds1=np.where(label==1)[0]
inds2=np.where(label==2)[0]

disk=im.makeDisk(2)
disk_r=np.where(disk>0)[0]-2
disk_c=np.where(disk>0)[1]-2
cell_image=np.tile(im.imNormalize(ave,99.9)[:,:,None],(1, 1, 3))
cell_image_pad=np.zeros((cell_image.shape[0]+image_radius*2,cell_image.shape[1]+image_radius*2,3))
cell_image_pad[image_radius:-image_radius,image_radius:-image_radius,:]=cell_image
cell_image_pad2=cell_image_pad.copy()
for i in range(len(cell_r)):
    if label[i]==1:
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,0]=1
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,1]=0
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,2]=0
    if label[i]==2:
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,0]=0
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,1]=1
        cell_image_pad[cell_r[i]+disk_r+image_radius,cell_c[i]+disk_c+image_radius,2]=1

plt.figure(2)
plt.imshow(cell_image_pad)

cell_inds1=im.proof_images(imlist,inds1)



cell_r=cell_r[cell_inds1]
cell_c=cell_c[cell_inds1]
cnum1=len(cell_inds1)


ROI_inds1=im.draw_periphery(ave,cell_r,cell_c,cell_radius*2)
    

for i in range(len(cell_inds1)):

    cell_image_pad2[ROI_inds1[i][0]+image_radius,ROI_inds1[i][1]+image_radius,0]=1
    cell_image_pad2[ROI_inds1[i][0]+image_radius,ROI_inds1[i][1]+image_radius,1]=0
    cell_image_pad2[ROI_inds1[i][0]+image_radius,ROI_inds1[i][1]+image_radius,2]=0
            
fig=plt.figure(3)
plt.subplot(121).imshow(im.imNormalize(ave_pad,99.9))
a2=plt.subplot(122)
xs=[]
ys=[]
while True:
    
    a2.cla()
    plt.subplot(122).imshow(cell_image_pad2)
    fig.canvas.draw() 
    xy=fig.ginput(1)
    
    if not xy:
        break
    else:
        if (xy[0][0]<ave.shape[1]+image_radius) and (xy[0][1]<ave.shape[0]+image_radius) and (min(xy[0])>=image_radius):
            xs.append(int(xy[0][0])-image_radius)
            ys.append(int(xy[0][1])-image_radius)
            ROI=im.draw_periphery_single(ave,int(xy[0][1]-image_radius),int(xy[0][0]-image_radius),cell_radius*2)
            cell_image_pad2[ROI[0]+image_radius,ROI[1]+image_radius,0]=1
            cell_image_pad2[ROI[0]+image_radius,ROI[1]+image_radius,1]=0
            cell_image_pad2[ROI[0]+image_radius,ROI[1]+image_radius,2]=0
            
            
            

ROI_inds2=im.draw_periphery(ave,np.array(ys),np.array(xs),cell_radius*2)
        
for i in range(len(ys)):

    cell_image_pad2[ROI_inds2[i][0]+image_radius,ROI_inds2[i][1]+image_radius,0]=1
    cell_image_pad2[ROI_inds2[i][0]+image_radius,ROI_inds2[i][1]+image_radius,1]=0
    cell_image_pad2[ROI_inds2[i][0]+image_radius,ROI_inds2[i][1]+image_radius,2]=0
    
fig=plt.figure(4)
plt.subplot(121).imshow(im.imNormalize(ave_pad,99.9))
plt.subplot(122).imshow(cell_image_pad2)

ROI_list=ROI_inds1+ROI_inds2


np.save(imdir+"\\ROI_list.npy",ROI_list)

nROI=len(ROI_list)

plt.figure(5,figsize=(18,6))
plt.subplot(131).imshow(im.imNormalize(ave.astype('float'),99.9), cmap='gray')
img_color=np.tile(im.imNormalize(ave.astype('float'),99.9)[:,:,None],(1,1,3))
for n in range(nROI):
    inds=ROI_list[n]
    img_color[inds[0].astype('int'),inds[1].astype('int'),0]=1
    img_color[inds[0].astype('int'),inds[1].astype('int'),1]=0
    img_color[inds[0].astype('int'),inds[1].astype('int'),2]=0

plt.subplot(132).imshow(img_color)

ax=plt.subplot(133)
tmp=np.zeros(ave.shape)
for n in range(nROI):
    inds=ROI_list[n]
    tmp[inds[0].astype('int'),inds[1].astype('int')]=1
    
plt.imshow(tmp, cmap='gray')

for n in range(nROI):
    inds=ROI_list[n]
    location=(inds[0].mean(),inds[1].mean())
    ax.text(location[1],location[0],n,fontsize=12, color='r')

plt.savefig(imdir+"ROI.png")
