# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:16:25 2017

@author: kawashimat
"""

plt.close("all")
clear_all()

import ep
import os
import Volt_Imfunctions as im
import array as ar
import struct as st
import scipy.io as sio
import pandas as pd
import Volt_ROI as ROI
from scipy import stats
import glob


dirname=r"D:\Takashi\SPIM\06152017";

subdir=glob.glob(dirname+"\\Fish*\\")


for d in range(len(subdir)):
    pathname=subdir[d]
    plt.close("all")

    print(pathname)
    imdir = pathname+"Registered\\"
    
    full_fname=imdir+"ave.tif"
    stack_fname=imdir+"Registered.tif"
    ROI_fname=imdir+"ROI_list.npy"
    
    
    if ((os.path.isfile(stack_fname)) and (os.path.isfile(ROI_fname))):
        
        ROI_list=np.load(imdir+"ROI_list.npy")[()]
        ave=imread(full_fname)
        back=(np.sort(ave.flatten())[0:int(ave.size/20)]).mean()
        nROI=len(ROI_list)
        
        
        stack=imread(stack_fname)
        image_len=stack.shape[0]
        
        for n in range(0,nROI):
            print("Neuron %d" % n)
            inds=ROI_list[n]
            if n==0:
                ROI_info=ROI.optimize_trace(stack,ave,(inds[0],inds[1]))
            else:
                ROI_info=np.append(ROI_info,ROI.optimize_trace(stack,ave,(inds[0],inds[1])))
            
        np.save(imdir+"ROI_info.npy",ROI_info)
        
        
        active_cell=np.zeros((len(ROI_info),))
        active_n=0
        plt.figure(1,figsize=(16,8))
        for i in range(nROI):
            t=ROI_info[i]
            plt.subplot(1,2,1)
            plt.ylim(-0.1,0.2*nROI+0.1)    
            if t['active']==0:
                plt.plot(np.arange(len(t['norm_tcourse1']))/300,t['norm_tcourse1']-1+0.2*i,color=(0.3,0.3,0.3))
            else:
                plt.plot(np.arange(len(t['norm_tcourse1']))/300,t['norm_tcourse1']-1+0.2*i,color=(1,0,0))
                
            if t['active']==1:
                active_n+=1;
                active_cell[i]=1
                tlimit=t['tlimit2']
                spikes=np.where(t['spike_tcourse2']>0)[0]
                
                plt.subplot(1,2,2).plot(np.arange(tlimit)/300,t['norm_tcourse2'][:tlimit]-1+0.2*(active_n-1))
                plt.subplot(1,2,2).plot(np.arange(tlimit,len(t['norm_tcourse2']))/300,t['norm_tcourse2'][tlimit:len(t['norm_tcourse2'])]-1+0.2*(active_n-1),color=(0.3,0.3,0.3))
                
                for s in range(len(spikes)):
                    plt.subplot(1,2,2).plot(spikes[s]/300,0.15+0.2*(active_n-1),'ko',markersize=1)
                plt.ylim(-0.1,0.2*active_n+0.1) 
                
        plt.savefig(imdir+"activity_timecourse.png")
                

        
        active_cells=np.where(active_cell)[0]
        active_tcourse=np.zeros((active_n,image_len))
        active_spike_tcourse=np.zeros((active_n,image_len))
        
        for i in range(len(active_cells)):
            active_tcourse[i,:]=ROI_info[active_cells[i]]['norm_tcourse2'];
            active_spike_tcourse[i,:]=ROI_info[active_cells[i]]['spike_tcourse2'];
        
        np.save(imdir+"active_tcourse.npy",active_tcourse)
        np.save(imdir+"active_spike_tcourse.npy",active_spike_tcourse)
        
        
        plt.figure(3,figsize=(18,6))
        plt.subplot(131).imshow(im.imNormalize(ave.astype('float'),99.9), cmap='gray')
        
        img_color=np.tile(im.imNormalize(ave.astype('float'),99.9)[:,:,None],(1,1,3))
        for n in range(len(active_cells)):
            inds=ROI_list[active_cells[n]]
            img_color[inds[0].astype('int'),inds[1].astype('int'),0]=1
            img_color[inds[0].astype('int'),inds[1].astype('int'),1]=0
            img_color[inds[0].astype('int'),inds[1].astype('int'),2]=0
        
        plt.subplot(132).imshow(img_color)
        
        ax=plt.subplot(133)
        tmp=np.zeros(ave.shape)
        for n in range(len(active_cells)):
            inds=ROI_list[active_cells[n]]
            tmp[inds[0].astype('int'),inds[1].astype('int')]=1
            
        plt.imshow(tmp, cmap='gray')
        
        for n in range(len(active_cells)):
            inds=ROI_list[active_cells[n]]
            location=(inds[0].mean(),inds[1].mean())
            ax.text(location[1],location[0],n,fontsize=12, color='r')
        
        plt.savefig(imdir+"active_ROI.png")
        
            
        active_cell_inds=np.where(active_cell)[0];
        
        
        f=plt.figure(4,figsize=(12,6))
        nrow=np.ceil(len(active_cells)/5)
        for c in range(len(active_cell_inds)):
            t=ROI_info[active_cell_inds[c]]
            if t['active']==1:
                spikes=np.where(t['spike_tcourse2']==1)[0]
                spikes=spikes[(spikes > 10) & (spikes<image_len-11)]
                spike_matrix=np.zeros((len(spikes),21))
                for i in range(len(spikes)):
                    spike_matrix[i,:]=t['norm_tcourse2'][spikes[i]-10:spikes[i]+11]
                    spike_matrix[i,:]-=np.median(spike_matrix[i,:])
                ave=spike_matrix.mean(axis=0)
                std=spike_matrix.std(axis=0)
                
                ax=plt.subplot(nrow,5,c+1)
                plt.plot(np.arange(-10,11)/300*1000,spike_matrix.T,color= (0.8, 0.8, 0.8))
                plt.plot(np.arange(-10,11)/300*1000,ave)
        
        f.subplots_adjust(hspace=0.7)  
        
        plt.savefig(imdir+"spike_shape.png")
