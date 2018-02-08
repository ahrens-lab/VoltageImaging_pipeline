#if defined _WIN32
    #define DLL_PUBLIC __declspec(dllexport)
#else
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
#endif

#include "imfc.hpp"



// making ones template for rank calculation

DLL_PUBLIC void imonesrank(int *dim, int *mask, int *maskdim, int *result){
 
	int maskp[2],margin[2];

	maskp[0]=maskdim[0]*2-1;
	maskp[1]=maskdim[1]*2-1;

	margin[0]=(maskdim[0]-1)/2;
	margin[1]=(maskdim[1]-1)/2;
	int total;
	total=dim[0]*dim[1];  

 	int i0,i1, d0,d1, dp0,dp1, dt0,dt1;
	int *moveinds1, *T;
	int total_move=0;
	 
	T = (int *)calloc(maskdim[1]*maskdim[0],sizeof(int));
	moveinds1=(int *)calloc(maskdim[1]*maskdim[0],sizeof(int));
	 
	for (i1=0;i1<maskdim[0];i1++){
	     dt1=maskp[1]*(i1-margin[0]);
	     d1=maskdim[1]*i1;

	     for (i0=0;i0<maskdim[1];i0++){
		 dt0=i0-margin[1];             
		 if (mask[d1+i0]==1){
			 moveinds1[total_move]=dt1+dt0;
			 total_move +=1;
		 }
	     }
	 } 

	 // make template  ///////////////////////////////////////////////////////////////////////////
	  
	  
	 int total_template=maskdim[0]*maskdim[1];
	 int total_template_padding=maskp[0]*maskp[1];
	  
	 int *templ;
	 templ=(int *)calloc(total_template_padding,sizeof(int));
	 int *temp_inds;
	 temp_inds=(int *)calloc(total_template,sizeof(int));
	 
	 for (i1=0;i1<maskdim[0];i1++){    
	     dp1=maskp[1]*(margin[0]+i1);
	     d1 =maskdim[1]*i1;    

	     for (i0=0;i0<maskdim[1];i0++){
		 d0=d1+i0;
		 dp0=dp1+(margin[1]+i0);
		 templ[dp0]=1;
		 temp_inds[d0]=dp0;
	     }  
	 }

	    
	 int i, j,l;
	 for (j=0;j<total_template;j++){
	   T[j]=0;
	   l=temp_inds[j];
	   i=total_move;
	   while (i--){
		T[j] +=templ[l+moveinds1[i]];
	   }
	 }
	  

	  
	 // make move index ///////////////////////////////////////////////////////////////////////////
	  
	 int t1,t0; 
	     
	 for (i1=0;i1<dim[0];i1++){
	     if (i1<margin[0]){t1=i1;}else{
	     if (i1>=dim[0]-margin[0]){t1=maskdim[0]-(dim[0]-i1);}else{
	     t1=margin[0];}} 

	     dp1=maskdim[1]*t1;
	     d1 =dim[1]*i1;

	     for (i0=0;i0<dim[1];i0++){             
		 if (i0<margin[1]){t0=i0;}else{
		 if (i0>=dim[1]-margin[1]){t0=maskdim[1]-(dim[1]-i0);}else{
		 t0=margin[1];}}                      

		 d0=i0;
		 dp0=t0;
		 result[d1+d0]=T[dp1+dp0];             

	     }  
	 }
 

}


// thresholding based on local contrast

DLL_PUBLIC void threshold_contrast(unsigned short *img, int *dim, int *grid, int *thre, unsigned char *output){
        
    int ylim=dim[0];
    int xlim=dim[1];
    int gridn=grid[0];
    int gridlen=grid[0]*grid[0];
    
    
    int ygrids=ylim/ gridn;
    int xgrids=xlim/ gridn;
    int xmod=xlim % gridn;
    int xgrids2=xgrids+(xmod>0);
    int ngrids=xgrids2*ygrids;
    int gridlen2=xmod*gridn;
    unsigned short vthre=(unsigned short)*thre;
    int yy,xx,zz,iii,jjj;

    
    // calculating index within each grids
    
    int *gridplus,*gridplus2;
    gridplus=(int *)calloc(gridlen, sizeof (int));
    gridplus2=(int *)calloc(gridlen2, sizeof (int));
    
    
    int ii=0;
    for (yy=0;yy<gridn;yy++)            
    {
        for (xx=0;xx<gridn;xx++)
        {
            gridplus[ii]=yy*xlim+xx;      
            ii++;
        }
    }

    ii=0;
    for (yy=0;yy<gridn;yy++)            
    {
        for (xx=0;xx<xmod;xx++)
        {
            gridplus2[ii]=yy*xlim+xx;      
            ii++;
        }
    }
    
    // calculating minimum value for each grid
    unsigned short vmin,tempv;
    int sourceinds,inds;   
    
    unsigned short *minarray;
    minarray=(unsigned short *)calloc(ngrids,sizeof(unsigned short));
    
    for (yy=0;yy<ygrids;yy++)            
    {
        for (xx=0;xx<xgrids;xx++)
        {
            sourceinds=yy*gridn*xlim + xx*gridn;
            vmin=60000;
            for (zz=0; zz<gridlen; zz++)
            {
                tempv=img[sourceinds+gridplus[zz]];
                if (vmin > tempv){vmin=tempv;};
            }
            minarray[xgrids2*yy+xx]=vmin;
        }
    }
    

    
    if (xmod>0){
                
        for (yy=0;yy<ygrids;yy++)
        {
            sourceinds=gridn*xlim*yy + xgrids*gridn;
            vmin=60000;
            for (zz=0; zz<gridlen2; zz++)
            {
                tempv=img[sourceinds+gridplus2[zz]];
                if (vmin > tempv){vmin=tempv;};
            }
            minarray[xgrids2*yy+xgrids]=vmin;
        }
    }
    
    // local averaging minimum value for each grid
    
    int xave[3]={-1,0,1};
    int yave[3]={-1,0,1};
    int ave_inds[9],xinds[9],yinds[9];
    int xind, yind;
    unsigned short *minarray2;
    float tmp,tt;
    
    minarray2=(unsigned short *)calloc(ngrids,sizeof(unsigned short));
    
    for (iii=0;iii<3;iii++){
        for (jjj=0;jjj<3;jjj++){
            ave_inds[iii*3+jjj]=yave[iii]*xgrids2+xave[jjj];
            xinds[iii*3+jjj]=xave[jjj];
            yinds[iii*3+jjj]=yave[iii];
        }
    }
    
    
    for (yy=0;yy<ygrids;yy++)            
    {
        for (xx=0;xx<xgrids2;xx++)
        {
            sourceinds=yy*xgrids2 + xx;
            
            tt=0; tmp=0;
            for (zz=0; zz<9; zz++)
            {
                inds=sourceinds+ave_inds[zz];
				xind=xx+xinds[zz];
				yind=yy+yinds[zz];

                if (xind>=0 && xind<xgrids2 && yind>=0 && yind<ygrids)
                {
                    tt += 1;
                    tmp +=(float)minarray[inds];
                }                    
            }
            minarray2[sourceinds]=(unsigned short)(tmp/tt);
        }
    }
        
    
    
    
    
    // thresholding each pixel on entire image
    
    
    
    for (yy=0;yy<ygrids;yy++)            
    {
        for (xx=0;xx<xgrids;xx++)
        {
            sourceinds=yy*gridn*xlim + xx*gridn;
         
            vmin=minarray2[yy*xgrids2+xx];
            //vmin=110;
            
            for (zz=0; zz<gridlen ; zz++){
                inds=sourceinds+gridplus[zz];
                output[inds]=((img[inds]-vmin)>vthre);
            }
        }
    }

    
    if (xmod>0){
        
        for (yy=0;yy<ygrids;yy++)
        {
            sourceinds=gridn*xlim*yy + xgrids*gridn;
            
            vmin=minarray2[yy*xgrids2+xgrids];
	    //vmin=110;
            
            for (zz=0; zz<gridlen2 ; zz++){
                inds=sourceinds+gridplus2[zz];
                output[inds]=((img[inds]-vmin)>vthre);
            }
        }
    }
            
}



// calculating local rank of pixels

DLL_PUBLIC void imrank(unsigned short *imp,unsigned short *oop,int *inds,int indlen, int *rankinds, int ranklen, unsigned short *temp_out)
{

  unsigned short *temp_out_ori=temp_out;
  
  int i, j,mindex;  
  j=indlen;
  while (j--){
      i=ranklen;      
      while (i--){
          mindex= *inds + rankinds[i];
          *temp_out +=(oop[mindex])*(imp[*inds]>=imp[mindex]);
         
      }
      inds++;
      temp_out++;
  }

  temp_out=temp_out_ori;

}

DLL_PUBLIC void local_average(float *img, int*dim, int *aveinds, int avelen, int *candidates, int indlen, float *out)
{
    
    
    int before_inds,i,j;
    float tmp;
    int *aveinds_ori=aveinds;
    int imlen=dim[0]*dim[1];    

    for (i=0; i<indlen; i++){
        tmp=0;
        for (j=0; j<avelen; j++){
            before_inds = ((*candidates-*aveinds)+imlen) % imlen;
            tmp += img[before_inds];
	    aveinds++;
        }

        out[*candidates]=tmp/(float)avelen;
	candidates++;
        aveinds=aveinds_ori;
    }

}


DLL_PUBLIC void local_max(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out)
{
    
    int before_inds,i,j;
    int imlen=dim[0]*dim[1];    
    unsigned char tmp;
    int *maxinds_ori=maxinds;
    float cent;

    
    
    for (i=0; i<indlen; i++){
        tmp=1;
        cent=img[*candidates];
        for (j=0; (j<maxlen) && (tmp==1) ; j++){
            before_inds = ((*candidates-*maxinds)+imlen) % imlen;
            tmp *= (cent>=img[before_inds]);
            maxinds++;
        }
        out[*candidates]=tmp;
        candidates++;
        maxinds=maxinds_ori;
    }
}

DLL_PUBLIC void local_min(float *img, int*dim, int *maxinds, int maxlen, int *candidates, int indlen, unsigned char *out)
{
    
    int before_inds,i,j;
    int imlen=dim[0]*dim[1];    
    unsigned char tmp;
    int *maxinds_ori=maxinds;
    float cent;

    
    for (i=0; i<indlen; i++){
        tmp=1;
        cent=img[*candidates];
        for (j=0; (j<maxlen) && (tmp==1) ; j++){
            before_inds = ((*candidates-*maxinds)+imlen) % imlen;
            tmp *= (cent<=img[before_inds]);
            maxinds++;
        }
        out[*candidates]=tmp;
        candidates++;
        maxinds=maxinds_ori;
    }
          

}



