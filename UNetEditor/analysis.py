from PIL import Image
import numpy as np
from numba import njit
import cv2
import scipy.spatial
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpl_toolkits.axes_grid1
plt.style.use('seaborn-v0_8-dark')

class ImageEditor:
    def __init__(self,invert=False,intensity_clip=(0,1),xc=None,yc=None,sigma=0):
        self.invert = invert
        self.intensity_clip = intensity_clip
        self.xc = xc
        self.yc = yc
        self.sigma = sigma

    def set_sigma(self,sigma):
        self.sigma = sigma

    def set_invert(self,invert):
        self.invert = invert

    def set_intensity_clip(self,intensity_clip):
        self.intensity_clip = intensity_clip

    def set_xc(self,xc):
        self.xc = xc

    def set_yc(self,yc):
        self.yc = yc

    def get_sigma(self):
        return self.sigma

    def get_invert(self):
        return self.invert

    def get_intensity_clip(self):
        return self.intensity_clip

    def get_xc(self):
        return self.xc 

    def get_yc(self):
        return self.yc
    
    def __call__(self,x):
        img = x.copy()
        if len(img.shape)>2:
            img =  np.mean(img,axis=-1)

        if self.xc is not None:
            x0,x1 = self.xc
            x0,x1 = max(x0,0),min(x1,img.shape[1])
            img = img[:,int(np.floor(x0)):int(np.ceil(x1))]
        
        if self.yc is not None:
            y0,y1 = self.yc
            y0,y1 = max(y0,0),min(y1,img.shape[0])
            img = img[int(np.floor(y0)):int(np.ceil(y1)),:]
        
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = np.fft.fft2(img)
        img = scipy.ndimage.fourier_gaussian(img,self.sigma)
        img = (np.fft.ifft2(img)).real
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        if self.invert:
            img = 1-img
        v0,v1 = self.intensity_clip
        output = {}
        output["clipimg"] = np.clip(img,v0,v1)
        output["cliprescaleimg"] = np.clip(1/(v1-v0)*(img-v0),0,1)
        output["histinfo"] = img.flatten()
        return output
@njit
def fpca(mask,labels,x,y):
    sm = np.sum(mask)
    diameter_pixel = 2*np.sqrt(sm/np.pi) 
    xc = np.sum(x*mask)/sm
    yc = np.sum(y*mask)/sm
    x2 = np.sum((x-xc)**2*mask)/sm
    y2 = np.sum((y-yc)**2*mask)/sm
    xy = np.sum((y-yc)*(x-xc)*mask)/sm
    eigenvalues,eigenvectors = np.linalg.eig(np.array([[x2,xy],[xy,y2]]))
    eigenvalues = 4*np.sqrt(eigenvalues)
    imin,imax = np.argsort(eigenvalues)
    vmax = eigenvectors[:,imax]
    if vmax[0]<0:
        vmax = -vmax
    angle = -np.arctan2(vmax[1],vmax[0])*180/np.pi#minus, since y axis starts at the top
    if np.abs(eigenvalues[imax])!=0:
        ellip = (eigenvalues[imax]-eigenvalues[imin])/eigenvalues[imax]
    else:
        ellip = 0
    
    return xc,yc,diameter_pixel,eigenvalues[imax],eigenvalues[imin],angle,ellip


class MaskAnalysis:
    def __init__(self,nbins=40):
        self.pred_label = None
        self.img = None
        quant = self.quant = [{"ix":0,"quant":r"Skyrmion diameter $d_A=2\sqrt{A/\pi}$ [pixel]","title":"Skyrmion diameter statistic","unit":"pixel","qminmax_standard":(0.5,"img_s_max"),"slider_type":"log"},
        {"ix":1,"quant":r"Skyrmion manjor axis $a_1$ [pixel]","title":"Skyrmion major axis statistic","unit":"pixel","qminmax_standard":(0.5,"img_s_max"),"slider_type":"log"},
        {"ix":2,"quant":r"Skyrmion minor axis $a_2$ [pixel]","title":"Skyrmion minor axis statistic","unit":"pixel","qminmax_standard":(0.5,"img_s_max"),"slider_type":"log"},
        {"ix":3,"quant":r"Skyrmion orientation [°]","title":"Skyrmion orientation statistic","vmin":-90,"vmax":90,"colbar":"hsv","unit":"°","qminmax_standard":(-90,90),"slider_type":"lin"},
        {"ix":4,"quant":r"Skyrmion ellipticity $(a_1-a_2)/a_1$ [1]","title":"Skyrmion ellipticity statistic","unit":"","qminmax_standard":(0,1),"slider_type":"lin"},
        {"ix":5,"quant":r"Skyrmion convex hull diameter $d_H$ [1]","title":"Skyrmion convex hull diameter statistic","unit":"","qminmax_standard":(0.5,"img_s_max"),"slider_type":"log"},
        {"ix":6,"quant":r"Skyrmion diameter to convex hull diameter ratio $(d_H-d_A)/d_H$ [1]","title":"Skyrmion diameter to convex hull diameter ratio statistic","unit":"","qminmax_standard":(0,1),"slider_type":"lin"},
        {"ix":7,"quant":r"Skyrmion hole area [$\mathrm{pixel}^2$]","title":"Skyrmion hole area","unit":"$\mathrm{pixel}^2$","qminmax_standard":(0.5,"img_s_max"),"slider_type":"log"}]

        for ix in range(len(quant)):
            quant[ix]["qminmax"] = (None,None)

        self.option = [{"ix":"boundary","option":"Filter skyrmions in contact with boundary","value":True,"standard_value":True}]
    
        self.min_max_diameter_range = None
        self.nbins = nbins
        with plt.ioff():
            self.fig = plt.figure(dpi=100,figsize=(10,4*(1+len(self.quant))))
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False
        gs = self.fig.add_gridspec(len(self.quant)+1,3,width_ratios=[1,1,0.05],height_ratios=[1]+[1]*(len(self.quant)))
        self.ax = [self.fig.add_subplot(gs[0,0]),self.fig.add_subplot(gs[0,1])]
        self.im1 = self.ax[0].imshow(np.random.rand(20,20,3))
        self.im2 = self.ax[1].imshow(np.random.rand(20,20),cmap="gray",vmin=0,vmax=1,zorder=-900)
        self.lcontobj =  []
        self.ax[1].set_title("Skyrmion contours")
        self.ax[0].set_title("Single skyrmion mask")
        self.ax[0].grid(False)
        self.ax[1].grid(False)
        self.qaxes = []
        for i in range(len(quant)):
            quant[i]["qax"] = qax = [self.fig.add_subplot(gs[i+1,0]),self.fig.add_subplot(gs[i+1,1]),self.fig.add_subplot(gs[i+1,2])]
            quant[i]["qaxhist"],quant[i]["qaxim"],quant[i]["qaxc"] = qax
            quant[i]["qim"] = qim = qax[1].imshow(np.random.rand(20,20),cmap=quant[i].get("colbar","viridis"),
                                                  vmax=quant[i].get("vmin",None),vmin=quant[i].get("vmax",None))
            quant[i]["qcb"] = qcb = plt.colorbar(qim,cax=qax[2],label=quant[i]["quant"])#,location='left')
            qax[0].set_title(quant[i]["title"])
            qax[0].set_xlabel(quant[i]["quant"])
            qax[0].set_ylabel("Frequency")
            
        self.fig.tight_layout()

    def get_quant(self):
        return [ele["quant"] for ele in self.quant]
        
    def set_min_max_quant_select_range(self,ra):
        for ix in range(len(self.quant)):
            self.quant[ix]["qminmax"] = ra[ix]
        #import time
        #t0 = time.time()
        #print("->",ra[0])
        if self.pred_label is not None: self.analysis_2()
        #print("->",time.time()-t0)

    def get_min_max_quant_select_range(self):
        return [list(self.quant[ix]["qminmax"]) for ix in range(len(self.quant))]

    def get_slider_type(self):
        return [self.quant[ix]["slider_type"] for ix in range(len(self.quant))]

    def get_min_max_quant_standard_select_range(self):
        return [self.quant[ix]["qminmax_standard"] for ix in range(len(self.quant))]

    def get_option_name(self):
        return [ele["option"] for ele in self.option]
        
    def set_option(self,ra):
        for ix in range(len(self.option)):
            self.option[ix]["value"] = ra[ix]
        if self.pred_label is not None: self.analysis_2()

    def get_option(self):
        return [self.option[ix]["value"] for ix in range(len(self.option))]

    def get_standard_option(self):
        return [self.option[ix]["standard_value"] for ix in range(len(self.option))]
        
    def __call__(self,pred_label,img,not_update_plot=False):
        self.pred_label = pred_label
        self.img = img
        self.analysis_1()
        if not not_update_plot:
            self.analysis_2()
        
    def analysis_1(self):
        #import time
        #t0 = time.time()
        _,labels,stats,_ = cv2.connectedComponentsWithStats((self.pred_label==0).astype(np.uint8),cv2.CV_32S)
        
        self.labels = labels.copy()
        labels2 = np.repeat(np.repeat(labels,2,axis=0),2,axis=1)
        xi,yi = np.meshgrid(np.arange(labels.shape[1]),np.arange(labels.shape[0]))
        ldata = []
        lcont = []

        for ii in range(1,np.max(labels)+1):
            j0,j1,i0,i1 = stats[ii,0],stats[ii,0]+stats[ii,2],stats[ii,1],stats[ii,1]+stats[ii,3]
            mask = labels[i0:i1,j0:j1]==ii
            mask2 = labels2[2*i0:2*i1,2*j0:2*j1]==ii
            xc,yc,diameter_pixel,eigenvalues_max,eigenvalues_min,angle,ellip = fpca(mask,labels,xi[i0:i1,j0:j1],yi[i0:i1,j0:j1])
                
            li,_ = cv2.findContours((mask).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for i in range(len(li)):
                li[i][:,0,0] += j0
                li[i][:,0,1] += i0
            lcont.append(li)
            li2,_ = cv2.findContours((mask2).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            lc = np.array([np.ceil(cv2.convexHull(li2[i])/2) for i in range(len(li2))],dtype=int)
            Aconvexhull = sum([cv2.contourArea(lc[e]) for e in range(len(lc))])
            diameter_convexhull = 2*np.sqrt(Aconvexhull/np.pi) 
            if np.abs(diameter_convexhull)>0:
                diameter_convexeps = (diameter_convexhull-diameter_pixel)/diameter_convexhull
            else:
                diameter_convexeps = 0

            li2h,hhist = cv2.findContours((mask2).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            lch = [np.array(np.ceil(li2h[i]/2),dtype=int) for i in range(len(li2h))]
            lhole = [cv2.contourArea(lch[e]) for e in range(len(lch)) if hhist[0,e,-1]>=0]
            if len(lhole)==0:
                shole = 0
            else:
                shole = sum(lhole)
            
            ldata.append((xc,yc,diameter_pixel,eigenvalues_max,eigenvalues_min,angle,ellip,
                           diameter_convexhull,diameter_convexeps,shole))
            
        ldata = self.ldata = np.array(ldata)
        self.lcont = lcont
        self.colmap = np.vstack((np.ones(3),0.87*np.random.rand(np.max(labels),3)))
        
        #t1 = time.time()
        #print("time",t1-t0)
        
    def get_ixl(self):
        labels = self.labels.copy()
        ixlb = np.ones(np.max(labels),dtype=bool)
        
        for i in range(len(self.quant)):
            v0,v1 = self.quant[i]["qminmax"]
            ix = self.quant[i]["ix"]
            value = self.ldata[:,2+ix]
            if v0 is not None and (type(self.quant[i]["qminmax_standard"][0]) != str and v0!= self.quant[i]["qminmax_standard"][0]):
                ixlb[value<v0] = False
            if v1 is not None:
                ixlb[value>v1] = False
            
        ixlbn = np.zeros(np.max(labels)+1,dtype=bool)
        ixlbn[1:] = ixlb

        filterboundary = True
        for ele in self.option:
            if ele["ix"] == "boundary":
                filterboundary = ele["value"]

        if filterboundary:
            boundaryixl = np.array(list(set(labels[0])|set(labels[-1])|set(labels[:,0])|set(labels[:,-1])))
            ixlbn[boundaryixl] = False
            
        ixlbn[0] = False
        return np.arange(len(ixlbn))[ixlbn]

    def analysis_2(self):
        while len(self.lcontobj)>0:
            obj = self.lcontobj.pop()
            obj.remove()
        
        labels = self.labels.copy()
        ixl = self.get_ixl()
        labels[np.invert(np.isin(labels,ixl))] = 0

        for i in range(len(self.lcont)):
            if i+1 not in ixl: continue
            for e in range(len(self.lcont[i])):
                lx,ly = list(self.lcont[i][e][:,0,0]),list(self.lcont[i][e][:,0,1])
                self.lcontobj.append(self.ax[1].plot(lx+[lx[0]],ly+[ly[0]],color="r",lw=0.5)[0])
        
        self.im1.set_data(self.colmap[labels])
        self.im2.set_data(self.img)
        self.im1.set_extent((0,self.pred_label.shape[1],self.pred_label.shape[0],0))
        self.im2.set_extent((0,self.pred_label.shape[1],self.pred_label.shape[0],0))
        self.fig.tight_layout()

        if len(ixl) > 0:
            for i in range(len(self.quant)):
                ix = self.quant[i]["ix"]
                qvalue = self.ldata[ixl-1,2+ix]
                colormap = np.array([np.nan]+list(self.ldata[:,2+ix]))
                qimg = colormap[labels]
                self.quant[i]["qim"].set_data(qimg)
                self.quant[i]["qim"].set_extent((0,self.pred_label.shape[1],self.pred_label.shape[0],0))
                values = qimg[np.invert(np.isnan(qimg))].flatten()
                #print(np.min(values),np.max(values))
                self.quant[i]["qim"].set_clim(self.quant[i].get("vmin",np.min(qvalue)),self.quant[i].get("vmax",np.max(qvalue)))
                self.quant[i]["qaxim"].set_facecolor((1,1,1,1))
                binsl = np.linspace(np.min(qvalue)-1e-12,np.max(qvalue)+1e-12,self.nbins)
                hb,xb,bobj = self.quant[i]["qaxhist"].hist(qvalue,bins=binsl,color="#1f77b4",density=True)
                self.lcontobj += bobj
                self.quant[i]["qaxhist"].set_ylim(0,np.max(hb)*1.1)
                self.quant[i]["qaxhist"].set_xlim(self.quant[i].get("vmin",np.min(qvalue)-1e-12),
                                                  self.quant[i].get("vmax",np.max(qvalue)+1e-12))
                mrl,msl = np.mean(qvalue),np.std(qvalue)
                meanrhist =  self.quant[i]["qaxhist"].axvline(mrl,color="red",lw=3,label="Mean value:\n"+fr"(${mrl:.2f}\pm {msl:.2f}$) "+self.quant[i]["unit"])
                self.lcontobj += [meanrhist]
                #t = np.linspace(np.min(qvalue),np.max(qvalue),100)
                #pl = self.quant[i]["qaxhist"].plot(t,1/np.sqrt(2*np.pi*msl**2)*np.exp(-(t-mrl)**2/(2*msl**2)),color="red")
                #self.lcontobj += pl
        else:
            for i in range(len(self.quant)):
                self.quant[i]["qim"].set_data([[np.nan]])
            
    def get_filter_label(self):
        labels = self.labels.copy()
        ixl = self.get_ixl()
        labels[np.invert(np.isin(labels,ixl))] = 0
        limg = np.zeros(labels.shape,dtype=np.uint8)
        limg[labels==0] = 2
        return limg
        
    def update_canvas(self):
        self.fig.canvas.draw()

    def get_datatable(self):
        ixl = self.get_ixl()-1
        #ixl = ixl[np.argsort(self.ldata[ixl,2+self.quant[0]["ix"]])[::-1]]
        D = {"$p_x$ [pixel]":self.ldata[ixl,0],"$p_y$ [pixel]":self.ldata[ixl,1]}
        D.update({self.quant[i]["quant"]:self.ldata[ixl,2+self.quant[i]["ix"]] for i in range(len(self.quant))})
        return pd.DataFrame(D)

    def get_posl(self):
        ixl = self.get_ixl()-1
        return self.ldata[ixl,:2]

class PosAnalysis:
    def __init__(self):
        self.posl = None
        self.min_angle_range = 0
        with plt.ioff():
            self.fig = plt.figure(dpi=100,figsize=(10,18))
        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = False
        #gs = self.fig.add_gridspec(4,2,width_ratios=[1.5,4],height_ratios=[2,0.5,0.5,0.5])
        gs = self.fig.add_gridspec(4,2,width_ratios=[4,4],height_ratios=[2.5,1,1,1])
        self.ax = [self.fig.add_subplot(gs[0,:]),self.fig.add_subplot(gs[1,1]),self.fig.add_subplot(gs[1,0]),self.fig.add_subplot(gs[2,0]),self.fig.add_subplot(gs[2,1]),self.fig.add_subplot(gs[3,0]),self.fig.add_subplot(gs[3,1])]
        self.lobj = []

        import mpl_toolkits.axes_grid1
        cax1 = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax[2]).append_axes('right', size='5%', pad=0.03)
        self.sc1 = self.ax[2].scatter(x=[],y=[],c=[],cmap="viridis")
        self.cb1 = plt.colorbar(self.sc1,cax=cax1,label="Sky-Sky dist.")
        self.lobj.append(self.sc1)
        
        cax2 = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax[3]).append_axes('right', size='5%', pad=0.03)
        self.sc2 = self.ax[3].scatter(x=[],y=[],c=[],cmap="viridis")
        self.cb2 = plt.colorbar(self.sc2,cax=cax2,label="$\Psi_6$")
        self.lobj.append(self.sc2)

        cax3 = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax[5]).append_axes('right', size='5%', pad=0.03)
        self.sc3 = self.ax[5].scatter(x=[],y=[],c=[],cmap="viridis")
        self.cb3 = plt.colorbar(self.sc3,cax=cax3,label="$\Psi_4$")
        self.lobj.append(self.sc3)
        
        self.ax[0].set_title("Delaunay triangulation result")
        self.ax[0].grid(False)
        self.ax[1].set_xlabel(r"Skyrmion-Skyrmion distance [pixel]")
        #self.ax[1].legend()
        self.ax[1].set_ylabel(r"Frequency")
        self.ax[4].set_ylabel(r"Frequency")
        self.ax[6].set_ylabel(r"Frequency")
        self.ax[1].set_title("Skyrmion-Skyrmion statistic")
        self.ax[4].set_title(r"$\Psi_6$ analysis")
        self.ax[6].set_title(r"$\Psi_4$ analysis")
        self.ax[4].set_xlabel(r"$\Psi_6$")
        self.ax[6].set_xlabel(r"$\Psi_4$")
        self.im1 = self.ax[0].imshow(np.zeros((20,20)),cmap="gray",vmin=0,vmax=1)
        self.fig.tight_layout()
        
    def set_min_angle_select_range(self,v):
        self.min_angle_range = v
        if self.posl is not None: self.analysis_2()

    def get_min_angle_select_range(self):
        return self.min_angle_range
        
    def __call__(self,posl,img):
        self.posl = posl
        self.img = img
        self.analysis_1()
        self.analysis_2()

    def analysis_1(self):
        if len(self.posl)>=4:
            self.voronoi = scipy.spatial.Voronoi(self.posl)
            self.delaunay = scipy.spatial.Delaunay(self.posl)
        else:
            self.voronoi = None
            self.delaunay = None

    @staticmethod
    def get_distance_stat(posl,lcon):
        ld = np.zeros((len(lcon),3))
        posxl,posyl,distl = [],[],[]
        for ix,(a,b) in enumerate(lcon):
            mpos = 0.5*(posl[a]+posl[b])
            ld[ix,0] = mpos[0]
            ld[ix,1] = mpos[1]
            ld[ix,2] = np.linalg.norm(posl[a]-posl[b])
        return ld

    @staticmethod
    def get_psi(posl,lcon,S,n):
        #ld = np.zeros((len(S),3))
        ld = {}
        
        for e,ix0 in enumerate(S):
            tmpl = np.setdiff1d(np.unique(lcon[np.any(lcon==ix0,axis=1)]),[ix0])
            p0 = posl[ix0]
            l,o = 0.0,0.0
            for ix1 in tmpl:
                p1 = posl[ix1]
                v0 = p1-p0
                l0 = np.sqrt(v0[0]**2+v0[1]**2)
                theta =  np.arctan2(v0[1]/l0,v0[0]/l0)
                o += np.exp(n*1j*theta)
                l += l0
            #posxl.append(posl[ix0,0])
            #posyl.append(posl[ix0,1])
            if len(tmpl)==0:
                print(np.unique(lcon[np.any(lcon==ix0,axis=1)]))
            #ol.append(np.abs(o/len(tmpl)))
            #meanl.append(l/len(tmpl))
            ld[ix0] = (np.abs(o/len(tmpl)),l/len(tmpl))
        #print(np.sum(np.abs(posl[S,0]-np.array(posxl))),np.sum(np.abs(posl[S,1]-np.array(posyl))))
        
        return ld
        
    @staticmethod
    def get_delaunay_filter(posl,delaunay,min_angle_range=15,psi4=False):
        distance = []
        #select out triangles with small angles (only occurs at the boundary of the image; therefore, they do not represent real skyrmion distances)
        lallcon = []
        for ele in delaunay.simplices:
            for i in range(len(ele)):
                a,b = ele[i],ele[(i+1)%len(ele)]
                lallcon.append([min(a,b),max(a,b)])
        lcon,ncon = np.unique(np.array(lallcon),axis=0,return_counts=True)
        lbond = set([tuple(ele) for ele in lcon[ncon!=2]])
    
        lcon,lcon1,lcon2 = [],[],[]
        for it,ele in enumerate(delaunay.simplices):
            ok = True
            for i in range(len(ele)):
                v1,v2 = posl[ele[(i+1)%len(ele)]]-posl[ele[i]],posl[ele[(i-1)%len(ele)]]-posl[ele[i]]
                if not (np.pi/180*min_angle_range<=np.arccos(np.clip(np.dot(v1,v2)/np.sqrt(np.dot(v1,v1)*np.dot(v2,v2)),-1,1))):#<=np.pi/180*self.min_max_angle_range[1]):
                    ok = False
                    break
            if ok:
                tmpl = []
                for i in range(len(ele)):
                    a,b = ele[i],ele[(i+1)%len(ele)]
                    tmpl.append([np.linalg.norm(posl[a]-posl[b]),min(a,b),max(a,b)])
                tmpl = sorted(tmpl)
                if psi4==True:
                    lzu = [lcon, lcon, lcon2]
                else:
                    lzu = [lcon, lcon, lcon]
                for i, lcon_list in zip(range(3), lzu):
                    a, b = tmpl[i][1], tmpl[i][2]
                    if (a, b) not in lbond:
                        lcon_list.append([a, b])
                    else:
                        lcon1.append([a, b])
            else:
                for i in range(len(ele)):
                    a,b = ele[i],ele[(i+1)%len(ele)]
                    lcon1.append([min(a,b),max(a,b)])
    
        lcon,lcon1,lcon2 = np.array(lcon),np.array(lcon1),np.array(lcon2)
        lcon,ncon = np.unique(lcon,axis=0,return_counts=True) 
        lcon1,ncon1 = np.unique(lcon1,axis=0,return_counts=True)
        l1 = np.unique(lcon)
        l2 = np.unique(lcon1)
        S = np.setdiff1d(l1,l2)
        nf = np.any(np.isin(lcon,S),axis=1)
        lcon1 = np.vstack((lcon1,lcon[~nf]))
        lcon = lcon[nf]
        lcon2,ncon2 = np.unique(lcon2,axis=0,return_counts=True)
        return lcon,lcon1,lcon2,S
    
    def analysis_2(self):
        #return None
        while len(self.lobj)>0:
            obj = self.lobj.pop()
            obj.remove()
        
        
        
        if (self.delaunay is not None) and (self.voronoi is not None):
            #lo.append([self.posl[ix0][0],self.posl[ix0][1],o6,o4,l,ix0])
            lcon,lcon1,lcon2,S = self.get_delaunay_filter(self.posl,self.delaunay,self.min_angle_range,psi4=False)
            self.ld = ld = self.get_distance_stat(self.posl,lcon)
            psi6order = self.get_psi(self.posl,lcon,S,6)
            lcon_psi4,lcon1_psi4,lcon2_psi4,S_psi4 = self.get_delaunay_filter(self.posl,self.delaunay,self.min_angle_range,psi4=True)
            psi4order = self.get_psi(self.posl,lcon_psi4,S_psi4,4)

            lo = np.zeros((len(self.posl),6))
            for ix0 in range(len(self.posl)):
                lo[ix0,0] = self.posl[ix0,0]
                lo[ix0,1] = self.posl[ix0,1]
                
                if ix0 in psi6order:
                    lo[ix0,2] = psi6order[ix0][0]
                    lo[ix0,4] = psi6order[ix0][1] 
                else:
                    lo[ix0,2] = np.nan
                    lo[ix0,4] = np.nan

                if ix0 in psi4order:
                    lo[ix0,3] = psi4order[ix0][0]
                else:
                    lo[ix0,3] = np.nan
                
                lo[ix0,5] = ix0
                
            #[self.posl[ix0][0],self.posl[ix0][1],o6,o4,l,ix0]
            self.lo = lo = np.array(lo)
            
            for ele in self.voronoi.regions:
                if len(ele)==0 or len(list(filter(lambda x:x==-1,ele)))>0: continue
                self.lobj.append(self.ax[0].plot([self.voronoi.vertices[i,0] for i in list(ele)+[ele[0]]],[self.voronoi.vertices[i,1] for i in list(ele)+[ele[0]]],lw=1.5,color="b")[0])
            
            #for a,b in lcon:
            #    self.lobj.append(self.ax[0].plot([self.posl[a,0],self.posl[b,0]],[self.posl[a,1],self.posl[b,1]],color="r",lw=1)[0])
            self.lobj += self.ax[0].plot(np.array([[self.posl[a,0],self.posl[b,0]] for a,b in lcon]).T,np.array([[self.posl[a,1],self.posl[b,1]] for a,b in lcon]).T,color="r",lw=1)   
            
            #for a,b in lcon1:
            #    self.lobj.append(self.ax[0].plot([self.posl[a,0],self.posl[b,0]],[self.posl[a,1],self.posl[b,1]],color="g",lw=1)[0])
            self.lobj += self.ax[0].plot(np.array([[self.posl[a,0],self.posl[b,0]] for a,b in lcon1]).T,np.array([[self.posl[a,1],self.posl[b,1]] for a,b in lcon1]).T,color="g",lw=1)   
            
            
            #for ele in S:
            #    self.lobj.append(self.ax[0].plot(self.delaunay.points[ele][0],self.delaunay.points[ele][1],"mo",ms=7)[0])
            if len(S)>0:
                self.lobj.append(self.ax[0].plot(self.posl[S][:,0],self.posl[S][:,1],"mo",ms=7)[0])
            
            
            self.sc1 = self.ax[2].scatter(x=ld[:,0],y=ld[:,1],c=ld[:,2],s=2,cmap="viridis")
            self.cb1.update_normal(self.sc1)
            self.lobj.append(self.sc1)
            self.sc2 = self.ax[3].scatter(x=lo[:,0],y=lo[:,1],c=lo[:,2],s=2,cmap="viridis")
            self.cb2.update_normal(self.sc2)
            self.lobj.append(self.sc2)
            self.sc3 = self.ax[5].scatter(x=lo[:,0],y=lo[:,1],c=lo[:,3],s=2,cmap="viridis")
            self.cb3.update_normal(self.sc3)
            self.lobj.append(self.sc3)
            
            if len(ld[:,2])>0:
                yh,xh,histobj = self.ax[1].hist(ld[:,2],color='#1f77b4',density=True,bins=40)
                nanmask = ~np.isnan(ld[:,2])
                mean,sigma = np.mean(ld[nanmask,2]),np.std(ld[nanmask,2])
                #t = np.linspace(np.min(ld[:,2]),np.max(ld[:,2]),100)
                #pl = self.ax[1].plot(t,1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t-mean)**2/(2*sigma**2)),color="red")[0]
                histmean = self.ax[1].axvline(np.mean(ld[:,2]),color="red",lw=3,label=fr"Mean value: (${mean:.2f}\pm {sigma:.2f}$) pixel")
                self.lobj += [histobj,histmean]
                self.ax[1].set_ylim(0,np.max(yh)*1.2)
                self.ax[1].set_xlim(np.min(xh),np.max(xh))
                self.ax[1].legend()

            if len(lo)>0:
                yh,xh,histobj = self.ax[4].hist(lo[:,2],color='#1f77b4',density=True,bins=40)
                nanmask = ~np.isnan(lo[:,2])
                mean,sigma = np.mean(lo[nanmask,2]),np.std(lo[nanmask,2])
                #t = np.linspace(np.min(lo[:,2]),np.max(lo[:,2]),100)
                #pl = self.ax[4].plot(t,1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t-mean)**2/(2*sigma**2)),color="red")[0]
                histmean = self.ax[4].axvline(mean,color="red",lw=3,label=fr"Mean value: ${mean:.2f}\pm {sigma:.2f}$")
                self.lobj += [histobj,histmean]
                self.ax[4].set_ylim(0,np.max(yh)*1.2)
                self.ax[4].set_xlim(np.min(xh),np.max(xh))

                yh,xh,histobj = self.ax[6].hist(lo[:,3],color='#1f77b4',density=True,bins=40)
                nanmask = ~np.isnan(lo[:,3])
                mean,sigma = np.mean(lo[nanmask,3]),np.std(lo[nanmask,3])
                #t = np.linspace(np.min(lo[:,3]),np.max(lo[:,3]),100)
                #pl = self.ax[6].plot(t,1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(t-mean)**2/(2*sigma**2)),color="red")[0]
                histmean = self.ax[6].axvline(mean,color="red",lw=3,label=fr"Mean value: ${mean:.2f}\pm {sigma:.2f}$")
                self.lobj += [histobj,histmean]
                self.ax[6].set_ylim(0,np.max(yh)*1.2)
                self.ax[6].set_xlim(np.min(xh),np.max(xh))
                self.ax[4].legend()
                self.ax[6].legend()
                
        
        self.im1.set_data(self.img)#,cmap="gray",origin="lower")
        self.im1.set_extent((0,self.img.shape[1],self.img.shape[0],0))

        for ix in [0,2,3,5]:
            self.ax[ix].set_xlim(0,self.img.shape[1])
            self.ax[ix].set_ylim(self.img.shape[0],0)
        
        
        self.fig.tight_layout()

    def update_canvas(self):
        self.fig.canvas.draw()

    def get_datatable(self):
        #ixl = self.get_ixl()-1
        #ixl = ixl[np.argsort(self.ldata[ixl,2+self.quant[0]["ix"]])[::-1]]
        #D = {"$p_x$ [pixel]":self.ldata[ixl,0],"$p_y$ [pixel]":self.ldata[ixl,1]}
        #D.update({self.quant[i]["quant"]:self.ldata[ixl,2+self.quant[i]["ix"]] for i in range(len(self.quant))})
        return pd.DataFrame({"Skyrmion-Skyrmion distance [pixel]":self.lo[:,4],
                             "$\Psi_6$ [1]":self.lo[:,2],"$\Psi_4$ [1]":self.lo[:,3]},index=self.lo[:,5].astype(int))

from .prediction import SkyUNet
import shutil
import os
import glob
import io

def get_video_analysis(fnv,config,tmp_folder,result_file):
    if (config["mask_analysis_editor"] is None) or (config["pos_analysis_editor"] is None):
        return

    imgeditor = ImageEditor(invert=config["img_editor"]["invert"],
                            intensity_clip=config["img_editor"]["intensity_clip"],
                            xc=config["img_editor"]["xc"],
                            yc=config["img_editor"]["yc"],
                            sigma=config["img_editor"]["sigma"])
    
    sky_unet = SkyUNet()
    sky_unet.set_model(config["prediction_editor"]["unet_model"][0],model_ver=config["prediction_editor"]["unet_model"][2])
    
    analysis1 = MaskAnalysis()
    analysis1.set_min_max_quant_select_range(config["mask_analysis_editor"]["shape_range"])
    analysis1.set_option(config["mask_analysis_editor"]["shape_option"])
    
    analysis1.fig.set_dpi(300)
    
    analysis2 = PosAnalysis()
    analysis2.set_min_angle_select_range(config["pos_analysis_editor"]["angle_range"])
    analysis2.fig.set_dpi(300)
    
    for ele in glob.iglob(tmp_folder+"*"):
        os.remove(ele)
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)
    if os.path.isfile(result_file+".zip"):
        os.remove(result_file+".zip")

    prefix_file = tmp_folder+os.path.splitext(os.path.basename(fnv))[0]+"_frame_"
    #print(fnv)
    cap = cv2.VideoCapture(fnv)
    capl = int(np.ceil(np.log10(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))+1))
    Q = []
    ix = 0
    while True:
        ret,frame = cap.read()
        if ret:
            Q.append(imgeditor(frame)["cliprescaleimg"])

        if len(Q) == 5 or (not ret and len(Q)>0):
            pred = sky_unet(np.array(Q))
            for z in range(len(pred)):
                analysis1(pred[z],Q[z],True)
                predimg = sky_unet.trafo_channel_to_rgb(analysis1.get_filter_label())
                plt.imsave(prefix_file+f"{ix:0{capl}}"+"_pred.png",predimg)        
                analysis1.get_datatable().to_csv(prefix_file+f"{ix:0{capl}}"+"_datatable.csv",index=False)
                ix += 1
            del Q
            Q = []
            
        if not ret:
            break
         
    shutil.make_archive(result_file,"zip",tmp_folder)
    for ele in glob.iglob(tmp_folder+"*"):
        os.remove(ele)
    return result_file+".zip"

def get_batch_analysis(fnl,config,tmp_folder,result_file):
    if (config["mask_analysis_editor"] is None) or (config["pos_analysis_editor"] is None):
        return

    imgeditor = ImageEditor(invert=config["img_editor"]["invert"],
                            intensity_clip=config["img_editor"]["intensity_clip"],
                            xc=config["img_editor"]["xc"],
                            yc=config["img_editor"]["yc"],
                            sigma=config["img_editor"]["sigma"])
    
    sky_unet = SkyUNet()
    sky_unet.set_model(config["prediction_editor"]["unet_model"][0],model_ver=config["prediction_editor"]["unet_model"][2])
    
    analysis1 = MaskAnalysis()
    analysis1.set_min_max_quant_select_range(config["mask_analysis_editor"]["shape_range"])
    analysis1.set_option(config["mask_analysis_editor"]["shape_option"])
    
    analysis1.fig.set_dpi(300)
    
    analysis2 = PosAnalysis()
    analysis2.set_min_angle_select_range(config["pos_analysis_editor"]["angle_range"])
    analysis2.fig.set_dpi(300)

    for ele in glob.iglob(tmp_folder+"*"):
        os.remove(ele)
    if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)
    if os.path.isfile(result_file+".zip"):
        os.remove(result_file+".zip")
        
    for i,(input,inputfn) in enumerate(fnl):
        #prefix_file = tmp_folder+os.path.splitext(ele.name)[0]+"_ix_"+str(i)
        #input = np.array(Image.open(io.BytesIO(ele.content.tobytes())))
        prefix_file = tmp_folder+os.path.splitext(inputfn)[0]+"_ix_"+str(i)
        img = imgeditor(input)["cliprescaleimg"]
        pred = sky_unet(img)
        predimg = sky_unet.trafo_channel_to_rgb(pred)
        analysis1(pred,img)
        analysis2(analysis1.get_posl(),img)
        
        plt.imsave(prefix_file+"_input.png",input,cmap="gray")
        plt.imsave(prefix_file+"_edited.png",img,cmap="gray")
        plt.imsave(prefix_file+"_skyunet_prediction.png",predimg)
        ((analysis1.get_datatable()).join(analysis2.get_datatable())).to_csv(prefix_file+"_datatable.csv",index=False)
        #analysis1.get_datatable().to_csv(prefix_file+"_datatable.csv")
        analysis1.fig.savefig(prefix_file+"_mask_analysis.png",bbox_inches="tight")
        analysis2.fig.savefig(prefix_file+"_pos_analysis.png",bbox_inches="tight")
        #break
    
    shutil.make_archive(result_file,"zip",tmp_folder)
    for ele in glob.iglob(tmp_folder+"*"):
        os.remove(ele)
    return result_file+".zip"