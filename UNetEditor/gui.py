import ipywidgets
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('seaborn-v0_8-dark')
import numpy as np
from .analysis import ImageEditor,MaskAnalysis,PosAnalysis,get_batch_analysis,get_video_analysis
from .prediction import SkyUNet
from PIL import Image
import shutil
import os
import glob
import io

class ImageEditorGUI(ipywidgets.HBox):
    def __init__(self,update_event=None,reset_event=None):
        super().__init__()
        with plt.ioff():
            self.fig1,self.ax1 = plt.subplots(dpi=100)
            self.fig2,self.ax2 = plt.subplots(dpi=100,figsize=(2,2))
            self.fig1.canvas.header_visible = False
            self.fig2.canvas.header_visible = False
            self.fig1.canvas.toolbar_visible = False
            self.fig2.canvas.toolbar_visible = False

        self.update_event = update_event
        self.reset_event = reset_event
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")
        self.ax1.grid(False)
        cmg = matplotlib.colormaps.get_cmap("gray")
        clis = cmg(np.arange(cmg.N))
        clis[0] = [1,0,0,1]
        clis[-1] = [0,0,1,1]
        self.ncmap = matplotlib.colors.ListedColormap(clis)
        self.imgedit = ImageEditor()
        self.img = np.random.rand(20,20)
        self.im1 = self.ax1.imshow(self.img,cmap=self.ncmap,vmin=0,vmax=1)
        self.cax = make_axes_locatable(self.ax1).append_axes("right",size = "2%",pad=0.03)
        self.colorbar = plt.colorbar(self.im1,cax=self.cax)
        self.bins = np.linspace(0,1,40)
        _,_,self.hist = self.ax2.hist(self.img.flatten(),bins=self.bins,density=True)

        self.ax2.set_xlim(0,1)
        self.ax2.set_xlabel("Intensity")
        self.lowerl = self.ax2.axvline(0,color="red")
        self.higherl = self.ax2.axvline(1,color="blue")
        self.fig2.tight_layout()

        self.int_slider = ipywidgets.FloatRangeSlider(description="Intensity clip",step=0.01)
        self.yc_slider = ipywidgets.IntRangeSlider(description="y-crop")
        self.xc_slider = ipywidgets.IntRangeSlider(description="x-crop")
        self.inv_check = ipywidgets.Checkbox(description="Inversion")
        self.zoom_slider1 = ipywidgets.FloatSlider(min=0.2,max=5,value=1.5,description="Plot zoom")
        self.zoom_slider2 = ipywidgets.FloatSlider(min=0.2,max=5,value=1.5,description="Hist. zoom")
        self.gaussian_slider = ipywidgets.FloatSlider(min=0.0,max=5,value=0,description="Gaussian")
        self.rzoom_button = ipywidgets.Button(description="Reset zoom")
        self.rzoom_button.on_click(self.event_resetzoom)
        
        self.children = [ipywidgets.VBox([self.xc_slider,self.yc_slider,self.inv_check,self.int_slider,self.gaussian_slider,self.zoom_slider1,self.zoom_slider2,self.rzoom_button,self.fig2.canvas],layout=ipywidgets.Layout(width="30%",align_items="center")),self.fig1.canvas]  

        self.widget_observe()
    
    def event_resetzoom(self,x):
        self.zoom_slider1.value = 1.5
        self.zoom_slider2.value = 1.5
    
    def event_intslider(self,x):
        self.imgedit.set_intensity_clip(x.new)
        self.update()

    def event_xcslider(self,x):
        self.imgedit.set_xc(x.new)
        self.update()

    def event_ycslider(self,x):
        self.imgedit.set_yc(x.new)
        self.update()

    def event_invcheck(self,x):
        self.imgedit.set_invert(x.new)
        self.update()

    def event_zoomslider1(self,x):
        self.fig1.set_dpi(100*x.new)
        self.update()

    def event_zoomslider2(self,x):
        self.fig2.set_dpi(100*x.new)
        self.update()

    def event_gaussian(self,x):
        self.imgedit.set_sigma(x.new)
        self.update()
    
    def widget_observe(self):
        self.int_slider.observe(self.event_intslider,"value")
        self.yc_slider.observe(self.event_ycslider,"value")
        self.xc_slider.observe(self.event_xcslider,"value")
        self.inv_check.observe(self.event_invcheck,"value")
        self.zoom_slider1.observe(self.event_zoomslider1,"value")
        self.zoom_slider2.observe(self.event_zoomslider2,"value")   
        self.gaussian_slider.observe(self.event_gaussian,"value")   
    
    def widget_unobserve(self):
        self.int_slider.unobserve(self.event_intslider,"value")
        self.yc_slider.unobserve(self.event_ycslider,"value")
        self.xc_slider.unobserve(self.event_xcslider,"value")
        self.inv_check.unobserve(self.event_invcheck,"value")
        self.zoom_slider1.unobserve(self.event_zoomslider1,"value")
        self.zoom_slider2.unobserve(self.event_zoomslider2,"value")  
        self.gaussian_slider.unobserve(self.event_gaussian,"value")  

    def __call__(self,img,canvasdraw=True):
        self.img = img
        self.init_gui(canvasdraw)

    def init_gui(self,canvasdraw=True):
        self.imgedit.set_xc((0,self.img.shape[1]))
        self.imgedit.set_yc((0,self.img.shape[0]))
        self.imgedit.set_intensity_clip((0,1))
        self.imgedit.set_sigma(0)
        self.imgedit.set_invert(False)
        
        self.widget_unobserve()
        self.int_slider.min,self.int_slider.max = self.imgedit.get_intensity_clip()
        self.xc_slider.min,self.xc_slider.max = self.imgedit.get_xc()
        self.yc_slider.min,self.yc_slider.max = self.imgedit.get_yc()
        self.gaussian_slider.min,self.gaussian_slider.max = 0,max(self.img.shape[1],self.img.shape[0])/5
        self.int_slider.value = self.imgedit.get_intensity_clip()
        self.xc_slider.value = self.imgedit.get_xc()
        self.yc_slider.value = self.imgedit.get_yc()
        self.inv_check.value = self.imgedit.get_invert()      
        self.gaussian_slider.value = 0.0
        self.zoom_slider1.value = 1.5
        self.zoom_slider2.value = 1.5
        self.widget_observe()

        self.fig1.set_dpi(150)
        self.fig2.set_dpi(150)
        self.update(canvasdraw)

    def update(self,canvasdraw=False):
        output = self.imgedit(self.img)
        editimg,histinfo = output["clipimg"],output["histinfo"]
        
        self.im1.set_data(editimg)
        xc,yc = self.imgedit.get_xc(),self.imgedit.get_yc()
        self.im1.set_extent((xc[0],xc[1],yc[1],yc[0]))
        
        v0,v1 = self.imgedit.get_intensity_clip()
        self.lowerl.set_xdata([v0,v0])
        self.higherl.set_xdata([v1,v1])
        self.ax1.set_aspect(1)
        
        self.im1.set_clim(v0,v1)
        self.colorbar.update_normal(self.im1)
        
        nhist = np.histogram(histinfo,self.bins,density=True)[0]
        self.nhist=nhist
        for i,ele in enumerate(self.hist):
            ele.set_height(nhist[i])
        self.ax2.set_ylim(0,np.max(nhist)*1.1)
        
        if canvasdraw:
            self.fig1.canvas.draw()
            self.fig2.canvas.draw()    

        if self.update_event is not None:
            self.update_event()

    def get_editimg(self):
        return self.imgedit(self.img)["cliprescaleimg"]

    def get_config(self):
        return {"intensity_clip":self.imgedit.get_intensity_clip(),
                "xc":self.imgedit.get_xc(),
                "yc":self.imgedit.get_yc(),
                "invert":self.imgedit.get_invert(),
                "sigma":self.imgedit.get_sigma()}

class UNetPredictionGUI(ipywidgets.VBox):
    def __init__(self,modeldic,pred_event=None):
        super().__init__()
        self.modeldic = modeldic
        self.cmodel = modeldic[0]
        self.skyunet = SkyUNet()

        self.pred_event = pred_event
        self.button_predict = ipywidgets.Button(description="Predict")
        self.model_dropdown = ipywidgets.Dropdown(options={b[1]:a for a,b in modeldic.items()})
        self.zoomslider = ipywidgets.FloatSlider(min=0.2,max=5,value=1.5,description="Plot zoom")
        self.resetzoom = ipywidgets.Button(description="Reset zoom")
        self.out = ipywidgets.Output()

        with plt.ioff():
            self.fig,self.ax = plt.subplots(ncols=2,dpi=100,figsize=(10,5))
            self.fig.canvas.header_visible = False
            self.fig.canvas.toolbar_visible = False
        
        self.fig.canvas.header_visible = False
        self.img = np.random.rand(20,20)
        self.pred = None
        self.im1 = self.ax[0].imshow(self.img,cmap="gray",vmin=0,vmax=1)
        self.im2 = self.ax[1].imshow(np.ones((20,20,3)))
        self.ax[0].set_title("Kerr image")
        self.ax[1].set_title("Predicted label")
        self.ax[0].grid(False)
        self.ax[1].grid(False)
        self.textobj = self.ax[1].text(0,self.img.shape[0]/2,"Please click the \"Predict\" button to make a prediction with the U-Net :-)",fontsize=20,wrap=True)        
        self.children = [ipywidgets.HBox([self.zoomslider,self.resetzoom]),ipywidgets.HBox([self.model_dropdown,self.button_predict]),self.out,self.fig.canvas]
    
        self.resetzoom.on_click(self.event_resetzoom)
        self.button_predict.on_click(self.event_predict)
        self.widget_observe()
        self.start_out()
        
    def get_config(self):
        fnmodel = self.cmodel[0]
        return {"unet_model":[ele for ele in self.modeldic.values() if ele[0]==fnmodel][0]}
    
    def start_out(self):
        with self.out:
            self.out.clear_output()
            print("",end="\r")
    
    def event_resetzoom(self,x):
        self.zoomslider.value = 1.5

    def event_predict(self,x):
        self.start_out()
        log = False
        with self.out:
            if log:
                import time
                t0 = time.time()
            modelt = self.cmodel
            self.skyunet.set_model(modelt[0],model_ver=modelt[2])
            if log:
                print("set model",time.time()-t0)
                t0 = time.time()
            self.predix = self.skyunet(self.img)
            if log:
                print(time.time()-t0)
                t0 = time.time()
            self.pred = self.skyunet.trafo_channel_to_rgb(self.predix)
            if self.pred_event is not None:
                self.pred_event()
            if log:
                print("pred event",time.time()-t0)
                t0 = time.time()
            self.update()
            self.fig.canvas.draw()
            if log:
                print("end",time.time()-t0)

    def __call__(self,img,canvasdraw=True):
        self.img = img
        self.textobjremove()
        self.pred = None
        self.init_gui(canvasdraw)

    def set_model(self,x):
        self.cmodel = self.modeldic[x.new]

    def widget_observe(self):
        self.zoomslider.observe(self.event_zoomslider,"value")
        self.model_dropdown.observe(self.set_model,"value")
        
    def widget_unobserve(self):
        self.zoomslider.unobserve(self.event_zoomslider,"value")
        self.model_dropdown.unobserve(self.set_model,"value")

    def event_zoomslider(self,x):
        self.fig.set_dpi(100*x.new)

    def textobjremove(self):
        try:
            self.textobj.remove()
        except:
            pass
    
    def update(self,canvasdraw=True):
        self.im1.set_data(self.img)
        self.im1.set_extent((0,self.img.shape[1],0,self.img.shape[0]))
        self.textobjremove()
        if self.pred is None:
            pred = np.ones((self.img.shape[0],self.img.shape[1],3))
        else:
            pred = self.pred
            
        self.im2.set_data(pred)
        self.im2.set_extent((0,pred.shape[1],0,pred.shape[0]))
        self.fig.tight_layout()
        if self.pred is None:
            self.textobj = self.ax[1].text(0,self.img.shape[0]/2,"Please click the \"Predict\" button to make a prediction with the U-Net :-)",fontsize=20,wrap=True)    

        

    def get_prediction(self):
        return self.predix

    def init_gui(self,canvasdraw=True):
        self.update(canvasdraw)

    def reset_gui(self):
        self.re = 43
        self.widget_unobserve()
        self.model_dropdown.value = 0
        self.zoomslider.value = 1.5
        self.fig.set_dpi(150)
        self.textobjremove()
        self.pred = None
        self.widget_observe()
        self.update()

class MaskAnalysisGUI(ipywidgets.VBox):
    def __init__(self,event_maskanalysis):
        super().__init__()
        self.label = None
        self.event_maskanalysis = event_maskanalysis
        self.zoomslider = ipywidgets.FloatSlider(min=0.2,max=5,value=1.5,description="Plot zoom")
        self.resetzoom = ipywidgets.Button(description="Reset zoom")

        self.analysis = MaskAnalysis()
        self.analysis.fig.set_visible(False)
    
        self.qlabels = qlabels = self.analysis.get_quant()
        self.qslidertyp = qslidertyp = self.analysis.get_slider_type()
        qsliderwidget = []
        qhboxl = []
        
        for i in range(len(qlabels)):
            q_label = ipywidgets.Label(qlabels[i])
            q_label.layout.width = "35%"
            
            if qslidertyp[i] == "lin":            
                q_slider_max = ipywidgets.FloatSlider(min=0.5,max=100,value=0.5,readout_format=".1f",description="max",step=0.01,continuous_update=False)
                q_slider_min = ipywidgets.FloatSlider(min=0.5,max=100,value=0.5,readout_format=".1f",description="min",step=0.01,continuous_update=False)
            else:
                q_slider_max = ipywidgets.FloatLogSlider(min=np.log10(0.5),max=np.log10(100),value=0.5,readout_format=".1f",description="max",step=0.01,continuous_update=False)
                q_slider_min = ipywidgets.FloatLogSlider(min=np.log10(0.5),max=np.log10(100),value=0.5,readout_format=".1f",description="min",step=0.01,continuous_update=False)

            q_slider_min.layout.width="30%"
            q_slider_max.layout.width="30%"
            qsliderwidget.append((q_slider_min,q_slider_max))
            qhboxl.append(ipywidgets.HBox([q_label,q_slider_min,q_slider_max]))

        self.qsliderwidget = qsliderwidget

        self.qoptlabels = qoptlabels = self.analysis.get_option_name()
        qoptwidget = []
        
        for i in range(len(qoptlabels)):
            q_label = ipywidgets.Label(qoptlabels[i])
            q_label.layout.width = "35%"
            q_check = ipywidgets.Checkbox(value=True)
            q_check.layout.width="30%"
            qoptwidget.append(q_check)
            qhboxl.append(ipywidgets.HBox([q_label,q_check]))

        self.qoptwidget = qoptwidget
        #self.out = ipywidgets.Output()
        self.children = [ipywidgets.HBox([self.zoomslider,self.resetzoom])]+qhboxl+[self.analysis.fig.canvas]
        self.resetzoom.on_click(self.event_resetzoom)
        self.widget_observe()    

    def event_resetzoom(self,x):
        self.zoomslider.value = 1.5

    def __call__(self,label,img,canvasdraw=True):
        self.label = label
        self.img = img
        self.init_gui(canvasdraw)

    """
    def event_diameter_max_sel(self,x):
        #self.diameter_slider_min.max = np.log10(x.new)
        self.event_diameter_sel_2()

    def event_diameter_min_sel(self,x):
        #self.diameter_slider_max.min = np.log10(x.new)
        self.event_diameter_sel_2()

    def event_diameter_sel_2(self):
        self.analysis.set_min_max_diameter_select_range((self.diameter_slider_min.value,self.diameter_slider_max.value))
        if self.event_maskanalysis is not None:
            self.event_maskanalysis()
    """
    def event_slider(self,x):
        #with self.out:
        #print("-",x)
        l = self.analysis.get_min_max_quant_select_range()
        wobj = x.owner
        for i in range(len(self.qsliderwidget)):
            wa,wb = self.qsliderwidget[i]
            if wa==wobj:
                l[i][0] = x.new
            elif wb==wobj:
                l[i][1] = x.new
        self.analysis.set_min_max_quant_select_range(l)
        if self.event_maskanalysis is not None:
            self.event_maskanalysis()
        self.analysis.update_canvas()

    def event_opt(self,x):
        #with self.out:
        l = self.analysis.get_option()
        wobj = x.owner
        for i in range(len(self.qoptwidget)):
            if self.qoptwidget[i]==wobj:
                l[i] = x.new
        self.analysis.set_option(l)

        if self.event_maskanalysis is not None:
            self.event_maskanalysis()
    
    def widget_observe(self):
        self.zoomslider.observe(self.event_zoomslider,"value")
        for qa,qb in self.qsliderwidget:
            try:
                qa.observe(self.event_slider,"value")
            except:
                pass
            try:
                qb.observe(self.event_slider,"value")
            except:
                pass
        for qa in self.qoptwidget:
            try:
                qa.observe(self.event_opt,"value")
            except:
                pass
        #self.diameter_slider_max.observe(self.event_diameter_max_sel,"value")
        #self.diameter_slider_min.observe(self.event_diameter_min_sel,"value")
        
    def widget_unobserve(self):
        try:
            self.zoomslider.unobserve(self.event_zoomslider,"value")
        except:
            pass
            
        for qa,qb in self.qsliderwidget:
            try:
                qa.unobserve(self.event_slider,"value")
            except:
                pass
            try:
                qb.unobserve(self.event_slider,"value")
            except:
                pass
                
        for qa in self.qoptwidget:
            try:
                qa.unobserve(self.event_opt,"value")
            except:
                pass
    
    def event_zoomslider(self,x):
        self.analysis.fig.set_dpi(100*x.new)
         
    def init_gui(self,canvasdraw=True):
        self.analysis(self.label,self.img)
        self.analysis.fig.set_visible(True)
        v1 = max(self.label.shape[0],self.label.shape[1])
        #v1 = np.ceil(v1*10)/10
        #self.analysis.set_min_max_diameter_select_range((v0,v1))
        val = self.analysis.get_min_max_quant_standard_select_range()
        for i in range(len(val)):
            a,b = val[i]
            if a=="img_s_max":a = v1
            if b=="img_s_max":b = v1
            val[i] = (a,b)
        #print(val)
        self.analysis.set_min_max_quant_select_range(val)
        
        self.widget_unobserve()

        for i in range(len(self.qsliderwidget)):
            for z in range(2):
                if self.qslidertyp[i] == "lin":
                    self.qsliderwidget[i][0].max = val[i][1]
                    self.qsliderwidget[i][0].min = val[i][0]
                    self.qsliderwidget[i][1].max = val[i][1]
                    self.qsliderwidget[i][1].min = val[i][0]
                else:
                    self.qsliderwidget[i][0].max = np.log10(val[i][1])
                    self.qsliderwidget[i][0].min = np.log10(val[i][0])
                    self.qsliderwidget[i][1].max = np.log10(val[i][1])
                    self.qsliderwidget[i][1].min = np.log10(val[i][0])
                
        
            self.qsliderwidget[i][0].value = val[i][0]
            self.qsliderwidget[i][1].value = val[i][1]

        val = self.analysis.get_standard_option()
        for i in range(len(self.qoptwidget)):
            self.qoptwidget[i].value = val[i]
        
        #self.diameter_slider_max.value = v1
        #self.diameter_slider_min.value = 3
        #self.diameter_slider.value = (v0,v1)
        self.widget_observe()
        if self.event_maskanalysis is not None:
            self.event_maskanalysis()

    def reset_gui(self):
        self.widget_unobserve()
        self.zoomslider.value = 1.5
        self.analysis.fig.set_visible(False)
        self.analysis.fig.set_dpi(150)
        self.label = None
        self.img = None
        self.widget_observe()
        #self.update()

    def get_posl(self):
        return self.analysis.get_posl()

    def get_config(self):
        if self.label is None:
            return None
        return {"shape_range":self.analysis.get_min_max_quant_select_range(),
                "shape_option":self.analysis.get_option()}


class PosAnalysisGUI(ipywidgets.VBox):
    def __init__(self):
        super().__init__()
        self.posl = None
        self.zoomslider = ipywidgets.FloatSlider(min=0.2,max=5,value=1.5,description="Plot zoom")
        self.resetzoom = ipywidgets.Button(description="Reset zoom")

        self.analysis = PosAnalysis()
        self.analysis.fig.set_visible(False)
        self.angle_standard_range = 20
        
        self.angle_slider = ipywidgets.FloatSlider(value=self.angle_standard_range,min=0,max=60,readout_format=".1f",description="min. angle [°]",step=0.01,continuous_update=False)
        self.angle_slider.layout.width="40%"
        #self.out = ipywidgets.Output()
        self.children = [ipywidgets.HBox([self.zoomslider,self.resetzoom]),self.angle_slider,self.analysis.fig.canvas]
        self.resetzoom.on_click(self.event_resetzoom)
        self.widget_observe()    

    def event_resetzoom(self,x):
        self.zoomslider.value = 1.5

    def __call__(self,posl,img,canvasdraw=True):
        self.posl = posl
        self.img = img
        self.init_gui(canvasdraw)

    def event_angle_sel(self,x):
        #with self.out:
        self.analysis.set_min_angle_select_range(x.new)
        self.analysis.update_canvas()

    def widget_observe(self):
        self.zoomslider.observe(self.event_zoomslider,"value")
        self.angle_slider.observe(self.event_angle_sel,"value")
    
    def widget_unobserve(self):
        try:
            self.zoomslider.unobserve(self.event_zoomslider,"value")
        except:
            pass
        try:
            self.angle_slider.unobserve(self.event_angle_sel,"value")
        except:
            pass
    
    def event_zoomslider(self,x):
        self.analysis.fig.set_dpi(100*x.new)
         
    def init_gui(self,canvasdraw=True):
        self.widget_unobserve()
        self.analysis(self.posl,self.img)
        self.analysis.set_min_angle_select_range(self.angle_standard_range)
        self.analysis.fig.set_visible(True)
        self.angle_slider.min = 0
        self.angle_slider.max = 60
        self.angle_slider.value = self.angle_standard_range
        self.widget_observe()
    
    def reset_gui(self):
        self.widget_unobserve()
        self.zoomslider.value = 1.5
        self.analysis.fig.set_visible(False)
        self.analysis.fig.set_dpi(150)
        self.posl = None
        self.img = None
        self.widget_observe
        
    def get_config(self):
        if self.posl is None:
            return None
        return {"angle_range":self.analysis.get_min_angle_select_range()}


class CustomTab(ipywidgets.VBox):
    def event(self,x):
        with self.out:
            for i,ele in enumerate(self.l):
                if ele==x:
                    self.children = [self.head,self.ch[i],self.out]
                    break
    
    def __init__(self,children,desc):
        super().__init__()
        self.ch = children
        
        self.l = [ipywidgets.Button(description=ele) for ele in desc]
        self.head = ipywidgets.HBox(self.l)
        self.out = ipywidgets.Output()
        self.children = [self.head,self.ch[0],self.out]
        self.layout = ipywidgets.Layout(border='solid 1px')

        for ele in self.l:
            ele.on_click(self.event)


class UButton(ipywidgets.VBox):
    def __init__(self,desc,func):
        super().__init__()
        self.but = ipywidgets.Button(description = desc)
        self.out = ipywidgets.Output()
        self.func = func
        
        self.but.on_click(self.clickb)
        self.children = [self.but,self.out]

    def clickb(self,x):
        self.out.clear_output()
        with self.out:
            from google.colab import files
            self.res = files.upload()
            if len(self.res.keys())>0:
                self.func([(ele,self.res[ele]) for ele in  self.res.keys()])
        self.out.clear_output()

class UNetGUI(ipywidgets.VBox):
    def __init__(self,models,input_img,tmp_folder,zip_folder,result_file,colab=False,change_img=True):
        super().__init__()
        self.change_img = change_img
        self.colab = colab
        self.tmp_folder = tmp_folder#"./tmp/"
        self.zip_folder = zip_folder#"zipfolder/"
        self.result_file = result_file#"result"
        self.modeldic = {ix:ele for ix,ele in enumerate(models+[("","Binarization",1)])}
        self.posanalysisgui = PosAnalysisGUI()
        self.maskanalysisgui = MaskAnalysisGUI(self.mask_analysis_update)
        
        self.predgui = UNetPredictionGUI(self.modeldic,self.pred_update)
        self.editorgui = ImageEditorGUI(self.editor_update,self.reset_gui)
        self.editorgui(input_img,False)

        self.res_button = ipywidgets.Button(description="Reset")
        self.res_button.on_click(self.reset_gui_button)
        if not colab:
            self.fileup = ipywidgets.FileUpload()
            self.fileup.observe(self.event_fileup_ipy,"value")
        else:
            self.fileup = UButton("Upload",self.event_fileup)

        if not colab:
            self.fileupbatch = ipywidgets.FileUpload(multiple=True,description="Upload images & Batch analysis & Download (after parameter setup via GUI)")
            self.fileupbatch.layout.width="60%"
            self.fileupbatch.observe(self.event_filebatch_ipy,"value")
    
            self.fileupvideo = ipywidgets.FileUpload(multiple=False,description="Upload video & Frame analysis & Download (after parameter setup via GUI)")
            self.fileupvideo.layout.width="60%"
            self.fileupvideo.observe(self.event_filevideo_ipy,"value")
        else:
            self.fileupbatch = UButton("Upload images & Batch analysis & Download (after parameter setup via GUI)",self.process_batch)
            self.fileupbatch.layout.width="60%"
            self.fileupbatch.but.layout.width="60%"
            
            self.fileupvideo = UButton("Upload video & Frame analysis & Download (after parameter setup via GUI)",self.process_video)
            self.fileupvideo.layout.width="60%"
            self.fileupvideo.but.layout.width="60%"
            

        if not colab:
            self.tab = ipywidgets.Tab([self.editorgui,self.predgui,self.maskanalysisgui,self.posanalysisgui])
            self.tab.set_title(0,"1) Image Editor")
            self.tab.set_title(1,"2) Prediction")
            self.tab.set_title(2,"3) Mask analysis")
            self.tab.set_title(3,"4) Position analysis")
        else:
            self.tab = CustomTab([self.editorgui,self.predgui,self.maskanalysisgui,self.posanalysisgui],
                                ["1) Image Editor","2) Prediction","3) Mask analysis","4) Position analysis"])
        
        self.outbatch = ipywidgets.Output()
        self.outvideo = ipywidgets.Output()
        self.clear_output()

        if change_img:
            self.children = [ipywidgets.HBox([self.fileup,self.res_button]),self.tab,
                             ipywidgets.HBox([self.fileupbatch,self.outbatch]),
                             ipywidgets.HBox([self.fileupvideo,self.outvideo])]
        else:
            self.children = [ipywidgets.HBox([self.res_button]),self.tab]

    def clear_output(self):
        with self.outbatch:
            self.outbatch.clear_output()
        with self.outvideo:
            self.outvideo.clear_output()

    def event_fileup_ipy(self,x):
        with self.outvideo:
            self.event_fileup([(ele.name,ele.content.tobytes()) for ele in x.new])

        
    def event_fileup(self,x):
        self.reset_gui()
        fn,bytes = x[0]#x.new[0].content.tobytes()
        
        try:
            img = np.array(Image.open(io.BytesIO(bytes)))
        except:
            fnv = self.tmp_folder+x.new[0].name
            with open(fnv,"wb") as f:
                f.write(bytes)
            import cv2
            cap = cv2.VideoCapture(fnv)
            ret,img = cap.read()
    
        self.editorgui(img)
        if self.colab:
            self.fileup.unobserve(self.event_fileup_ipy,"value")
            self.fileup.value = ()
            self.fileup.observe(self.event_fileup_ipy,"value")
    
    def event_filebatch_ipy(self,x):
        self.process_batch([(ele.name,(ele.content.tobytes())) for ele in x.new])#x.new)

    def event_filevideo_ipy(self,x):
        self.process_video([(ele.name,(ele.content.tobytes())) for ele in x.new])#x.new)

    def process_video(self,x):
        self.clear_output()
        with self.outvideo:
            print("Analysing ...")
        from IPython.display import FileLink,display
        #with self.outvideo:
        config = self.get_config()
        if not ((config["mask_analysis_editor"] is None) or (config["pos_analysis_editor"] is None)):
          #fnl = [(b,a) for a,b in fnl]
            name,bytes = x[0]
            #bytes = x[0].content.tobytes()
            fn_vid = self.tmp_folder+name#x[0].name
            with open(fn_vid,"wb") as f:
                f.write(bytes)
            if not self.colab:
              self.fileupvideo.unobserve(self.event_filevideo_ipy,"value")
              self.fileupvideo.value = ()
              self.fileupvideo.observe(self.event_filevideo_ipy,"value")
            
            with self.outvideo:
              file = get_video_analysis(fn_vid,config,self.tmp_folder+self.zip_folder,self.tmp_folder+self.result_file)
            import os
            os.remove(fn_vid)
            if file is not None:
                self.clear_output()
                with self.outvideo:
                    if not self.colab:
                      display(FileLink(file))
                    else:
                      from google.colab import files
                      files.download(file)

        else:
            if not self.colab:
              self.fileupvideo.unobserve(self.event_filevideo_ipy,"value")
              self.fileupvideo.value = ()
              self.fileupvideo.observe(self.event_filevideo_ipy,"value")
            self.clear_output()
            with self.outvideo:
                print("Set the parameters ! (min. click predict)")
           
    def process_batch(self,fnl):
        self.clear_output()
        with self.outbatch:
            print("Analysing ...")
        from IPython.display import FileLink,display
        config = self.get_config()
        if not ((config["mask_analysis_editor"] is None) or (config["pos_analysis_editor"] is None)):
            #fnl = [(np.array(Image.open(io.BytesIO(ele.content.tobytes()))),ele.name) for ele in fnl]
            fnl = [(np.array(Image.open(io.BytesIO(b))),a) for a,b in fnl]
            if not self.colab:
              self.fileupbatch.unobserve(self.event_filebatch_ipy,"value")
              self.fileupbatch.value = ()
              self.fileupbatch.observe(self.event_filebatch_ipy,"value")
            with self.outbatch:
                file = get_batch_analysis(fnl,config,self.tmp_folder+self.zip_folder,self.tmp_folder+self.result_file)
            if file is not None:
                self.clear_output()
                with self.outbatch:
                    if not self.colab:
                      display(FileLink(file))
                    else:
                      from google.colab import files
                      files.download(file)
        else:
            if not self.colab:
              self.fileupbatch.unobserve(self.event_filebatch_ipy,"value")
              self.fileupbatch.value = ()
              self.fileupbatch.observe(self.event_filebatch_ipy,"value")
            self.clear_output()
            with self.outbatch:
                print("Set the parameters ! (min. click predict)")
        
    def editor_update(self):
        self.predgui(self.editorgui.get_editimg())

    def pred_update(self):
        self.clear_output()
        self.maskanalysisgui(self.predgui.get_prediction(),self.editorgui.get_editimg())

    def reset_gui_button(self,x):
        self.reset_gui()

    def mask_analysis_update(self):
        self.posanalysisgui(self.maskanalysisgui.get_posl(),self.editorgui.get_editimg())
    
    def reset_gui(self):
        self.clear_output()
        self.predgui.reset_gui()
        self.editorgui.init_gui()
        self.maskanalysisgui.reset_gui()
        self.posanalysisgui.reset_gui()

    def get_config(self):
        return {"img_editor":self.editorgui.get_config(),
                "prediction_editor":self.predgui.get_config(),
                "mask_analysis_editor":self.maskanalysisgui.get_config(),
                "pos_analysis_editor":self.posanalysisgui.get_config()}