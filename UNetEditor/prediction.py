import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
is_gpu_available = lambda : len(tf.config.list_physical_devices('GPU'))
#if is_gpu_available(): print("GPU available:",tf.config.list_physical_devices('GPU'))
# Basic activation layer
class MishLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.activations.mish(x)
        
class SkyUNetModel:
    class MishLayer(tf.keras.layers.Layer):
            def call(self, x):
                return tf.keras.activations.mish(x)
                
    def __init__(self,param):
        self.param = param

    def __repr__(self):
        return f"<SkyUNet: {self.param['name']}>"

    # Basic Convolution Block
    def get_conv_block(self,x, n_channels):
        x = tf.keras.layers.Conv2D(n_channels, kernel_size=self.param["kernel_size"],kernel_initializer=self.param["kernel_initialization"],padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x) 
        x = self.MishLayer()(x)
        return x
    
    # Double Convolution Block used in "encoder" and "bottleneck"
    def get_double_conv_block(self, x, n_channels):
        x = self.get_conv_block(x,n_channels)
        x = self.get_conv_block(x,n_channels)
        return x
    
    # Downsample block for feature extraction (encoder)
    def get_downsample_block(self, x, n_channels):
        f = self.get_double_conv_block(x, n_channels)
        p = tf.keras.layers.MaxPool2D(pool_size=(2,2))(f)
        p = tf.keras.layers.Dropout(self.param["dropout"])(p)
        return f, p
    
    # Upsample block for the decoder
    def get_upsample_block(self, x, conv_features, n_channels):
        x = tf.keras.layers.Conv2DTranspose(n_channels*self.param["upsample_channel_multiplier"], self.param["kernel_size"], strides=(2,2), padding='same')(x)
        x = tf.keras.layers.concatenate([x, conv_features])
        x = tf.keras.layers.Dropout(self.param["dropout"])(x)
        x = self.get_double_conv_block(x, n_channels)
        return x

    def get_model(self):
        input = tf.keras.layers.Input(shape=self.param["input_shape"]+(1,))
        next_input = input
        
        l_residual_con = []
        for i in range(self.param["n_depth"]):
            residual_con,next_input = self.get_downsample_block(next_input, (2**i)*self.param["filter_multiplier"])
            l_residual_con.append(residual_con)
    
        next_input = self.get_double_conv_block(next_input, (2**self.param["n_depth"])*self.param["filter_multiplier"])
    
        for i in range(self.param["n_depth"]):
            next_input = self.get_upsample_block(next_input, l_residual_con[self.param["n_depth"]-1-i], (2**(self.param["n_depth"]-1-i))*self.param["filter_multiplier"])
    
        output = tf.keras.layers.Conv2D(self.param["n_class"], (1,1), padding="same", activation = "softmax",dtype='float32')(next_input)    
        
        return tf.keras.Model(input, output, name=self.param["name"])

class SkyUNet:
    def __init__(self):
        self.model = None
        self.model_ver = 2
        self.fn_model = ""

    def set_model(self,fn_model,model_ver=2):
        if self.fn_model != fn_model:
            if fn_model != "":
                model = tf.keras.models.load_model(fn_model,compile=False,custom_objects={'MishLayer': MishLayer})
                if not is_gpu_available():
                    #create identical model, only with pure float_32 policy
                    nmodel = SkyUNetModel({"name":"unet","input_shape": (512,512), "n_class":3,"filter_multiplier":16,"n_depth":4,
                    "kernel_initialization":"he_normal","dropout":0.1,"kernel_size":(3,3),"upsample_channel_multiplier":8}).get_model()
                    nmodel.set_weights(model.weights)
                    model = nmodel
                self.model = model
            else:
                self.model = None
            self.model_ver = model_ver
            self.fn_model = fn_model

    def predict(self,x,batch_size = 5,normalize_255=False):
        if self.model is None:
            if normalize_255:
                x = x/255
            ylabel = (x>=0.5).astype(np.int8)
            ylabel[ylabel==1] = 2
            return ylabel
        
        if not is_gpu_available():
            batch_size = 1
        #print(len(x))
        n = int(np.ceil(len(x)/batch_size))
        lix = [np.array(range(j*batch_size,min((j+1)*batch_size,len(x)))) for j in range(n)]
        ylabel = np.zeros(x.shape,dtype=np.uint8)
        progbar = tf.keras.utils.Progbar(n)
        for i in range(n):            
            progbar.update(i)
            input = x[lix[i]]
            if normalize_255:
                input = input/255
            ylabel[lix[i]] = self.model.predict(input,verbose=False).argmax(-1)
        progbar.update(n,finalize=True)
        if self.model_ver>1:
            #Swap class index of 1 and 2, since for model 2023 the class indeces are (skyrmion:0, background:1, defects:2) and the functions are written for class indeces (skyrmion:0, defects:1, background:2)
            ylabel[ylabel==1] = 5
            ylabel[ylabel==2] = 1
            ylabel[ylabel==5] = 2
        return ylabel

    def __call__(self,img,batch_size=5):
        #split image in 512x512 tiles
        oshape = True
        if len(img.shape)==3:
            oshape = False
        if oshape:
            img = img[np.newaxis]
            
        size_img,sizey,sizex = img.shape
        lix = [((j*512,min((j+1)*512,sizey)),(i*512,min((i+1)*512,sizex))) for j in range(int(np.ceil(sizey/512))) for i in range(int(np.ceil(sizex/512)))]
        #print("->lix ",len(lix))
        limgarray = []
        for iximg in range(size_img):
            for ele in lix:
                pimg = img[iximg,ele[0][0]:ele[0][1],ele[1][0]:ele[1][1]]
                nimg = np.ones((512,512))
                nimg[:min(512,pimg.shape[0]),:min(512,pimg.shape[1])] = pimg
                limgarray.append(nimg)
        limgarray = np.array(limgarray)
        #Predict label
        lpredict = self.predict(limgarray,batch_size)
        
        #reconstruct full image from tiles
        pred_label = np.zeros((size_img,sizey,sizex),dtype=lpredict.dtype)
        for iximg in range(size_img):
            ix0 = iximg*len(lix)
            for i,ele in enumerate(lix):
                pred_label[iximg,ele[0][0]:ele[0][1],ele[1][0]:ele[1][1]] = lpredict[i+ix0,:ele[0][1]-ele[0][0],:ele[1][1]-ele[1][0]]
        if oshape:
            return pred_label[0]        
        return pred_label

    @staticmethod
    def trafo_channel_to_rgb(I):
        basis = np.array([[255,0,0],[0,255,0],[0,0,255]],dtype=np.uint8)
        return basis[I]

    @staticmethod
    def trafo_rgb_to_channel(I):
        Q = np.zeros((I.shape[0],I.shape[1]),dtype=np.uint8)
        R,G,B = I[:,:,0],I[:,:,1],I[:,:,2]
        skyrmion_mask = (R>=128)&(G<128)&(B<128)
        defect_mask = (R<128)&(G>=128)&(B<128)
        bck_mask = ~(skyrmion_mask|defect_mask)
        Q[skyrmion_mask] = 0
        Q[defect_mask] = 1
        Q[bck_mask] = 2
        return Q