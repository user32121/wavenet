import tensorflow as tf
import numpy as np
import typing

# basic wavenet architecture has a
# dilated (casual) convolution (1) residual (2) network with 
# gated activations (3) and 
# skip connections (4) to output.

# 1. dilated casual convolutions: 
# dilation - skipping every few inputs in a convolution to allow for larger coverage
# casual - only looking at data from the past
#
# 2. residual network: input is carried over to next layer
#
# 3. gated activation units: z = tanh (W_f_i ∗ x) ⊙ σ (W_g_i ∗ x),
# x - input
# z - output
# ∗ - convolution
# ⊙ - element-wise multiplication
# σ - sigmoid function
# W - weights
# f/g - filter or gate
# i - index
# see diagram
#
# 4. skip connections: layer outputs are sent to next layer but also "skipped" to model output
# skipped values are processed more and determine model's output

#note special non-linear transformation on input/output data so we can use a range of [0,255] while preserving sound quality



folder = "rain/"
filePath = folder+"rain.mp3"
checkpointDir = "./"+folder+"training_checkpoints"

QUANTIZATIONCHANNELS = 256
DILATIONS = [1,2,4,8,16,32,64,128,256,512,
             1,2,4,8,16,32,64,128,256,512]

EPOCHS = 100
BATCHSIZE = 32



def mu_law_encode(audio, quantizationChannels):
    '''Quantizes waveform amplitudes.
    :param audio: the raw audio values in range [-1,1]
    :param quantizationChannels: the number of channels
    '''
    # y = sign(x) * ln(1 + mu * |x|) / ln(mu)
    # output = (y + 1) / 2 * mu + 0.5
    
    mu = float(quantizationChannels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)
    
def mu_law_decode(output, quantizationChannels):
    '''Recovers waveform from quantized values.'''
    # y = 2x/mu - 1
    # 

    mu = quantizationChannels - 1
    # Map values back to [-1, 1].
    signal = 2 * (output.astype(np.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude



def buildModel(quantizationChannels:int, blockDilations:typing.List[int]) -> tf.keras.Model:
    '''Creates a wavenet model
    
    :param quantizationChannels: an int representing the number of unique inputs there will be for embedding
    :param blockDilations: an array of dilation values assigned to the dilation blocks

    :return: a tensorflow model
    '''
    #input length allows there to be 1 timestep after all the convolutions
    inputs = tf.keras.layers.Input(shape=1 + sum(blockDilations))

    #add residual blocks
    layerInput = tf.keras.layers.Embedding(quantizationChannels, 256)(inputs)
    skipOut = []
    for i,dilation in enumerate(blockDilations):
        #gated activation unit
        convFilter = tf.keras.layers.Conv1D(32, 2, dilation_rate=dilation, activation="tanh", name="conv1d_filter_"+str(i))(layerInput)
        convGate = tf.keras.layers.Conv1D(32, 2, dilation_rate=dilation, activation="sigmoid", name="conv1d_gate_"+str(i))(layerInput)
        GAUOut = convFilter * convGate

        #skip connection
        skip1x1 = tf.keras.layers.Conv1D(quantizationChannels, 1, name="skip1x1_"+str(i))(GAUOut)
        #slice skip1x1 so it has 1 time step
        skip1x1 = tf.slice(skip1x1, [0, skip1x1.shape[1]-1, 0], [-1, 1, -1], name="slice_skip_"+str(i))
        skipOut.append(skip1x1)

        #residual
        out1x1 = tf.keras.layers.Conv1D(256, 1, name="out1x1_"+str(i))(GAUOut)
        #slice layerInput so it's the same shape as GAUOut
        layerInput = tf.slice(layerInput, [0, layerInput.shape[1] - out1x1.shape[1], 0], [-1, out1x1.shape[1], -1], name="slice_res_"+str(i))
        layerInput = tf.keras.layers.Add(name="residual_"+str(i))([out1x1, layerInput])
        
    #sum, relu, conv, (relu), conv of skipOut
    outputs = tf.keras.layers.LeakyReLU()(
        tf.keras.layers.Dense(256)(
            tf.keras.layers.LeakyReLU()(
                tf.keras.layers.Dense(256)(
                    tf.keras.layers.LeakyReLU()(
                        sum(skipOut))))))    

    #create model
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="wavenet")

def buildDiscriminator(quantizationChannels:int, blockDilations:typing.List[int]) -> tf.keras.Model:
    '''Creates a wavenet discrminator model, which is a normal wavenet modified to output a true/false prediction
    
    :param quantizationChannels: an int representing the number of unique inputs there will be for embedding
    :param blockDilations: an array of dilation values assigned to the dilation blocks

    :return: a tensorflow model
    '''
    #input length allows there to be 1 timestep after all the convolutions
    genInps = tf.keras.layers.Input(shape=1 + sum(blockDilations))
    genPred = tf.keras.layers.Input(shape=1)
    inputs = tf.concat([genInps[:,1:], genPred], axis=-1)

    #add residual blocks
    layerInput = tf.keras.layers.Embedding(quantizationChannels, 256)(inputs)
    skipOut = []
    for i,dilation in enumerate(blockDilations):
        #gated activation unit
        convFilter = tf.keras.layers.Conv1D(32, 2, dilation_rate=dilation, activation="tanh", name="conv1d_filter_"+str(i))(layerInput)
        convGate = tf.keras.layers.Conv1D(32, 2, dilation_rate=dilation, activation="sigmoid", name="conv1d_gate_"+str(i))(layerInput)
        GAUOut = convFilter * convGate

        #skip connection
        skip1x1 = tf.keras.layers.Conv1D(quantizationChannels, 1, name="skip1x1_"+str(i))(GAUOut)
        #slice skip1x1 so it has 1 time step
        skip1x1 = tf.slice(skip1x1, [0, skip1x1.shape[1]-1, 0], [-1, 1, -1], name="slice_skip_"+str(i))
        skipOut.append(skip1x1)

        #residual
        out1x1 = tf.keras.layers.Conv1D(256, 1, name="out1x1_"+str(i))(GAUOut)
        #slice layerInput so it's the same shape as GAUOut
        layerInput = tf.slice(layerInput, [0, layerInput.shape[1] - out1x1.shape[1], 0], [-1, out1x1.shape[1], -1], name="slice_res_"+str(i))
        layerInput = tf.keras.layers.Add(name="residual_"+str(i))([out1x1, layerInput])
        
    #sum, relu, conv, (relu), conv of skipOut
    outputs = tf.keras.layers.LeakyReLU()(
        tf.keras.layers.Dense(256)(
            tf.keras.layers.LeakyReLU()(
                tf.keras.layers.Dense(1)(
                    tf.keras.layers.LeakyReLU()(
                        sum(skipOut))))))    

    #create model
    return tf.keras.Model(inputs=(genInps, genPred), outputs=outputs, name="wavenet")



class PrintWeightChange(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.initialWeights = [tf.Variable(layer) for layer in model.weights]

    def on_epoch_end(self, epoch, logs=None):
        dif = 0
        for layOld,layNew in zip(self.initialWeights,self.model.weights):
            for wOld,wNew in zip(np.nditer(layOld.numpy()),np.nditer(layNew.numpy())):
                dif += abs(wOld - wNew)
        print('weights changed: ', dif)



if(__name__=="__main__"):
    model = buildModel(QUANTIZATIONCHANNELS, DILATIONS)
    model.summary()
    print(model.input)
    print(model.output)
