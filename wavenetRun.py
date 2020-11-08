import tensorflow as tf
import numpy as np
import time

import wavenet, mp3Reader



#create model
model = wavenet.buildModel(wavenet.QUANTIZATIONCHANNELS, wavenet.DILATIONS)
seqLength = model.input.shape[1]



#get sound file
frequency, stream = mp3Reader.read(wavenet.filePath, normalized=True, allow2Channels=False)
#use mu transform
stream = wavenet.mu_law_encode(stream, wavenet.QUANTIZATIONCHANNELS)
#extract last part of audio
stream = stream[-seqLength:]

print("aaaaaaaaa\n\n\n")



if(tf.train.latest_checkpoint(wavenet.checkpointDir)):
    model.load_weights(tf.train.latest_checkpoint(wavenet.checkpointDir))
print("loading ",tf.train.latest_checkpoint(wavenet.checkpointDir))



#generate audio
NUMSTOGENERATE = 10000
numsGenerated = [] #store output



t0 = time.time()
for i in range(NUMSTOGENERATE):
    output = model(tf.expand_dims(stream, 0))
    value = tf.argmax(output.numpy()[0][0])  #get index
    numsGenerated.append(stream[0])  #get first item from stream
    stream[:-1] = stream[1:]  #add value to end of stream
    stream[-1] = value

    if(time.time()-t0 > 5):
        print("progress: ",i/NUMSTOGENERATE,", value: ",numsGenerated[-1])
        t0 = time.time()
numsGenerated.extend(stream)

#mu decode
numsGenerated = wavenet.mu_law_decode(np.array(numsGenerated), wavenet.QUANTIZATIONCHANNELS)

#save to file
audio = mp3Reader.write(wavenet.folder+"output.mp3", mp3Reader.read(wavenet.filePath)[0], numsGenerated, normalized=True)
print("generated result in "+wavenet.folder+"output.mp3")
