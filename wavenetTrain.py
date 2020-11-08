import tensorflow as tf
import numpy as np
import os

import wavenet, mp3Reader



#create model
model = wavenet.buildModel(wavenet.QUANTIZATIONCHANNELS, wavenet.DILATIONS)
seqLength = model.input.shape[1]



print("loading "+wavenet.filePath)

#get sound file
frequency, stream = mp3Reader.read(wavenet.filePath, normalized=True, allow2Channels=False)

#use mu transform
stream = wavenet.mu_law_encode(stream, wavenet.QUANTIZATIONCHANNELS)

#convert to ranges
data = []
for i in range(0, len(stream) - seqLength - 1, 1000):
    data.append(stream[i:i + seqLength + 1])

sequences = tf.data.Dataset.from_tensor_slices(data)  #(None)
# sequences = charDataset.batch(seqLength+1, True)  #(None, seqLength+1)
dataset = sequences.map(lambda chunk : (chunk[:-1], chunk[-1]))  #(None, Tuple<Tensor,int>)
dataset = dataset.batch(wavenet.BATCHSIZE, drop_remainder = True).cache()  #(None, BATCHSIZE, Tuple<Tensor,int>)

print("aaaaaaaaa\n\n\n")



if(tf.train.latest_checkpoint(wavenet.checkpointDir)):
    model.load_weights(tf.train.latest_checkpoint(wavenet.checkpointDir))
print("loading ",tf.train.latest_checkpoint(wavenet.checkpointDir))

model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

checkpointPrefix = os.path.join(wavenet.checkpointDir,"ckpt")
checkpoint_callBack = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPrefix, save_weights_only=True) #checkpoint path

history = model.fit(dataset, epochs=wavenet.EPOCHS, callbacks=[checkpoint_callBack, wavenet.PrintWeightChange(model)])    #train the model
