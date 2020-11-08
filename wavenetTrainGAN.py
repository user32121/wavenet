#NOTE: this program is currently non functional


import tensorflow as tf
import numpy as np
import os

import wavenet, mp3Reader



#create model
generator = wavenet.buildModel(wavenet.QUANTIZATIONCHANNELS, wavenet.DILATIONS)
seqLength = generator.input.shape[1]
discriminator = wavenet.buildDiscriminator(wavenet.QUANTIZATIONCHANNELS, wavenet.DILATIONS)



#get sound file
frequency, stream = mp3Reader.read(wavenet.filePath, normalized=True, allow2Channels=False)

#use mu transform
stream = wavenet.mu_law_encode(stream, wavenet.QUANTIZATIONCHANNELS)

#convert to ranges
data = []
for i in range(0, len(stream) - seqLength - 1, 40000):
    data.append(stream[i:i + seqLength + 1])

sequences = tf.data.Dataset.from_tensor_slices(data)  #(None)
dataset = sequences.map(lambda chunk : (chunk[:-1], chunk[-1]))  #(None, (Tensor,int))
dataset = dataset.batch(wavenet.BATCHSIZE, drop_remainder = True).cache()  #(None, BATCHSIZE, (Tensor,int))
idat = iter(dataset)



#define losses and optimizers
crossEntropy = tf.losses.BinaryCrossentropy(from_logits=True)
def genLossFn(output, target=tf.ones_like): 
    return crossEntropy(target(output), output)

def discLossFn(real, fake, fakeTarget=tf.zeros_like): 
    return (crossEntropy(tf.ones_like(real), real) if real is not None else 0) + (crossEntropy(fakeTarget(fake), fake) if fake is not None else 0)

genOpt = tf.optimizers.Adam(1e-5)
discOpt = tf.optimizers.Adam(1e-5)

#checkpoint
ckpt = tf.train.CheckpointManager(tf.train.Checkpoint(generator=generator, genOpt=genOpt, discriminator=discriminator, discOpt=discOpt), wavenet.checkpointDir, 1)
ckpt.restore_or_initialize()
print("aaaaaaaaa\n\n\n")
print("loading ",tf.train.latest_checkpoint(wavenet.checkpointDir))

#weight change
genChange = wavenet.PrintWeightChange(generator)
discChange = wavenet.PrintWeightChange(discriminator)



for i in range(wavenet.EPOCHS):
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        #get next batch
        try:
            batch = next(idat)
        except StopIteration as stop:
            break

        genPred = generator(batch[0], training=True)

        
        realPred = discriminator(batch, training=True)
        fakePred = discriminator((batch[0], genPred), training=True)
        genLoss = genLossFn(fakePred, tf.ones_like)
        discLoss = discLossFn(realPred, fakePred, tf.zeros_like)
    
    genGrad = genTape.gradient(genLoss, generator.trainable_variables)
    discGrad = discTape.gradient(discLoss, discriminator.trainable_variables)

    genOpt.apply_gradients(zip(genGrad, generator.trainable_variables))
    discOpt.apply_gradients(zip(discGrad, discriminator.trainable_variables))

    if(i % 1) == 0:
        ckpt.save()
        print(i)
        print("Generator: loss:", genLoss.numpy(), end="")
        genChange.printChange(generator)
        print("Discriminator: loss:", discLoss.numpy(), end="")
        discChange.printChange(discriminator)
