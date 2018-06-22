import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adamax, SGD
import os


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)

        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []

        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trial = trial.reshape((1, -1))
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)

            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


class GAN():
    def __init__(self, rows=3, cols=1875):
        self.signal_rows = rows
        self.signal_cols = cols
        # self.channels = 1
        # , self.channels)
        self.signal_shape = (self.signal_rows, self.signal_cols)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.2)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        signal = self.generator(z)

        # For the combined model I will only train the generator
        self.discriminator.trainable = False

        validity = self.discriminator(signal)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.directory = 'samples'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.signal_shape), activation='tanh'))
        model.add(Reshape(self.signal_shape))

        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        signal = model(noise)

        return Model(noise, signal)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.signal_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        signal = Input(shape=self.signal_shape)
        validity = model(signal)

        return Model(signal, validity)

    def train(self, dataset, epochs=1000, batch_size=128, sample_interval=100):

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        errors = []

        for epoch in range(epochs):

            idx = np.random.randint(0, dataset.shape[0], batch_size)
            signals = dataset[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_signals = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(signals, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            g_loss = self.combined.train_on_batch(noise, valid)

            errors.append([d_loss[0], g_loss])

            if epoch % sample_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
                      (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.sample_signal(epoch)

        plt.plot(errors)
        plt.xlabel('Epochs')
        plt.grid()
        plt.show()

        print(errors[-1])

    def sample_signal(self, epoch):

        num_signals = 273
        noise = np.random.normal(0, 1, (num_signals, self.latent_dim))
        gen_signal = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_signal = 0.5 * gen_signal + 0.5

        fig, axs = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        fig.set_size_inches(18.5, 5.5)

        # This is for index 0 and that is the channel C3
        axs.imshow(gen_signal[:, 0, :])

        fig.savefig("%s/%d.png" % (self.directory, epoch))
        plt.close()


datasetA1 = MotorImageryDataset()
trials, classes = datasetA1.get_trials_from_channels()
trials = np.concatenate([trials], axis=2)

fixed_trials = trials.reshape((-1, 3, 1875))
minft = fixed_trials.min()
maxft = fixed_trials.max()
fixed_trials = ((fixed_trials - minft)/(maxft - minft))

gan = GAN(3, 1875)
gan.train(fixed_trials, epochs=1000, batch_size=50)
