


__all__ = ["Generator_v2", "Critic_v2"]

from tensorflow.keras import layers
import tensorflow as tf



class Generator_v2( object ):

  def __init__(self, generator_path=None):

      self.latent_dim = 100
      self.height     = 256
      self.width      = 256
      self.leaky_relu_alpha = 0.3
    
      if generator_path:
        self.model = tf.keras.models.load_model(generator_path)
      else:
        self.compile()
      


  #@tf.function
  def generate(self, nsamples):
    z = tf.random.normal( (nsamples, self.latent_dim) )
    return self.model.predict( z , batch_size=32)
  

  def compile(self):

      ip = layers.Input(shape=(self.latent_dim,))
      # Input (None, latent space (100?) )
      y = layers.Dense(units=16*16*32, input_shape=(self.latent_dim,))(ip)
      # Output (None, 64*3^2 )
      y = layers.Reshape(target_shape=(16,16, 32))(y)
      #y = layers.BatchNormalization()(y)
      #y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
      #y = layers.UpSampling1D()(y)
      # Output (None, 3^2*2, 64)
      y = layers.Conv2DTranspose(64, (8,8), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      #y = layers.UpSampling1D(size=2*2)(y)
      # Output (None, 3^2*2^3, 128)
      y = layers.Conv2DTranspose(64, (8,8), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      #y = layers.UpSampling1D(size=2*2)(y)
      # Output (None, 3^2*2^5, 256)
      y = layers.Conv2DTranspose(128, (8,8), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)

      y = layers.Conv2DTranspose(128, (8,8), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)

      # Output (None, 3^2*2^5, 64)
      out = layers.Conv2DTranspose(1, (8,8), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
      # Output (None, 3^2*2^5, 1)
      model = tf.keras.Model(ip, out)
      model.compile()
      model.summary()
      self.model = model



class Critic_v2( object ):

  def __init__(self, critic_path=None):

      self.height     = 256
      self.width      = 256
    
      if critic_path:
        self.model = tf.keras.models.load_model(critic_path)
      else:
        self.compile()

  def predict(self, samples, batch_size=1024):
    return self.model.predict(samples, verbose=1, batch_size=batch_size)

  def compile(self):

      ip = layers.Input(shape=( self.height,self.width,1))
      # TODO Add other normalization scheme as mentioned in the article
      # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
      y = layers.Conv2D(128, (10,10), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(self.height,self.width,1))(ip)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(64, (10,10), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(64, (10,10), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)

      y = layers.Conv2D(32, (10,10), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)


      # Output (None, 3^2*2, 128)
      y = layers.Flatten()(y)
      # Output (None, 3*256)
      out = layers.Dense(1, activation='linear')(y)
      # Output (None, 1)
      model = tf.keras.Model(ip, out)
      model.compile()
      model.summary()
      self.model = model
