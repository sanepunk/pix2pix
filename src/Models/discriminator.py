from .utils import *

class Discriminator(nnx.Module):
	def __init__(self, in_features, kernel_size, maxpool_size, stride, padding, rngs: nnx.Rngs, training: bool = True):
		"""
		Initialize the Discriminator network with residual blocks and dimension reduction layers.
		Args:
			in_features (int): Number of input channels
			kernel_size (tuple): Size of convolutional kernels
			maxpool_size (tuple): Size of maxpooling windows
			stride (tuple): Stride for convolution operations
			padding (str): Padding type for convolutions
			rngs (nnx.Rngs): Random number generators for initialization
			training (bool): Whether the model is in training mode
		"""
		self.residual1 = ResidualBlock(in_features, 64, kernel_size, stride, padding, rngs, training)

		self.residual2 = ResidualBlock(64, 128, kernel_size, stride, padding, rngs, training)

		self.residual3 = ResidualBlock(128, 256, kernel_size, stride, padding, rngs, training)

		self.residual4 = ResidualBlock(256, 512, kernel_size, stride, padding, rngs, training)

		self.residual5 = ResidualBlock(512, 1024, kernel_size, stride, padding, rngs, training)

		self.dim_reduce_1 = nnx.Conv(1024, 128, (3, 3), strides = stride, padding=0, kernel_init=nnx.initializers.kaiming_normal(), rngs=rngs)
		self.dim_reduce_2 = nnx.Conv(128, 1, (4, 4), strides = (7, 7), padding=0, kernel_init=nnx.initializers.kaiming_normal(), rngs=rngs)


	def __call__(self, x, training: bool = True):
		"""
		Forward pass through the discriminator network.

		Args:
			x (Array): Input tensor containing the image
			training (bool): Whether the model is in training mode

		Returns:
			Array: Classification output (0-1) indicating real or fake
		"""
		x = nnx.leaky_relu(nnx.max_pool(self.residual1(x, training), (9, 9), (1, 1)))
		x = nnx.leaky_relu(nnx.max_pool(self.residual2(x, training), (9, 9), (1, 1)))
		x = nnx.leaky_relu(nnx.max_pool(self.residual3(x, training), (9, 9), (1, 1)))
		x = nnx.leaky_relu(nnx.max_pool(self.residual4(x, training), (9, 9), (1, 1)))
		x = nnx.leaky_relu(nnx.max_pool(self.residual5(x, training), (9, 9), (1, 1)))

		x = nnx.leaky_relu(self.dim_reduce_1(x))
		x = nnx.sigmoid(self.dim_reduce_2(x))

		return x

