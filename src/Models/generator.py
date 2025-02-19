from .utils import *

class Generator(nnx.Module):
	def __init__(self, in_features, kernel_size, maxpool_size, stride, padding, rngs, training: bool = True):
		"""
		Initialize the U-Net style Generator
		Args:
			in_features (int): Number of input channels
			kernel_size (tuple): Size of the convolutional kernel
			maxpool_size (tuple): Size of the maxpooling window
			stride (tuple): Stride for convolution operations
			padding (str): Padding type for convolutions ('SAME' or 'VALID')
			rngs (nnx.Rngs): Random number generators for initialization
			training (bool, optional): Whether in training mode. Defaults to True
		"""
		self.downsample_1 = DownSample(in_features, 64, kernel_size, maxpool_size, stride, padding, rngs, training)
		self.downsample_2 = DownSample(64, 128, kernel_size, maxpool_size, stride, padding, rngs, training)
		self.downsample_3 = DownSample(128, 256, kernel_size, maxpool_size, stride, padding, rngs, training)
		self.downsample_4 = DownSample(256, 512, kernel_size, maxpool_size, stride, padding, rngs, training)
		self.downsample_5 = DownSample(512, 1024, kernel_size, maxpool_size, stride, padding, rngs, training)

		self.upsample_1 = UpSample(1024, 512, kernel_size, stride, padding, rngs, training)
		self.upsample_2 = UpSample(512, 256, kernel_size, stride, padding, rngs, training)
		self.upsample_3 = UpSample(256, 128, kernel_size, stride, padding, rngs, training)
		self.upsample_4 = UpSample(128, 64, kernel_size, stride, padding, rngs, training)
		self.upsample_5 = UpSample(64, in_features, kernel_size, stride, padding, rngs, training)

		self.conv = nnx.Conv(
			in_features=64,
			out_features=in_features,
			kernel_size=(1, 1),
			strides=stride,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

	def __call__(self, x, training: bool = True):
		"""
		Forward pass through the generator
		Args:
			x (Array): Input tensor
			training (bool, optional): Whether in training mode. Defaults to True

		Returns:
			Array: Generated output after sigmoid activation
		"""
		x1_residual_connection, x1 = self.downsample_1(x, training)
		x2_residual_connection, x2 = self.downsample_2(x1_residual_connection, training)
		x3_residual_connection, x3 = self.downsample_3(x2_residual_connection, training)
		x4_residual_connection, x4 = self.downsample_4(x3_residual_connection, training)
		x5_residual_connection, x5 = self.downsample_5(x4_residual_connection, training)

		x6 = self.upsample_1(x5, x4, training)
		x7 = self.upsample_2(x6, x3, training)
		x8 = self.upsample_3(x7, x2, training)
		x9 = self.upsample_4(x8, x1, training)
		return nnx.sigmoid(self.conv(x9))

