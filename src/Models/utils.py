from flax import nnx, linen as nn
import jax.numpy as jnp
import jax
from jax.random import PRNGKey


class ResidualBlock(nnx.Module):
	def __init__(self, in_features, out_features, kernel_size, stride, padding, rngs, training: bool = True):
		self.conv1 = nnx.Conv(
			in_features=in_features,
			out_features=out_features,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

		self.batch_norm1 = nnx.BatchNorm(
			num_features=out_features,
			use_running_average=not training,
			rngs=rngs,

		)

		self.conv2 = nnx.Conv(
			in_features=out_features,
			out_features=out_features,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

		self.batch_norm2 = nnx.BatchNorm(
			num_features=out_features,
			use_running_average=not training,
			rngs=rngs,
		)

		self.conv3 = nnx.Conv(
			in_features=out_features * 2,
			out_features=out_features,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

		self.batch_norm3 = nnx.BatchNorm(
			num_features=out_features,
			use_running_average=not training,
			rngs=rngs,
		)

	def __call__(self, x, training: bool = True):

		x = self.conv1(x)
		x = self.batch_norm1(x, use_running_average=not training)
		skip_connection_1  = jax.nn.leaky_relu(x)

		x = self.conv2(skip_connection_1)
		x = self.batch_norm2(x, use_running_average=not training)
		skip_connection_2 = jax.nn.leaky_relu(x)

		x = self.conv3(jnp.concatenate([skip_connection_1, skip_connection_2], axis=-1))
		x = self.batch_norm3(x, use_running_average=not training)
		return jax.nn.leaky_relu(x)


class DownSample(nnx.Module):
	def __init__(self, in_features, out_features, kernel_size, max_pooling_size, stride, padding, rngs, training: bool = True):
		self.conv = nnx.Conv(
			in_features=in_features,
			out_features=out_features,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

		self.max_pool = max_pooling_size
		self.batch_norm = nnx.BatchNorm(
			num_features=out_features,
			use_running_average=not training,
			rngs=rngs,
		)

		self.residual_block = ResidualBlock(out_features, out_features, kernel_size, stride, padding, rngs, training)

	def __call__(self, x, training: bool = True):
		x = self.conv(x)
		x = self.batch_norm(x, use_running_average=not training)
		x_residual_1 = jax.nn.leaky_relu(x)

		x_residual_2 = self.residual_block(x_residual_1, training)

		return nnx.max_pool(x_residual_2, window_shape=self.max_pool), x_residual_2


class UpSample(nnx.Module):
	def __init__(self, in_features, out_features, kernel_size, stride, padding, rngs, training: bool = True):
		self.conv = nnx.Conv(
			in_features=in_features,
			out_features=out_features,
			kernel_size=kernel_size,
			strides=stride,
			padding=padding,
			kernel_init=nnx.initializers.kaiming_normal(),
			rngs=rngs,
		)

		self.batch_norm = nnx.BatchNorm(
			num_features=out_features,
			use_running_average=False,
			rngs=rngs,
		)

		self.residual_block = ResidualBlock(out_features, out_features, kernel_size, stride, padding, rngs, training)

	def __call__(self, x, skip_connection, training: bool = True):
		x = jax.image.resize(x, (skip_connection.shape), method='nearest')
		x_concatenated = jnp.concatenate([x, skip_connection], axis=-1)

		x = self.conv(x_concatenated)
		x = self.batch_norm(x, use_running_average=not training)
		x = jax.nn.leaky_relu(x)

		x = self.residual_block(x, training)

		return x


# class Residue(nnx.Module):
# 	def __init__(self, in_features, out_features, kernel_size, stride, padding, rngs: nnx.Rngs, training: bool = True):
# 		self.residual = ResidualBlock(in_features, out_features, kernel_size, stride, padding, rngs, training)
# 		self.conv = nnx.Conv(out_features, out_features, kernel_size=(3, 3), strides=(1, 1), kernel_init=nnx.initializers.kaiming_normal(), rngs=rngs)
# 		self.batchnorm = nnx.BatchNorm(out_features, use_running_average=not training, rngs=rngs)
#
# 	def __call__(self, x, training: bool = True):
# 		x = self.residual(x, training)
# 		x = self.conv(x)
# 		x = self.batchnorm(x, use_running_average=not training)
# 		return x
