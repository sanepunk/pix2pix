from flax import nnx, linen as nn
import jax.numpy as jnp
import jax
from jax.random import PRNGKey


class ResidualBlock(nnx.Module):
	def __init__(self, in_features, out_features, kernel_size, stride, padding, rngs, training: bool = True):
		"""
        A residual block that consists of three convolutional layers with batch normalization and leaky ReLU activations.
        It uses skip connections to preserve spatial information and gradients.

        Parameters:
        - in_features: Number of input channels.
        - out_features: Number of output channels.
        - kernel_size: Size of the convolutional kernel.
        - stride: Stride value for the convolution operation.
        - padding: Padding type to maintain spatial dimensions.
        - rngs: Random number generator keys for parameter initialization.
        - training: Boolean indicating whether the model is in training mode.
        """

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
		"""
        Forward pass of the residual block.

        Parameters:
        - x: Input tensor.
        - training: Boolean indicating if the model is in training mode.

        Returns:
        - Output tensor after residual processing.
        """
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
		"""
        A downsampling module that applies a convolution, batch normalization, and a residual block.
        It reduces the spatial dimensions while preserving key features.

        Parameters:
        - in_features: Number of input channels.
        - out_features: Number of output channels.
        - kernel_size: Size of the convolutional kernel.
        - max_pooling_size: Size of the max pooling operation.
        - stride: Stride value for the convolution operation.
        - padding: Padding type to maintain spatial dimensions.
        - rngs: Random number generator keys for parameter initialization.
        - training: Boolean indicating whether the model is in training mode.
        """
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
		"""
        Forward pass of the downsampling module.

        Parameters:
        - x: Input tensor.
        - training: Boolean indicating if the model is in training mode.

        Returns:
        - Tuple containing the max pooled output and the residual block output.
        """
		x = self.conv(x)
		x = self.batch_norm(x, use_running_average=not training)
		x_residual_1 = jax.nn.leaky_relu(x)

		x_residual_2 = self.residual_block(x_residual_1, training)

		return nnx.max_pool(x_residual_2, window_shape=self.max_pool), x_residual_2


class UpSample(nnx.Module):
	def __init__(self, in_features, out_features, kernel_size, stride, padding, rngs, training: bool = True):
		"""
		An upsampling module that resizes the input and combines it with skip connections. Uses convolution, batch normalization, and residual processing for feature enhancement.
		Parameters:
		- in_features: Number of input channels.
		- out_features: Number of output channels.
		- kernel_size: Size of the convolutional kernel.
		- stride: Stride value for the convolution operation.
		- padding: Padding type to maintain spatial dimensions.
		- rngs: Random number generator keys for parameter initialization.
		- training: Boolean indicating whether the model is in training mode.
		"""
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
		"""
		Forward pass of the upsampling module.

		Parameters:
		- x: Input tensor to be upsampled
		- skip_connection: Tensor from the encoder path to be concatenated
		- training: Boolean indicating if the model is in training mode

		Returns:
		- Processed tensor after upsampling and feature fusion
		"""
		x = jax.image.resize(x, (skip_connection.shape), method='nearest')
		x_concatenated = jnp.concatenate([x, skip_connection], axis=-1)

		x = self.conv(x_concatenated)
		x = self.batch_norm(x, use_running_average=not training)
		x = jax.nn.leaky_relu(x)

		x = self.residual_block(x, training)

		return x


def critic_loss(critic: nnx.Module, generator: nnx.Module, real_output, fake_input_then_output, gp_weight, rng):
	fake_output = generator(fake_input_then_output, training=True)
	real_score = critic(real_output, training=True)
	fake_score = critic(fake_output, training=True)

	wasserstein_loss = fake_score.mean() - real_score.mean()
	epsilon = jax.random.uniform(rng, (real_output.shape[0], 1))
	interpolates = epsilon * real_output + (1 - epsilon) * fake_output
	interpolate_score = critic(interpolates, training=True)
	gradients = jax.grad(lambda x: critic(x, training=True).sum())(interpolates)
	gradient_norm = jnp.linalg.norm(gradients, axis=-1)
	gradient_penalty = gp_weight * jnp.mean((gradient_norm - 1) ** 2).mean()
	return wasserstein_loss + gradient_penalty

def generator_loss(critic: nnx.Module, generator: nnx.Module, fake_input_then_output):
	fake_output = generator(fake_input_then_output, training=True)
	fake_score = critic(fake_output, training=True)
	return -fake_score.mean()

def train_step(critic: nnx.Module, generator: nnx.Module, critic_optimizer: nnx.Optimizer, generator_optimizer: nnx.Optimizer, real_output, fake_input_then_output, gp_weight, rng):
	critic_loss_val, critic_grads = nnx.value_and_grad(critic_loss)(critic_optimizer.model, generator_optimizer.model, real_output, fake_input_then_output, gp_weight, rng)
	critic_optimizer.update(critic_grads)

	generator_loss_val, generator_grads = jax.value_and_grad(generator_loss)(critic_optimizer.model, generator_optimizer.model, fake_input_then_output)
	generator_optimizer.update(generator_grads)

	return critic_loss_val, generator_loss_val



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
