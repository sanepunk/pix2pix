import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
from scipy.optimize import direct

from generator import Generator
from discriminator import Discriminator
import optax
from flax import nnx


class Pix2Pix(nnx.Module):
	def __init__(self, in_features, kernel_size, maxpool_size, stride, padding, rngs: nnx.Rngs, training: bool = True, path: str = None):
		if path:
			checkpoint = ocp.PyTreeCheckpointer()
			self.generator = checkpoint.restore(path)['generator']
			self.discriminator = checkpoint.restore(path)['discriminator']
			self.generator_optimizer = checkpoint.restore(path)['generator_optimizer']
			self.discriminator_optimizer = checkpoint.restore(path)['discriminator_optimizer']
		else:
			self.generator = Generator(in_features, kernel_size, maxpool_size, stride, padding, rngs, training)
			self.discriminator = Discriminator(in_features, kernel_size, maxpool_size, stride, padding, rngs, training)
			self.generator_optimizer = nnx.Optimizer(self.generator, optax.adam(1e-4))
			self.discriminator_optimizer = nnx.Optimizer(self.discriminator, optax.adam(1e-4))

	def __call__(self, x, training: bool = True):
		return self.generator(x, training), self.discriminator(x, training)

	def generate(self, x, training: bool = False):
		return self.generator(x, training)



