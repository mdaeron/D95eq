from typing import Callable
from numpy.typing import ArrayLike

def confidence_band(
	t: ArrayLike,
	fx: Callable,
	gx: Callable,
	p: float = 0.95,
):
	"""
	Return an (N,2) array of (x, y) vertices
	"""
	pass
