"""
Test for clumped isotope equilibrium and estimate carbonate formation temperatures from dual clumped isotope measurements

.. include:: ../../docpages/install.md
.. include:: ../../docpages/cli.md

* * *
"""

from __future__ import annotations
from ._metadata import *
from ._tools import confidence_band

import sys
import numpy as _np
import ogls as _ogls
import uncertainties as _uc
import lmfit as _lmfit
import correldata as _cd
import typer as _typer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from matplotlib import pyplot as _ppl
	from matplotlib.patches import Ellipse as _Ellipse
	from matplotlib.patches import Polygon as _Polygon

from uncertainties import unumpy as _unp
from scipy.stats import chi2 as _chi2
from scipy.stats import norm as _norm
from scipy.linalg import eigh as _eigh
from scipy.linalg import cholesky as _cholesky
from scipy.optimize import fsolve as _fsolve
from numpy.typing import ArrayLike
from typing_extensions import Annotated as _Annotated
from typer import rich_utils as _rich_utils

from warnings import filterwarnings as _filterwarnings
_filterwarnings('ignore', category = FutureWarning, message = 'AffineScalarFunc')
_filterwarnings('ignore', category = RuntimeWarning, message = 'The iteration is not making good progress')


### Mathematical functions ###


def ufloat_compatible_interp(
	xi: (_cd.uarray | ArrayLike),
	yi: (_cd.uarray | ArrayLike),
	x: (float | _uc.UFloat | _cd.uarray | ArrayLike),
):
	"""
	Linear interpolation accepting UFloat values for all three input parameters.
	Only handles one interpolated value. For interpolated arrays, use `uarray_compatible_interp()`

	**Arguments**
	* `xi`: x-values defining the interpolated function
	* `yi`: y-values defining the interpolated function
	* `x`: x-value of the interpolation point

	Returns y-value of the interpolation point, either as a float or a UFloat.
	"""
	xn = x.nominal_value if isinstance(x, _uc.UFloat) else float(x)
	idx = _np.searchsorted(xi, xn)
	idx = _np.clip(idx, 1, len(xi) - 1)

	x0 = xi[idx-1]
	x1 = xi[idx]
	y0 = yi[idx-1]
	y1 = yi[idx]

	t = (x - x0) / (x1 - x0)
	return y0 + t * (y1 - y0)


def uarray_compatible_interp(xi, yi):
	"""
	Linear interpolation accepting UFloat values for all three input parameters.

	**Arguments**
	* `xi`: x-values defining the interpolated function
	* `yi`: y-values defining the interpolated function

	Returns an interpolation function which returns arrays or uarrays of y-values.
	"""
	return _np.vectorize(
		lambda x: ufloat_compatible_interp(xi, yi, x)
	)


def transform_pdf_monotonic(f_inv, df_inv, mu_x, sigma_x, yi):
	"""
	Compute probability distribution function of Y = f(X)
	where X ~ Normal(mu_x, sigma_x) and f is monotonic,
	based on the change-of-variables formula:

		p[y=f(x)] = p[x=f_inv(y)] * d(f_inv)/dy

	Additionally, if f_inv returns UFloats, the PDF is convolved with that local
	source of uncertainty (assumed to be Gaussian) at each grid point.

	As currently implemented, requires `yi` to be an equally spaced array-like.

	**Arguments**
		f_inv:   inverse of f, may return UFloats
		df_inv:  derivative of f_inv, should return UFloats if f_inv does
		mu_x:    mean of X PDF
		sigma_x: std dev of X PDF
		yi:      regularly spaced grid of y values at which to evaluate the PDF

	**Returns:**
		pdf: normalized PDF evaluated at yi
	"""

	if not _np.allclose(_np.diff(yi), yi[1] - yi[0]):
		raise ValueError("yi must be regularly spaced")

	xi = f_inv(yi) # may be floats or ufloats, depending on f_inv

	try:
		xi_nom = xi.n
		sigma_xi = xi.s
		has_ufloats = True
	except AttributeError:
		xi_nom = xi
		has_ufloats = False

	# Jacobian weights (account for irregular xi spacing)
	try:
		df_inv_nom = df_inv(yi).n
	except AttributeError:
		df_inv_nom = df_inv(yi)

	w_i = _norm.pdf(xi_nom, loc = mu_x, scale = sigma_x) * _np.abs(df_inv_nom)

	if not has_ufloats:
		return w_i / (_np.trapezoid(w_i, yi))

	# Propagate sigma from x-space to y-space via Jacobian: sigma_y = sigma_x / abs( dx/dy )
	sigma_yi = sigma_xi / _np.abs(df_inv_nom)

	# Convolution of Gaussians: each grid point j contributes N(yi; yj, σ_yj²) scaled by w_j
	gaussians = _norm.pdf(
		yi[:, None],
		loc = yi[None, :],
		scale = sigma_yi[None, :]
	) # NOTE: nice syntax to reshape ndarrays, perhaps use this in D4x_calib_function?

	pdf = (gaussians * w_i[None, :]).sum(axis = 1)

	return pdf / (_np.trapezoid(pdf, yi))


#### Calibration variables and functions ####


_D47_approx_calib_coefs = [0.159502986, 38588.1545] # computed from code in comments below
# from D47calib import OGLS23 as _OGLS23
# from D47calib import D47calib as _D47calib
#
# _D47_approx = _D47calib(
# 	samples = _OGLS23.samples,
# 	T = _OGLS23.T,
# 	sT = _OGLS23.sT,
# 	D47 = _OGLS23.D47,
# 	sD47 = _OGLS23.sD47,
# 	degrees = [0,2],
# )
# _D47_approx_calib_coefs = [_D47_approx.bfp['a0'], _D47_approx.bfp['a2']]


def _compute_D48_calib_coefficients(reprocess = False):
	"""
	Based on Fiebig et al. (2021, 2024)
	"""

	# D64 predictions
	a1 =  6.002
	a2 = -1.299e4
	a3 =  8.996e6
	a4 = -7.423e8

	if reprocess:

		# M. Bernecker, pers. comm.
		# after Fiebig et al. (2024) 10.1016/j.chemgeo.2024.122382
		datastr = '''
	    Sample,    D48, SE_D48,      T, SE_T, correl_T
	     LGB-2, 0.2606, 0.0103,    7.9,  0.2, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
	    DHC2-8, 0.2335, 0.0066,   33.7,  0.2, 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.
	     DVH-2, 0.2484, 0.0105,   33.7,  0.2, 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.
	     CA120, 0.1715, 0.0154,  120.0,   2., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.
	     CA170, 0.1621, 0.0142,  170.0,   2., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.
	     CA200, 0.1561, 0.0134,  200.0,   2., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.
	    CA250A, 0.1449, 0.0146,  250.0,   2., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.
	    CA250B, 0.1301, 0.0134,  250.0,   2., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.
	     CM351, 0.1220, 0.0073, 726.85,  10., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.
	ETH-1-1100, 0.1161, 0.0091, 1100.0,  10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.
	ETH-2-1100, 0.1225, 0.0070, 1100.0,  10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.
	'''[1:-2]

		data = _cd.read_str(datastr)
		T, D48 = data['T'], data['D48']


		D64_predicted = (
			a1 / (273.15 + T)
			+ a2 / (273.15 + T)**2
			+ a3 / (273.15 + T)**3
			+ a4 / (273.15 + T)**4
		)

		# affine regression of the form D48 = b0 + b1 * D64_theory
		R = _ogls.Polynomial(
			X = D64_predicted.n,
			sX = D64_predicted.covar,
			Y = D48.n,
			sY = D48.covar,
			degrees = [0,1],
		)

		R.regress(overdispersion_scaling = True)
		b0, b1 = _uc.correlated_values(R.bfp.values(), R.bfp_CM)
# 		print(_cd.data_string(dict(affine_coefs = _cd.uarray([b0, b1]))))

	else:

		# M. Bernecker, pers. comm.
		# after Fiebig et al. (2024) 10.1016/j.chemgeo.2024.122382
		# Caution: because Fiebig et al. ignored T uncertainties, these
		# coefficeients have smaller uncertainties than those computed above.
		b0, b1 = _uc.correlated_values(
			[
				0.12135157920099604,
				1.0379702801201238,
			], [
				[ 7.39697438e-06, -6.90467053e-05],
				[-6.90467053e-05,  1.46002771e-03],
			],
		)

	a0 = b0
	a1 *= b1
	a2 *= b1
	a3 *= b1
	a4 *= b1

	return _cd.uarray([a0, a1, a2, a3, a4])


def D4x_calib_function(
	T: (float | _uc.UFloat | _cd.uarray | ArrayLike),
	coefs: _cd.uarray,
	return_without_uncertainties: bool = False,
	ignore_calib_uncertainties: bool = False,
) -> (float | _uc.UFloat | _cd.uarray | ArrayLike):
	"""
	**Arguments**
	* `T`: temperature(s) for which to compute Δ<sub>4x</sub>
	* `return_without_uncertainties`: if `True`, returns Δ<sub>4x</sub> values without error propagation of any kind
	* `ignore_calib_uncertainties`: whether to propagate calibration uncertainties

	Returns equilibrium Δ<sub>4x</sub> value(s) corresponding to `T` value(s)
	"""
	degs = _np.arange(coefs.size)

	D4x = (
		_np.expand_dims(_cd.nv(coefs) if ignore_calib_uncertainties else coefs, 1) # shape = (coefs.size, 1)
		* _np.expand_dims((T+273.15)**-1, 0)                                       # shape = (1, T.size)
		** _np.expand_dims(degs, 1)                                                # shape = (coefs.size, 1)
	).sum(axis = 0 if isinstance(T, _np.ndarray) else None)

	if D4x.ndim == 0:
		return D4x.tolist().n if return_without_uncertainties else D4x.tolist()
	return D4x.n if return_without_uncertainties else D4x


def D4x_calib_derivative(
	T: (float | _uc.UFloat | _cd.uarray | ArrayLike),
	coefs: _cd.uarray,
	return_without_uncertainties: bool = False,
	ignore_calib_uncertainties: bool = False,
) -> (float | _uc.UFloat | _cd.uarray | ArrayLike):
	"""
	**Arguments**
	* `T`: temperature(s) for which to compute Δ<sub>4x</sub>
	* `return_without_uncertainties`: if `True`, returns D4x values without error propagation of any kind.
	* `ignore_calib_uncertainties`: whether to propagate calibration uncertainties.

	Returns d(D4x)/dT corresponding to `T` value(s)
	"""
	dcoefs = -_np.arange(len(coefs)) * coefs
	dcoefs = _cd.uarray((dcoefs[0], *dcoefs))
	return D4x_calib_function(
		T,
		dcoefs,
		return_without_uncertainties = return_without_uncertainties,
		ignore_calib_uncertainties = ignore_calib_uncertainties,
	)


#### Plotting functions ####


def conf_ellipse(
	X: (_cd.uarray | _np.ndarray | _uc.UFloat | float),
	Y: (_cd.uarray | _np.ndarray | _uc.UFloat | float) = None,
	p: float = 0.95,
	CM: (_np.ndarray | None) = None,
	Xse: (_np.ndarray | float | None) = None,
	Yse: (_np.ndarray | float | None) = None,
	ax: (_ppl.Axes | None) = None,
	**kwargs,
) -> tuple:
	"""
	Plot the joint *p*-level confidence ellipses for the elements of (X, Y)

	**Arguments**
	* `X`: x values
	* `Y`: y values
	* `p`: confidence level
	* `CM`: covariance matrix of (X, Y); not needed if X and Y are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html).
		or if (`Xse`, `Yse`) are specified.
	* `Xse`, `Yse`: SE of X and Y; not needed if X and Y are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html)
		or if `CM` is specified.
	* `ax`: which instance of `matplotlib.axes.Axes` to draw in; use current axes if `ax` = `None`.
	* `kwargs`: passed to `matplotlib.patches.Ellipse()`

	Returns a list of the `Ellipse` objects thus created.
	"""

	from matplotlib import pyplot as _ppl
	from matplotlib.patches import Ellipse as _Ellipse

	r2 = _chi2.ppf(p, 2)
	kwargs = dict(fc = 'None', ec = 'k', lw = 0.7) | kwargs

	if ax is None:
		ax = _ppl.gca()

	out = []

	for x, y in zip(
		*_cd.as_pair_of_uarrays(X, Y, CM = CM, Xse = Xse, Yse = Yse)
	):
		val, vec = _eigh(_uc.covariance_matrix((x, y)))
		width, height = 2 * (val[:, None] * r2)**0.5
		angle = _np.degrees(_np.arctan2(*vec[::-1, 0]))

		out.append(
			ax.add_patch(
				_Ellipse(
					xy = (x.n, y.n),
					width = width.item(),
					height = height.item(),
					angle = angle,
					**kwargs,
				)
			)
		)

	return (*out,)


### D95eq Engine implementation ###

class _Interpolation():
	pass

class Engine():
	"""
	Underlying engine to compute and plot nearest equilibrium temperatures and projected
	temperatures based on a consistent pair of Δ<sub>47</sub>, Δ<sub>48</sub> calibrations.
	"""

	# D47_calib_coefs from OGLS23 (D47calib v1.3.1)
	D47_calib_coefs = _cd.read_str('''
              coefs,                     SE,        correl,
0.17437754366432887,   4.911105567257293e-3,    1.        , -0.93797005,  0.8865771
 -18.14215245127414,      5.632326472234856,   -0.93797005,  1.        , -0.98994249
42.65722989162373e3,     1.27712751715908e3,    0.8865771 , -0.98994249,  1.
'''[1:-1])['coefs']
	"""
	Default (OGLS23) Δ<sub>47</sub> calibration coefficients based on [Daëron & Vermeesch (2024)](https://doi.org/10.1016/j.chemgeo.2023.121881)
	"""

	# D48_calib_coefs reprocessed from Fiebig et al. (2024):
	#
	# D48_calib_coefs = _compute_D48_calib_coefficients(reprocess = True)
	# print(_cd.data_string(
	# 	{'coefs': D48_calib_coefs},
	# 	float_format = 'z.12g',
	# 	correl_format = 'z.12f',
	# ))

	D48_calib_coefs = _cd.read_str('''
         coefs,         SE_coefs,    correl_coefs,                ,                ,                ,
0.121349237888, 0.00390048540724,  1.000000000000, -0.664181963395,  0.664181963395, -0.664181963395,  0.664181963395
 6.22931985613,    0.32896761459, -0.664181963395,  1.000000000000, -1.000000000000,  1.000000000000, -1.000000000000
 -13481.983494,    711.977559735,  0.664181963395, -1.000000000000,  1.000000000000, -1.000000000000,  1.000000000000
 9336714.66607,    493067.754224, -0.664181963395,  1.000000000000, -1.000000000000,  1.000000000000, -1.000000000000
-770413883.573,    40685214.9801,  0.664181963395, -1.000000000000,  1.000000000000, -1.000000000000,  1.000000000000
'''[1:-1])['coefs']
	"""
	Default Δ<sub>48</sub> calibration coefficients based on [Fiebig et al. (2024)](https://doi.org/10.1016/j.chemgeo.2024.122382)
	"""

	def __init__(
		self,
		D47_coefs: (_cd.uarray | ArrayLike | None) = None,
		D48_coefs: (_cd.uarray | ArrayLike | None) = None,
		Tmin_interp: float = -23.0,
		Tmax_interp: float = 1277.0,
		N_interp: float = 201,
	):
		"""
		**Arguments**
		* `D47_coefs`: `ndarray` or `uarray` of coefficients to use instead of default ones, ordered as (a0, a1, a2...)
		* `D48_coefs`: `ndarray` or `uarray` of coefficients to use instead of default ones, ordered as (a0, a1, a2...)
		* `Tmin_interp`: minimum temperature over which to interpolate for inverse function computations
		* `Tmax_interp`: maximum temperature over which to interpolate for inverse function computations
		* `N_interp`: number of points (equally-spaced in 1/T space) over which to interpolate for inverse function computations
		"""

		self.D47_coefs = Engine.D47_calib_coefs if D47_coefs is None else D47_coefs
		"""The Δ<sub>47</sub> calibration coefficients used by this `Engine` instance"""

		self.D48_coefs = Engine.D48_calib_coefs if D48_coefs is None else D48_coefs
		"""The Δ<sub>48</sub> calibration coefficients used by this `Engine` instance"""

		self.interp = _Interpolation()
		"""
		Holds equilibrium Δ<sub>47</sub> and Δ<sub>48</sub> values (ufloats) interpolated
		along an array of T values (regularly spaced increments of 1/T<sup>2</sup>).

		* `interp.T`: interpolation T values (floats) in regularly spaced increments of 1/T<sup>2</sup>
		* `interp.D47`: Equilibrium Δ<sub>47</sub> values (ufloats) interpolated along `interp.T`
		* `interp.D48`: Equilibrium Δ<sub>48</sub> values (ufloats) interpolated along `interp.T`
		* `interp.D47_no_calib_errors`: Equilibrium Δ<sub>47</sub> values (ufloats) interpolated along `interp.T`,
		ignoring calibration uncertainties
		* `interp.D48_no_calib_errors`: Equilibrium Δ<sub>48</sub> values (ufloats) interpolated along `interp.T`,
		ignoring calibration uncertainties
		"""

		self.interp.T = _np.linspace(
			(Tmax_interp+273.15)**-2,
			(Tmin_interp+273.15)**-2,
			N_interp,
		)**-0.5 - 273.15

		self.interp.D47 = self.D47_calib_function(
			self.interp.T,
			return_without_uncertainties = False,
			ignore_calib_uncertainties = False,
		)

		self.interp.D47_no_calib_errors = self.D47_calib_function(
			self.interp.T,
			return_without_uncertainties = False,
			ignore_calib_uncertainties = True,
		)

		self.interp.D48 = self.D48_calib_function(
			self.interp.T,
			return_without_uncertainties = False,
			ignore_calib_uncertainties = False,
		)

		self.interp.D48_no_calib_errors = self.D48_calib_function(
			self.interp.T,
			return_without_uncertainties = False,
			ignore_calib_uncertainties = True,
		)

		self.interp.D47u_as_function_of_D47n = uarray_compatible_interp(self.interp.D47.n, self.interp.D47)
		self.interp.D48u_as_function_of_D47n = uarray_compatible_interp(self.interp.D47.n, self.interp.D48)

		#inverse D47 calibration (ignoring calibration errors)
		self.interp.Teq_as_function_of_D47n = uarray_compatible_interp(self.interp.D47.n, self.interp.T)
		#inverse D47 calibration (including calibration errors)
		self.interp.Teq_as_function_of_D47u = uarray_compatible_interp(self.interp.D47, self.interp.T)

	def T_as_function_of_D47(
		self,
		D47: (_cd.uarray | ArrayLike),
		ignore_calib_uncertainties: bool = False,
	):
		"""
		Provided with one or more Δ<sub>47</sub> values (floats or ufloats), return ufloats for the
		corresponding equilibrium T values (ufloats with or without Δ<sub>47</sub> calibration uncertainties).

		**Arguments**
		* `D47`: array of Δ<sub>47</sub> values
		* `ignore_calib_uncertainties`: whether to propagate calibration uncertainties
		"""
		if ignore_calib_uncertainties:
			return _cd.uarray(self.interp.Teq_as_function_of_D47n(D47))
		else:
			return _cd.uarray(self.interp.Teq_as_function_of_D47u(D47))

	def D47u_as_function_of_D47n(
		self,
		D47: ArrayLike
	):
		"""
		Provided with one or more Δ<sub>47</sub> values (floats), return ufloats for the corresponding
		equilibrium Δ<sub>47</sub> values (ufloats with Δ<sub>47</sub> calibration uncertainties).
		"""
		return _cd.uarray(self.interp.D47u_as_function_of_D47n(D47))

	def D48u_as_function_of_D47n(
		self,
		D47: ArrayLike
	):
		"""
		Provided with one or more Δ<sub>47</sub> values (floats), return ufloats for the corresponding
		equilibrium Δ<sub>48</sub> values (ufloats with Δ<sub>48</sub> calibration uncertainties).
		"""
		return _cd.uarray(self.interp.D48u_as_function_of_D47n(D47))

	def D47_calib_function(
		self,
		T: (float | _uc.UFloat | _cd.uarray),
		return_without_uncertainties: bool = False,
		ignore_calib_uncertainties: bool = False,
	):
		return D4x_calib_function(
			T = T,
			coefs = self.D47_coefs,
			return_without_uncertainties = return_without_uncertainties,
			ignore_calib_uncertainties = ignore_calib_uncertainties,
		)

	def D48_calib_function(
		self,
		T: (float | _uc.UFloat | _cd.uarray),
		return_without_uncertainties: bool = False,
		ignore_calib_uncertainties: bool = False,
	):
		return D4x_calib_function(
			T = T,
			coefs = self.D48_coefs,
			return_without_uncertainties = return_without_uncertainties,
			ignore_calib_uncertainties = ignore_calib_uncertainties,
		)

	D47_calib_function.__doc__ = D4x_calib_function.__doc__.replace('Δ<sub>4x</sub>', 'Δ<sub>47</sub>')
	D48_calib_function.__doc__ = D4x_calib_function.__doc__.replace('Δ<sub>4x</sub>', 'Δ<sub>48</sub>')

	def T_ellipse(
		self,
		T: (_np.ndarray | _cd.uarray),
		p: float = 0.95,
		CM: (_np.ndarray | None) = None,
		Tse: (_np.ndarray | float | None) = None,
		ax: (_ppl.Axes | None) = None,
		**kwargs,
	) -> list:
		"""
		Plot the joint `p`-level confidence ellipses in (Δ<sub>47</sub>, Δ<sub>48</sub>)
		space, for temperatures equal to the elements of `T`, and return a list of the
		`Ellipse` objects thus created.

		**Arguments**
		* `T`: `ndarray` or `uarray` of temperatures to plot
		* `p`: confidence level
		* `ax`: which instance of `matplotlib.axes.Axes` to draw in; use current axes if `ax` = `None`.
		* `kwargs`: passed to `matplotlib.patches.Ellipse()`
		"""
		_T = _cd.as_uarray(T, CM = CM, Xse = Tse)
		return conf_ellipse(
			self.D47_calib_function(_T),
			self.D48_calib_function(_T),
			p = p,
			ax = ax,
			**kwargs,
		)

	def plot_D95_confidence_band(
		self,
		p: float = 0.95,
		Ti: (ArrayLike | None) = None,
		ax: (_ppl.Axes | None) = None,
		**kwargs,
	):
		"""
		Plot, for a given p-value, the confidence band of the thermodynamic equilibrium curve
		in (Δ<sub>47</sub>, Δ<sub>48</sub>) space.

		**Arguments**
		* `p`: confidence level
		* `Ti`: array of temperatures over which to evaluate confidence band (default: use `interp.T` attribute instead)
		* `ax`: `Axes` instance to plot to (default: use current Axes)
		* `kwargs`: passed to `patches.Polygon()`

		Returns the corresponding `Polygon` instance.
		"""

		from matplotlib import pyplot as _ppl
		from matplotlib.patches import Polygon as _Polygon

		if ax is None:
			ax = _ppl.gca()
		if Ti is None:
			Ti = self.interp.T
		polygon = ax.add_patch(
			_Polygon(
				confidence_band(
					Ti,
					self.D47_calib_function,
					self.D48_calib_function,
					p,
				),
				closed = True,
				**kwargs,
			)
		)
		return polygon


	def plot_D95_equilibrium(
		self,
		Tmin: float = 0.,
		Tmax: float = 1000.,
		NT: int = 101,
		Tmarkers: _np.typing.ArrayLike = [0, 25, 100, 250, 1000],
		kwargs_Tmarkers: dict = {},
		show_Tmarker_labels: bool = True,
		kwargs_Tmarker_labels: dict = {},
		show_Tmarker_ellipses: bool = False,
		kwargs_Tmarker_ellipses: dict = {},
		show_eqline: bool = True,
		kwargs_eqline: dict = {},
		show_confidence: bool = True,
		confidence_pvalue: float = 0.95,
		kwargs_confidence: dict = {},
		ax: (_ppl.Axes | None) = None,
		xlabel: str = '$Δ_{47}$   [‰]',
		ylabel: str = '$Δ_{48}$   [‰]',
		lw: float = 0.7,
	) -> (dict, dict):
		"""
		Plot a thermodynamic equilibrium curve in (Δ<sub>47</sub>, Δ<sub>48</sub>) space
		as a function of temperature.

		**Arguments**
		* `Tmin`: minimum T to plot
		* `Tmax`: maximum T to plot
		* `NT`: number of steps in equilibrium curve (interpolated at constant steps in 1/T<sup>2</sup> space)
		* `Tmarkers`: T markers to add along the curve
		* `kwargs_Tmarkers`: passed to `plot()` when plotting T markers
		* `show_Tmarker_labels`: whether to add T labels to T markers
		* `kwargs_Tmarker_labels`: passed to `text()` when plotting T markers
		* `show_Tmarker_ellipses`: whether to add confidence ellipses to T markers
		* `kwargs_Tmarker_ellipses`: passed to `T_ellipses()` when plotting T marker ellipses
		* `show_eqline`: whether to plot the equilibrium curve itself
		* `kwargs_eqline`: passed to `plot()` when plotting the equilibrium curve
		* `show_confidence`: whether to plot the confidence band of the equilibrium curve
		* `confidence_pvalue`: confidence level for the confidence band
		* `kwargs_confidence`: passed to `plot_D95_confidence_band()` when plotting the confidence band
		* `ax`: which instance of `matplotlib.axes.Axes` to draw in; use current axes if `ax` = `None`.
		* `xlabel`: string to pass to `xlabel()`
		* `ylabel`: string to pass to `ylabel()`
		* `lw`: default line width for most plot elements

		**Returns**
		* `data`: a dict of the T, Δ<sub>47</sub> and Δ<sub>48</sub> values generated for this plot:
			- `Te`  : temperature interpolated along the equilibrium curve
			- `D47e`: Δ<sub>47</sub> interpolated along the equilibrium curve
			- `D48e`: Δ<sub>48</sub> interpolated along the equilibrium curve
			- `Tm`  : temperature of T markers
			- `D47m`: Δ<sub>47</sub> of T markers
			- `D48m`: Δ<sub>48</sub> of T markers

		* `plot_elements`: a dict of the `Axes` elements generated for this plot:
			- `eqline`: `Line2D` of the equilibrium curve
			- `confidence`: `Polygon` object for the confidence band
			- `Tm`: `Line2D` of the T markers
			- `Tme`: list of `Ellipse` objects for the T marker ellipses
			- `Tml`: list of `Text` objects for the T marker labels
		"""

		from matplotlib import pyplot as _ppl

		default_kwargs_eqline = dict(
			marker = 'None',
			ls = '-',
			color = 'k',
			lw = lw,
		)
		default_kwargs_confidence = dict(
			ec = (0,0,0,1),
			fc = (0,0,0,0.15),
			lw = 0.,
		)
		default_kwargs_Tmarkers = dict(
			ls = 'None',
			marker = 'o',
			ms = 4,
			mfc = 'w',
			mec = 'k',
			mew = lw,
		)
		default_kwargs_Tmarker_ellipses = dict(
			fc = 'None',
			ec = 'k',
			lw = lw,
		)
		default_kwargs_Tmarker_labels = dict(
			size = 8,
			va = 'center',
			ha = 'left',
			linespacing = 3,
		)

		plot_elements = {}

		Ti = _np.linspace(
			(Tmin + 273.15)**-2,
			(Tmax + 273.15)**-2,
			NT
		)**-0.5 - 273.15

		Tmarkers = _np.array([_ for _ in Tmarkers if _ >= Ti.min() and _ <= Ti.max()])

		if ax is None:
			ax = _ppl.gca()
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		Xe = self.D47_calib_function(Ti)
		Ye = self.D48_calib_function(Ti)

		if show_eqline:
			plot_elements['eqline'], = ax.plot(
				_unp.nominal_values(Xe),
				_unp.nominal_values(Ye),
				**(default_kwargs_eqline | kwargs_eqline),
			)

		if show_confidence:
			plot_elements['confidence'] = self.plot_D95_confidence_band(
				p = confidence_pvalue,
				ax = ax,
				**(default_kwargs_confidence | kwargs_confidence),
			)

		Xm = self.D47_calib_function(Tmarkers)
		Ym = self.D48_calib_function(Tmarkers)
		if Tmarkers.size > 0:
			plot_elements['Tm'] = ax.plot(
				_unp.nominal_values(Xm),
				_unp.nominal_values(Ym),
				**(default_kwargs_Tmarkers | kwargs_Tmarkers),
			)
			if show_Tmarker_ellipses:
				plot_elements['Tme'] = conf_ellipse(
					Xm,
					Ym,
					ax = ax,
					**(default_kwargs_Tmarker_ellipses | kwargs_Tmarker_ellipses),
				)
			if show_Tmarker_labels:
				plot_elements['Tml'] = []
				for x,y,t in zip(Xm, Ym, Tmarkers):
					plot_elements['Tml'].append(
						ax.text(
							x.n,
							y.n,
							f'\n${t:.0f}\\,$°C',
							**(default_kwargs_Tmarker_labels | kwargs_Tmarker_labels),
						)
					)

		ax.autoscale_view()

		data = dict(
			Te = Ti,
			D47e = Xe,
			D48e = Ye,
			Tm = Tmarkers,
			D47m = Xm,
			D48m = Ym,
		)

		return data, plot_elements

	def _compute_p_and_D48eq_from_D47eq(
		self,
		D47,
		D48,
		D47eq,
		ignore_calib_uncertainties = False,
	):
		"""
		Used by the various `Engine.nearest_D47eq()` methods
		"""
		N = D47.size

		# Compute fit residuals for p values
		if ignore_calib_uncertainties:
			R = _cd.uarray(_np.concatenate((
				D47 - self.D47u_as_function_of_D47n(D47eq.n).n,
				D48 - self.D48u_as_function_of_D47n(D47eq.n).n,
			)))
		else:
			R = _cd.uarray(_np.concatenate((
				D47 - self.D47u_as_function_of_D47n(D47eq.n),
				D48 - self.D48u_as_function_of_D47n(D47eq.n),
			)))

		# Compute p values
		p = _np.zeros((N,))
		for k in range(N):
			r = R[k::N]
			z2 = r.m
			p[k] = 1-_chi2.cdf(z2, 1)

		# Compute D48eq
		D48eq = self.D48u_as_function_of_D47n(D47eq)

		return p, D48eq

	def nearest_D47eq(
		self,
		D47: _cd.uarray,
		D48: _cd.uarray,
		ignore_calib_uncertainties: bool = False,
	):
		"""
		Computes a `correldata.uarray` of *equilibrium* Δ<sub>47</sub> values, each of which is
		the closest (in the OGLS sense) to one (Δ<sub>47</sub>, Δ<sub>48</sub>) observation
		considered independently of the others.

		Also returns an array of corresponding p-values taking into account errors in Δ<sub>47</sub>
		and Δ<sub>48</sub> (and any covariance between the two) as well as errors in the
		Δ<sub>47</sub> and Δ<sub>48</sub> calibrations.

		> [!NOTE]
		> This is both the fastest and the strongly recommended version of this calculation.
		> It is expected to yield an `uarray` with reasonably accurate covariance between the
		> `D47eq` values, but also between `D47eq` and all other variables.
		"""

		N = D47.size
		N47 = self.D47_coefs.size
		N48 = self.D48_coefs.size
		D47eq = D47 * 0

		# _np.set_printoptions(threshold = _np.inf)
		# _np.set_printoptions(linewidth = _np.inf)

		for i in range(N):
			def fun(*args): # args = (D47, D48, *D47_calib_coefs, *D48_calib_coefs)

				args = _np.array(args)
				D47_n = args[0]
				D48_n = args[1]
				D47_calib_coefs_n = args[-N48-N47:-N48]
				D48_calib_coefs_n = args[-N48:]

				params = _lmfit.Parameters()
				params.add('D47eq', value = D47_n)

				D47_u = _cd.uarray([_uc.ufloat(D47_n, D47.s[i])])
				D48_u = _cd.uarray([_uc.ufloat(D48_n, D48.s[i])])
				D47_calib_coefs_u = _cd.uarray(_uc.correlated_values(D47_calib_coefs_n, self.D47_coefs.covar))
				D48_calib_coefs_u = _cd.uarray(_uc.correlated_values(D48_calib_coefs_n, self.D48_coefs.covar))

				D47i = D4x_calib_function(
					self.interp.T,
					D47_calib_coefs_u,
					return_without_uncertainties = False,
					ignore_calib_uncertainties = ignore_calib_uncertainties,
				)

				D48i = D4x_calib_function(
					self.interp.T,
					D48_calib_coefs_u,
					return_without_uncertainties = False,
					ignore_calib_uncertainties = ignore_calib_uncertainties,
				)

				D47_interp = uarray_compatible_interp(D47i.n, D47i)
				D48_interp = uarray_compatible_interp(D47i.n, D48i)

				def cost_fun(p):
					R = _cd.uarray(_np.concatenate((
						D47_u - D47_interp(p['D47eq'].value),
						D48_u - D48_interp(p['D47eq'].value),
					)))

					invS = _np.linalg.inv(R.covar)
					L = _cholesky(invS)

					return L @ R.n

				minresult = _lmfit.minimize(
					cost_fun,
					params,
					method = 'least_squares',
					scale_covar = False,
					jac = '3-point',
				)
				# slower but yields very similar results:
				# minresult = _lmfit.minimize(cost_fun, params, method = 'powell', scale_covar = False)

				return minresult.params['D47eq'].value

			wrapped_fun = _uc.wrap(fun)
			D47eq[i] = wrapped_fun(D47[i], D48[i], *self.D47_coefs, *self.D48_coefs)

		p, D48eq = self._compute_p_and_D48eq_from_D47eq(D47, D48, D47eq, ignore_calib_uncertainties = ignore_calib_uncertainties)

		return D47eq, D48eq, p

	def joint_nearest_D47eq(
		self,
		D47: _cd.uarray,
		D48: _cd.uarray,
		ignore_calib_uncertainties: bool = False,
	):
		"""
		Returns a `correldata.uarray` of equilibrium Δ<sub>47</sub> values which are *jointly* closest (in the OGLS sense)
		to a sequence of (Δ<sub>47</sub>, Δ<sub>48</sub>) pairs. Also returns an array of
		corresponding p-values taking into account errors in Δ<sub>47</sub> and Δ<sub>48</sub>
		(and any covariance between the two) as well as errors in the Δ<sub>47</sub> and
		Δ<sub>48</sub> calibrations.

		> [!CAUTION]
		> Caution: the use of this function is **not generally recommended** except for
		> experimentation purposes, because it is conceptually and numerically risky to *jointly*
		> fit the sequence of `Teq` values, as opposed to fitting each of them individually,
		> as done by the recommended function `nearest_D47eq()`.

		This is the most complete but slowest and not recommended version of this calculation.
		It is expected to yield an `uarray` with reasonably accurate covariance between the
		`D47eq` values, but also between `D47eq` and all other variables.

		A faster but incomplete and potentially less accurate version of this calculation is
		provided by `lazy_joint_nearest_D47eq()`.
		"""

		N = D47.size
		N47 = self.D47_coefs.size
		N48 = self.D48_coefs.size

		def fun(j, *args):

			args = _np.array(args)
			D47_n = args[:N]
			D48_n = args[N:2*N]
			D47_calib_coefs_n = args[-N48-N47:-N48]
			D48_calib_coefs_n = args[-N48:]

			params = _lmfit.Parameters()
			for k in range(N):
				params.add(f'D47eq{k}', value = D47_n[k])

			D47_u = _cd.uarray(_uc.correlated_values(D47_n, D47.covar))
			D48_u = _cd.uarray(_uc.correlated_values(D48_n, D48.covar))
			D47_calib_coefs_u = _cd.uarray(_uc.correlated_values(D47_calib_coefs_n, self.D47_coefs.covar))
			D48_calib_coefs_u = _cd.uarray(_uc.correlated_values(D48_calib_coefs_n, self.D48_coefs.covar))

			D47i = D4x_calib_function(
				self.interp.T,
				D47_calib_coefs_u,
				return_without_uncertainties = False,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			)

			D48i = D4x_calib_function(
				self.interp.T,
				D48_calib_coefs_u,
				return_without_uncertainties = False,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			)

			D47_interp = uarray_compatible_interp(D47i.n, D47i)
			D48_interp = uarray_compatible_interp(D47i.n, D48i)

			def cost_fun(p):
				_D47eq = _np.array([p[f'D47eq{k}'] for k in range(N)])
				R = _cd.uarray(_np.concatenate((
					D47_u - D47_interp(_D47eq),
					D48_u - D48_interp(_D47eq),
				)))

				invS = _np.linalg.inv(R.covar)
				L = _cholesky(invS)

				# print(((L @ R.n)**2).sum())
				return L @ R.n

			minresult = _lmfit.minimize(
				cost_fun,
				params,
				method = 'least_squares',
				scale_covar = False,
				jac = '3-point',
			)
			# slower but yields very similar results:
			# minresult = _lmfit.minimize(cost_fun, params, method = 'powell', scale_covar = False)

			return minresult.params[f'D47eq{j}'].value

		wrapped_fun = _uc.wrap(fun)

		D47eq = _cd.uarray([wrapped_fun(j, *D47, *D48, *self.D47_coefs, *self.D48_coefs) for j in range(N)])
		p, D48eq = self._compute_p_and_D48eq_from_D47eq(D47, D48, D47eq, ignore_calib_uncertainties = ignore_calib_uncertainties)

		return D47eq, D48eq, p

	def lazy_joint_nearest_D47eq(
		self,
		D47: _cd.uarray,
		D48: _cd.uarray,
		ignore_calib_uncertainties: bool = False,
	):
		"""
		Returns a `correldata.uarray` of equilibrium Δ<sub>47</sub> values which are *jointly* closest (in the OGLS sense)
		to a sequence of (Δ<sub>47</sub>, Δ<sub>48</sub>) pairs. Also returns an array of
		corresponding p-values taking into account errors in Δ<sub>47</sub> and Δ<sub>48</sub>
		(and any covariance between the two) as well as errors in the Δ<sub>47</sub> and
		Δ<sub>48</sub> calibrations.

		> [!CAUTION]
		> Caution: the use of this function is **not generally recommended** except for
		> experimentation purposes, because it is conceptually and numerically risky to *jointly*
		> fit the sequence of `Teq` values, as opposed to fitting each of them individually,
		> as done by the recommended function `nearest_D47eq()`.

		This is a faster but incomplete version of this calculation. It is expected to yield an
		`uarray` with roughly accurate covariance between the `Teq` values, but without computing
		the covariance with any other variables.

		A slower but complete and more accurate version of this calculation is provided by
		`joint_nearest_D47eq()`.
		"""

		N = D47.size

		params = _lmfit.Parameters()
		for k in range(N):
			params.add(f'D47eq{k}', value = D47[k].n)

		def cost_fun(p, ignore_calib_uncertainties = ignore_calib_uncertainties):
			_D47eq = _np.array([p[f'D47eq{k}'] for k in range(N)])

			if ignore_calib_uncertainties:
				R = _cd.uarray(_np.concatenate((
					D47 - self.D47u_as_function_of_D47n(_D47eq).n,
					D48 - self.D48u_as_function_of_D47n(_D47eq).n,
				)))
			else:
				R = _cd.uarray(_np.concatenate((
					D47 - self.D47u_as_function_of_D47n(_D47eq),
					D48 - self.D48u_as_function_of_D47n(_D47eq),
				)))

			invS = _np.linalg.inv(R.covar)
			L = _cholesky(invS)

			# print(((L @ R.n)**2).sum())
			return L @ R.n

		minresult = _lmfit.minimize(
			cost_fun,
			params,
			method = 'least_squares',
			scale_covar = False,
			jac = '3-point',
		)

		D47eq = _cd.uarray([minresult.uvars[f'D47eq{k}'] for k in range(N)])

		p, D48eq = self._compute_p_and_D48eq_from_D47eq(D47, D48, D47eq, ignore_calib_uncertainties = ignore_calib_uncertainties)

		return D47eq, D48eq, p

	def projected_D47eq(
		self,
		D47: _cd.uarray,
		D48: _cd.uarray,
		kinetic_slope: (float | _uc.UFloat),
	):
		"""
		Projects one or more (Δ<sub>47</sub>, Δ<sub>48</sub>) observations onto the equlibrium curve
		following a kinetic fractionation vector with a given slope (∂Δ<sub>48</sub>/∂Δ<sub>47</sub>).

		**Arguments**
		* `D47`: observed Δ<sub>47</sub> value(s)
		* `D48`: observed Δ<sub>48</sub> value(s)
		* `kinetic_slope`: kinetic fractionation slopw, with or without uncertainty

		Returns a tuple of uarrays corresponding to the projected Δ<sub>47</sub> and Δ<sub>48</sub> values.

		> [!NOTE]
		> This is not a least-squares minimization problem but a direct calculation, and should thus
		> be much faster than the various `CorelData.nearestD47eq()` methods.
		"""

		D47 = _cd.uarray(D47)
		D48 = _cd.uarray(D48)
		N = D47.size
		N47c = self.D47_coefs.size
		N48c = self.D48_coefs.size
		D47p = D47 * 0

		for i in range(N):

			# function to solve
			def fun(x, *args): # args = (D47, D48, kinetic_slope, *self.D47_coefs, *self.D48_coefs)

				args = _np.array(args)
				D47_n = args[0]
				D48_n = args[1]
				kslope_n = args[2]
				D47_calib_coefs_n = args[-N48c-N47c:-N48c]
				D48_calib_coefs_n = args[-N48c:]

				D47i = D4x_calib_function(
					self.interp.T,
					D47_calib_coefs_n,
					return_without_uncertainties = False,
				)

				D48i = D4x_calib_function(
					self.interp.T,
					D48_calib_coefs_n,
					return_without_uncertainties = False,
				)

				D48_interp = uarray_compatible_interp(D47i, D48i)

				return D48_n - D48_interp(x) - kslope_n * (D47_n - x)

			def g(*args):
				return _fsolve(fun, [100.], args = args)[0]

			wg = _uc.wrap(g)

			D47p[i] = wg(
				D47[i],
				D48[i],
				kinetic_slope,
				*self.D47_coefs,
				*self.D48_coefs,
			)

		_, D48p = self._compute_p_and_D48eq_from_D47eq(D47, D48, D47p, ignore_calib_uncertainties = False)

		return D47p, D48p

	def Teq_pdf(
		self,
		D47: _uc.ufloat,
		Tmin: (float | None)             = None,
		Tmax: (float | None)             = None,
		Tinc: float                      = 0.2,
		default_D47_sigmas: float        = 4.0,
		ignore_calib_uncertainties: bool = False,
		run_qmc: bool                    = False,
		N_qmc: int                       = 1024,
	):
		"""
		Compute the unit-normalized probability distribution function (PDF) of the
		equilibrium temperature (`Teq`) for a given (`UFloat`) value of Δ<sub>47</sub>.

		**Arguments**
		* `D47`: Δ<sub>47</sub> value (with uncertainty)
		* `Tmin`: minimum temperature over which to compute the PDF; if not specified,
		use temperature corresponding to `D47.n + `default_D47_sigmas` * D47.s`
		* `Tmax`: maximum temperature over which to compute the PDF; if not specified,
		use temperature corresponding to `D47.n - `default_D47_sigmas` * D47.s`
		* `Tinc`: temperature increment over which to compute the PDF
		* `default_D47_sigmas`: see `Tmin` and `Tmin` above
		* `ignore_calib_uncertainties`: whether to propagate calibration uncertainties
		* `run_qmc`: whether to also run a Quasi Monte carlo simulation to estimate the PDF
		* `N_qmc`: number of iterations in the above Quasi Monte Carlo simulation

		**Returns**
		* `Ti`: Evenly-spaced array of temperature values over which the PDF is computed
		* `pdf`: PDF evaluated over `Ti`
		* `Tqmc` (only returned if `run_qmc = True`): array of `N_qmc` temperature values
		computed in the Quasi Monte Carlo simulation
		"""

		if Tmin is None:
			Tmin = _np.floor(self.T_as_function_of_D47(
				D47.n + default_D47_sigmas * D47.s,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			).n)

		if Tmax is None:
			Tmax = _np.ceil(self.T_as_function_of_D47(
				D47.n - default_D47_sigmas * D47.s,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			).n)

		assert Tmin < Tmax, "Tmax must be strictly greater than Tmin"
		assert Tinc > 0, "Tinc must be strictly greater than zero"

		# compute interpolated Ti values
		Ti = _np.arange(Tmin, Tmax+Tinc, Tinc)

		pdf = transform_pdf_monotonic(
			f_inv   = lambda T: D4x_calib_function(
				T,
				self.D47_coefs,
				return_without_uncertainties = ignore_calib_uncertainties,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			),
			df_inv  = lambda T: D4x_calib_derivative(
				T,
				self.D47_coefs,
				return_without_uncertainties = ignore_calib_uncertainties,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			),
			mu_x    = D47.n,
			sigma_x = D47.s,
			yi      = Ti,
		)

		if run_qmc:

			from scipy.stats import qmc
			from tqdm.rich import tqdm

			#parameters to jiggle
			input_params = _cd.uarray([D47, *self.D47_coefs])

			# QMC sampler for the correlation matrix of these parameters
			qmc_dist = qmc.MultivariateNormalQMC(
				mean = input_params.n*0,
				cov = input_params.cor,
			)

			# QMC samples
			qmc_draws = input_params.n + qmc_dist.random(N_qmc) * input_params.s

			# initialize T_qmc
			Tqmc = _cd.uarray(_np.zeros((N_qmc,)))

			for k in tqdm(range(N_qmc)):
				# jiggled D47 and D47coefs
				_D47 = qmc_draws[k,0]
				if ignore_calib_uncertainties:
					_coefs = self.D47_coefs
				else:
					_coefs = _cd.uarray(_uc.correlated_values(qmc_draws[k,1:], self.D47_coefs.covar))

				# jiggled D47
				_D47i = D4x_calib_function(self.interp.T, _coefs)
				_f = uarray_compatible_interp(_D47i.n, self.interp.T)
				Tqmc[k] = _f(_D47)

			return Ti, pdf, Tqmc

		return Ti, pdf


### Utilities and CLI ###


def save_Teq_report(
	X,
	Y,
	T,
	p,
	filename,
	Xname = 'D47',
	Yname = 'D48',
	Tname = 'T95',
	labelname = 'Sample',
	fmt_Xnv = '.4f',
	fmt_Xse = '.4f',
	fmt_Ynv = '.4f',
	fmt_Yse = '.4f',
	fmt_Tnv = '.1f',
	fmt_Tse = '.1f',
	fmt_cm = '.6f',
	fmt_pv = '.2e',
	labels = None,
	sep = ',',
	p_cutoff = 0.05,
):
	"""
	Save a temperature report to a csv file.
	Includes observed `D47`, `D48`, p-equilibrium values, and nearest `Teq` with sensible precision defaults.
	Alternatively, users may find [`correldata.CorrelData.str()`](https://mdaeron.github.io/correldata/#CorrelData.str)
	to be more versatile.
	"""
	N = T.size
	if labels is None:
		labels = [str(k+1) for k in range(N)]

	with open(filename, 'w') as fid:
		fid.write(f'{labelname}{sep}{Xname}{sep}SE{sep}correl{sep*N}{Yname}{sep}SE{sep}correl{sep*N}p-value{sep}{Tname}{sep}SE{sep}correl')
		Xnv = _unp.nominal_values(X)
		Xse = _unp.std_devs(X)
		Xcm = _np.array(_uc.correlation_matrix(X))
		Ynv = _unp.nominal_values(Y)
		Yse = _unp.std_devs(Y)
		Ycm = _np.array(_uc.correlation_matrix(Y))
		Tnv = _unp.nominal_values(T)
		Tse = _unp.std_devs(T)
		Tcm = _np.array(_uc.correlation_matrix(T))
		for k in range(X.size):
			fid.write(f'\n{labels[k]}{sep}{Xnv[k]:{fmt_Xnv}}{sep}{Xse[k]:{fmt_Xse}}{sep}')
			fid.write(sep.join([f'{Xcm[j,k]:{fmt_cm}}' for j in range(N)]))
			fid.write(f'{sep}{Ynv[k]:{fmt_Ynv}}{sep}{Yse[k]:{fmt_Yse}}{sep}')
			fid.write(sep.join([f'{Ycm[j,k]:{fmt_cm}}' for j in range(N)]))
			fid.write(f'{sep}{p[k]:{fmt_pv}}')
			if p[k] >= p_cutoff:
				fid.write(f'{sep}{Tnv[k]:{fmt_Tnv}}{sep}{Tse[k]:{fmt_Tse}}{sep}')
				fid.write(sep.join([f'{Tcm[j,k]:{fmt_cm}}' for j in range(N)]))

_rich_utils.STYLE_HELPTEXT = ''

__app = _typer.Typer(
	add_completion = False,
	context_settings={'help_option_names': ['-h', '--help']},
	rich_markup_mode = 'rich',
)

@__app.command()
def _cli(
	input:   _Annotated[str, _typer.Option('--input', '-i', help = "Input file to read from (otherwise read from stdin).")] = None,
	output:  _Annotated[str, _typer.Option('--output', '-o', help = "Output file to write to (otherwise write to stdout).")] = None,
	kslope:  _Annotated[str, _typer.Option('--kslope', '-k', help = "Kinetic fractionation slope, using format [bold]'n(s)'[/bold] (with quotes), where [bold]n[/bold] is the slope and [bold]s[/bold] its standard error.")] = None,
	hpoutput: _Annotated[bool, _typer.Option('--high-precision-output', '-p', help = "Generate higher precision output.")] = False,
	show_mixed_correl: _Annotated[bool, _typer.Option('--show_mixed_correl', '-m', help = "Show correlations between different fields.")] = False,
	version: _Annotated[bool, _typer.Option('--version', '-v', help = 'Show version and exit.')] = False,
):
	"""
[b]Purpose:[/b]

Reads data from an input file, computes p-value and T estimates, and print out the results.
"""
	if version:
		print(__version__)
		return None

	if input is None:
		datastring = ''.join(sys.stdin)
	elif isinstance(input, str):
		with open(input) as fid:
			datastring = fid.read()

	data = _cd.read_str(datastring)

	E = Engine()

	D47eq, D48eq, p = E.nearest_D47eq(data['D47'], data['D48'])
	Teq = E.T_as_function_of_D47(D47eq)
	data['eq_pvalue'] = p
	data['Teq'] = Teq

	if isinstance(kslope, str):
		kslope = kslope.split(')')[0]
		kslope = kslope.split('(')
		kslope = _uc.ufloat(float(kslope[0]), float(kslope[1]))

		D47kp, D48kp = E.projected_D47eq(data['D47'], data['D48'], kinetic_slope = kslope)
		Tkp = E.T_as_function_of_D47(D47kp)

		data['kslope'] = _cd.uarray([kslope for _ in data['D47']])

		data['Tkp'] = Tkp

	ffmt = {
		'D47': '.6f',
		'D48': '.6f',
		'kslope': lambda x: f'{x:z.6f}'.rstrip('0'),
		'Teq': 'z.6f',
		'Tkp': 'z.6f',
	} if hpoutput else {
		'D47': '.4f',
		'D48': '.4f',
		'kslope': lambda x: f'{x:z.6f}'.rstrip('0'),
		'Teq': 'z.2f',
		'Tkp': 'z.2f',
	}

	out = data.str(
		float_format = ffmt,
		show_mixed_correl = show_mixed_correl,
		exclude_fields = ['correl_kslope'],
	)

	if output is None:
		print(out)
	elif isinstance(output, str):
		with open(output, 'w') as fid:
			fid.write(out)

def __cli(): __app()
