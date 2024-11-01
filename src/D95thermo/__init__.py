"""
Estimate carbonate formation temperatures from dual clumped isotope measurements

.. include:: ../../docpages/install.md
.. include:: ../../docpages/cli.md

* * *
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'mathieu@daeron.fr'
__copyright__ = 'Copyright (c) 2024 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__date__      = '2024-10-15'
__version__   = '0.9.0'


import sys
import numpy as _np
import ogls as _ogls
import uncertainties as _uc
import lmfit as _lmfit
import correldata as _cd
import typer as _typer

from uncertainties import unumpy as _unp
from matplotlib import pyplot as _ppl
from matplotlib.patches import Ellipse as _Ellipse
from scipy.stats import chi2 as _chi2
from scipy.linalg import eigh as _eigh
from scipy.linalg import cholesky as _cholesky
from scipy.optimize import fsolve as _fsolve
from typing_extensions import Annotated as _Annotated


#### Utility variables and functions ####

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
# 
# _D47_approx_calib_coefs = [_D47_approx.bfp['a0'], _D47_approx.bfp['a2']]


def _compute_D48_calib_coefficients(reprocess = False):
	"""
	Based on Fiebig et al. (2021)
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

		data = _cd.read_data(datastr)
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
			[0.12135157920099604, 1.0379702801201238],
			[[ 7.39697438e-06, -6.90467053e-05], [-6.90467053e-05,  1.46002771e-03]],
		)
	
	a0 = b0
	a1 *= b1
	a2 *= b1
	a3 *= b1
	a4 *= b1
	
	return _cd.uarray([a0, a1, a2, a3, a4])


#### Calibration variables and functions ####

def D4x_calib_function(
	T: (float | _uc.UFloat | _cd.uarray),
	coefs: _cd.uarray,
	return_without_uncertainties: bool = False,
	ignore_calib_uncertainties: bool = False,
) -> _np.ndarray:
	"""
	If `return_without_uncertainties` is False, returns one or more ufloat values.
	In that case, if T is a ufloat or an array of ufloats, the resulting D4x ufloat
	values will account for this source of uncertainty, but if T is a float or an
	array of floats, the D4x ufloat values will only account for uncertainties in
	the calibration coefficients. If `return_without_uncertainties` is True, returns
	the D4x values without error propagation of any kind.
	"""
	degs = _np.arange(coefs.size)
	
	D4x = (
		_np.expand_dims(_cd.nv(coefs) if ignore_calib_uncertainties else coefs, 1)
		* _np.expand_dims((T+273.15)**-1, 0)
		** _np.expand_dims(degs, 1)
	).sum(
		axis = 0
		if isinstance(T, _np.ndarray)
		else None
	)

	if D4x.ndim == 0:
		return D4x.tolist().n if return_without_uncertainties else D4x.tolist()
	return D4x.n if return_without_uncertainties else D4x


# D47_calib_coefs from OGLS23 (D47calib v1.3.1)
D47_calib_coefs = _cd.read_data('''
              coefs,                     SE,        correl,
0.17437754366432887,   4.911105567257293e-3,    1.        , -0.93797005,  0.8865771
 -18.14215245127414,      5.632326472234856,   -0.93797005,  1.        , -0.98994249
42.65722989162373e3,     1.27712751715908e3,    0.8865771 , -0.98994249,  1.
'''[1:-1])['coefs']

# D47_calib_coefs = _cd.uarray(_uc.correlated_values(D47_calib_coefs.n, D47_calib_coefs.covar / 1e6))

def D47_calib_function(
	T: (float | _uc.UFloat | _cd.uarray),
	coefs: _cd.uarray = D47_calib_coefs,
	return_without_uncertainties: bool = False,
	ignore_calib_uncertainties: bool = False,
):
	"""
	If `return_without_uncertainties` is False, returns one or more ufloat values.
	In that case, if T is a ufloat or an array of ufloats, the resulting D47 ufloat
	values will account for this source of uncertainty, but if T is a float or an
	array of floats, the D47 ufloat values will only account for uncertainties in
	the calibration coefficients. If `return_without_uncertainties` is True, returns
	the D47 values without error propagation of any kind.
	"""
	return D4x_calib_function(
		T = T,
		coefs = coefs,
		return_without_uncertainties = return_without_uncertainties,
		ignore_calib_uncertainties = ignore_calib_uncertainties,
	)

# D48_calib_coefs = _compute_D48_calib_coefficients(reprocess = True)
# print(_cd.data_string(
# 	{'coefs': D48_calib_coefs},
# 	float_format = 'z.12g',
# 	correl_format = 'z.12f',
# ))

D48_calib_coefs = _cd.read_data('''
         coefs,         SE_coefs,    correl_coefs,                ,                ,                ,                
0.121349237888, 0.00390048540724,  1.000000000000, -0.664181963395,  0.664181963395, -0.664181963395,  0.664181963395
 6.22931985613,    0.32896761459, -0.664181963395,  1.000000000000, -1.000000000000,  1.000000000000, -1.000000000000
 -13481.983494,    711.977559735,  0.664181963395, -1.000000000000,  1.000000000000, -1.000000000000,  1.000000000000
 9336714.66607,    493067.754224, -0.664181963395,  1.000000000000, -1.000000000000,  1.000000000000, -1.000000000000
-770413883.573,    40685214.9801,  0.664181963395, -1.000000000000,  1.000000000000, -1.000000000000,  1.000000000000
'''[1:-1])['coefs']

def D48_calib_function(
	T: (float | _uc.UFloat | _cd.uarray),
	coefs: _cd.uarray = D48_calib_coefs,
	return_without_uncertainties: bool = False,
	ignore_calib_uncertainties: bool = False,
):
	"""
	If `return_without_uncertainties` is False, returns one or more ufloat values.
	In that case, if T is a ufloat or an array of ufloats, the resulting D48 ufloat
	values will account for this source of uncertainty, but if T is a float or an
	array of floats, the D48 ufloat values will only account for uncertainties in
	the calibration coefficients. If `return_without_uncertainties` is True, returns
	the D48 values without error propagation of any kind.
	"""
	return D4x_calib_function(
		T = T,
		coefs = coefs,
		return_without_uncertainties = return_without_uncertainties,
		ignore_calib_uncertainties = ignore_calib_uncertainties,
	)


def to_pair_of_uarrays(
	X: (_cd.uarray | _np.ndarray | _uc.UFloat | float),
	Y: (_cd.uarray | _np.ndarray | _uc.UFloat | float) = None,
	CM: (_np.ndarray | None) = None,
	Xse: (_np.ndarray | float | None) = None,
	Yse: (_np.ndarray | float | None) = None,
) -> tuple:
	"""
	Convert (X, Y) to a pair of uarrays.
	
	**Arguments**
	* `X`: x values
	* `Y`: y values
	* `CM`: covariance matrix of `(*X, *Y)`; not needed if elements of X and Y are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html).
		or if (`Xse`, `Yse`) are specified.
	* `Xse`, `Yse`: SE of X and Y; not needed if elements of X and Y are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html)
		or if `CM` is specified.
	
	If neither `CM`, `Xse` nor `Yse` are specified, assume SE = 0.
	"""
	
	if type(X) is not type(Y):
		raise TypeError(f'X ({type(X)}) and Y ({type(Y)}) must have the same type.')

	if isinstance(X, _cd.uarray):
		return (X, Y)

	if isinstance(X, _np.ndarray):
		if (
			_np.all([isinstance(_, _uc.UFloat) for _ in X])
			and
			_np.all([isinstance(_, _uc.UFloat) for _ in Y])
		):
			return _cd.uarray(X), _cd.uarray(Y)
		else:
			X = X.astype(float)
			Y = Y.astype(float)
			
			if CM is not None:
				if Xse is not None: raise ValueError('Too much information: Xse is redundant because CM is already specified.')
				if Yse is not None: raise ValueError('Too much information: Yse is redundant because CM is already specified.')

			if CM is None:
				if Xse is None:
					Xse = X * 0
				if Yse is None:
					Yse = Y * 0

				CMx = _np.diag((*Xse,))**2
				CMy = _np.diag((*Yse,))**2			
				return _cd.uarray(_uc.correlated_values(X, CMx)), _cd.uarray(_uc.correlated_values(Y, CMy))

			else:
				XY = _cd.uarray(_uc.correlated_values([*X, *Y], CM))
				return XY[:X.size], XY[X.size:]
				
	if isinstance(X, _uc.UFloat):
		return _cd.uarray([X]), _cd.uarray([Y])

	if isinstance(X, (float, int)):

		if CM is not None:
			if Xse is not None: raise ValueError('Too much information: Xse is redundant because CM is already specified.')
			if Yse is not None: raise ValueError('Too much information: Yse is redundant because CM is already specified.')

		if CM is None:
			if Xse is None: raise ValueError('Not enough information: specify either CM or Xse.')
			if Yse is None: raise ValueError('Not enough information: specify either CM or Yse.')				

			CM = _np.diag([Xse, Yse])**2

		XY = _cd.uarray(_uc.correlated_values([X, Y], CM))
		return XY[:1], XY[1:]


def to_uarray(
	X: (_cd.uarray | _np.ndarray | _uc.UFloat | float),
	CM: (_np.ndarray | None) = None,
	Xse: (_np.ndarray | float | None) = None,
) -> _cd.uarray:
	"""
	Convert X to uarray type.
	
	**Arguments**
	* `X`: x values
	* `CM`: covariance matrix of X; not needed if elements of X are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html).
		or if `Xse` is specified.
	* `Xse`,: SE of X; not needed if elements of X are of type
		[`uncertainties.UFloat`](https://pythonhosted.org/uncertainties/tech_guide.html)
		or if `CM` is specified.
	
	If neither `CM` nor `Xse` are specified, assume SE = 0.
	"""
	
	if isinstance(X, _cd.uarray):
		return X

	if isinstance(X, _np.ndarray):
		if _np.all([isinstance(_, _uc.UFloat) for _ in X]):
			return _cd.uarray(X)
		else:
			X = X.astype(float)
			
			if CM is not None:
				if Xse is not None: raise ValueError('Too much information: Xse is redundant because CM is already specified.')

			if CM is None:
				if Xse is None:
					Xse = X * 0

				CM = _np.diag((*Xse,))**2

			return _cd.uarray(_uc.correlated_values(X, CM))
				
	if isinstance(X, _uc.UFloat):
		return _cd.uarray([X])

	if isinstance(X, (float, int)):

		if CM is not None:
			if Xse is not None: raise ValueError('Too much information: Xse is redundant because CM is already specified.')
			Xse = CM[0,0]**0.5

		return _cd.uarray([_uc.ufloat(X, Xse)])


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
	Plot the joint `p`-level confidence ellipses for the elements of (`X`, `Y`),
	and return a list of the `Ellipse` objects thus created.
	
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
	"""


	r2 = _chi2.ppf(p, 2)
	kwargs = dict(fc = 'None', ec = 'k', lw = 0.7) | kwargs

	if ax is None:
		ax = _ppl.gca()

	out = []

	for x, y in zip(
		*to_pair_of_uarrays(X, Y, CM = CM, Xse = Xse, Yse = Yse)
	):
		val, vec = _eigh(_uc.covariance_matrix((x, y)))
		width, height = 2 * (val[:, None] * r2)**0.5
		angle = _np.degrees(_np.arctan2(*vec[::-1, 0]))

		out.append(
			ax.add_patch(
				_Ellipse(
					xy = (x.n, y.n),
					width = width,
					height = height,
					angle = angle,
					**kwargs,
				)
			)
		)

	return (*out,)


def T_ellipse(
	T: (_np.ndarray | _cd.uarray),
	p: float = 0.95,
	CM: (_np.ndarray | None) = None,
	Tse: (_np.ndarray | float | None) = None,
	D47_calib_function = D47_calib_function,
	D48_calib_function = D48_calib_function,
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
	* `D47_calib_function`: specify Δ<sub>47</sub> calibration
	(yielding an `uarray` of Δ<sub>47</sub> values accounting
	for calibration uncertainties as well as uncertainties in `T`)
	* `D48_calib_function`: specify Δ<sub>48</sub> calibration
	(yielding an `uarray` of Δ<sub>48</sub> values accounting
	for calibration uncertainties as well as uncertainties in `T`)
	* `ax`: which instance of `matplotlib.axes.Axes` to draw in; use current axes if `ax` = `None`.
	* `kwargs`: passed to `matplotlib.patches.Ellipse()`	
	"""
	_T = to_uarray(T, CM = CM, Xse = Tse)
	return conf_ellipse(
		D47_calib_function(_T),
		D48_calib_function(_T),
		p = p,
		ax = ax,
		**kwargs,
	)


def plot_D95_equilibrium(
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
	show_D47ci: bool = True,
	kwargs_D47ci: dict = {},
	show_D48ci: bool = True,
	kwargs_D48ci: dict = {},
	ci_pvalue: float = 0.95,
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
	* `show_D47ci`: whether to plot the Δ<sub>47</sub> confidence band of the equilibrium curve
	* `kwargs_D47ci`: passed to `fill_betweenx()` when plotting the Δ<sub>47</sub> confidence band
	* `show_D48ci`: whether to plot the Δ<sub>48</sub> confidence band of the equilibrium curve
	* `kwargs_D48ci`: passed to `fill_betweenx()` when plotting the Δ<sub>48</sub> confidence band
	* `ci_pvalue`: confidence level for the Δ<sub>47</sub> and Δ<sub>48</sub> confidence band
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
		- `D47ci`: `PolyCollection` of the Δ<sub>47</sub> confidence band
		- `D48ci`: `PolyCollection` of the Δ<sub>48</sub> confidence band
		- `Tm`: `Line2D` of the T markers
		- `Tme`: list of `Ellipse` objects for the T marker ellipses
		- `Tml`: list of `Text` objects for the T marker labels
	"""
	
	default_kwargs_eqline = dict(
		marker = 'None',
		ls = '-',
		color = 'k',
		lw = lw,
	)
	default_kwargs_D48ci = dict(
		color = 'k',
		lw = 0,
		alpha = 0.15,
	)
	default_kwargs_D47ci = dict(
		color = 'k',
		lw = 0,
		alpha = 0.15,
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
	default_kwargs_Tmarker_errorbars = dict(
		ecolor = 'k',
		elinewidth = 0.7,
		capthick = 0.7,
		capsize = 3,
		ls = 'None',
		marker = 'None',
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

	cif = _chi2.ppf(ci_pvalue, 1)**.5

	Xe = D47_calib_function(Ti)
	Ye = D48_calib_function(Ti)
	
	if show_eqline:
		plot_elements['eqline'], = ax.plot(
			_unp.nominal_values(Xe),
			_unp.nominal_values(Ye),
			**(default_kwargs_eqline | kwargs_eqline),
		)

	if show_D48ci:
		plot_elements['D48ci'] = ax.fill_between(
			_unp.nominal_values(Xe),
			_unp.nominal_values(Ye) - cif * _unp.std_devs(Ye),
			_unp.nominal_values(Ye) + cif * _unp.std_devs(Ye),
			**(default_kwargs_D48ci | kwargs_D48ci),
		)

	if show_D47ci:
		plot_elements['D47ci'] = ax.fill_betweenx(
			_unp.nominal_values(Ye),
			_unp.nominal_values(Xe) - cif * _unp.std_devs(Xe),
			_unp.nominal_values(Xe) + cif * _unp.std_devs(Xe),
			**(default_kwargs_D47ci | kwargs_D47ci),
		)
	
	Xm = D47_calib_function(Tmarkers)
	Ym = D48_calib_function(Tmarkers)
	if Tmarkers.size > 0:
		plot_elements['Tm'] = ax.plot(
			_unp.nominal_values(Xm),
			_unp.nominal_values(Ym),
			**(default_kwargs_Tmarkers | kwargs_Tmarkers),
		)
		if show_Tmarker_ellipses:
			plot_elements['Tme'] = conf_ellipses(
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


#### Temperature estimates ####


def nearest_Teq(
	D47: _cd.uarray,
	D48: _cd.uarray,
	D47_calib_coefs: _cd.uarray = D47_calib_coefs,
	D48_calib_coefs: _cd.uarray = D48_calib_coefs,
	ignore_calib_uncertainties: bool = False,
):
	"""
	Returns a `correldata.uarray` of T values, each of which is the closest (in the OGLS sense)
	to one (Δ<sub>47</sub>, Δ<sub>48</sub>) pair considered independently of the others.
	Also returns an array of corresponding p-values taking into account errors in Δ<sub>47</sub>
	and Δ<sub>48</sub> (and any covariance between the two) as well as errors in the
	Δ<sub>47</sub> and Δ<sub>48</sub> calibrations.
	
	This is both the fastest and the strongly recommended version of this calculation.
	It is expected to yield an `uarray` with reasonably accurate covariance between the
	`Teq` values, but also between `Teq` and all other variables. Usually, these `Teq`
	values are strongly correlated with the corresponding values of `D47`.
	"""

	N = D47.size
	N47 = D47_calib_coefs.size
	N48 = D48_calib_coefs.size
	Teq = D47 * 0
	
	for i in range(N):
		def fun(*args): # args = (D47, D48, *D47_calib_coefs, *D48_calib_coefs)
	
			args = _np.array(args)
			D47_n = args[0]
			D48_n = args[1]
			D47_calib_coefs_n = args[-N48-N47:-N48]
			D48_calib_coefs_n = args[-N48:]
				
			params = _lmfit.Parameters()
			a0, a2 = _D47_approx_calib_coefs
			params.add('Ti', value = ((D47_n - a0) / a2)**-.5 - 273.15)
			
			D47_u = _cd.uarray([_uc.ufloat(D47_n, D47.s[i])])
			D48_u = _cd.uarray([_uc.ufloat(D48_n, D48.s[i])])
			D47_calib_coefs_u = _cd.uarray(_uc.correlated_values(D47_calib_coefs_n, D47_calib_coefs.covar))
			D48_calib_coefs_u = _cd.uarray(_uc.correlated_values(D48_calib_coefs_n, D48_calib_coefs.covar))
	
			def cost_fun(
				p,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
			):
				R = _cd.uarray(_np.concatenate((
					D47_u - D4x_calib_function(p['Ti'], D47_calib_coefs_u, ignore_calib_uncertainties = ignore_calib_uncertainties),
					D48_u - D4x_calib_function(p['Ti'], D48_calib_coefs_u, ignore_calib_uncertainties = ignore_calib_uncertainties),
				)))
				
				invS = _np.linalg.inv(R.covar)
				L = _cholesky(invS)
		
				return L @ R.n
			
			minresult = _lmfit.minimize(cost_fun, params, method = 'least_squares', scale_covar = False, jac = '3-point')
			# slower but yields very similar results:
			# minresult = _lmfit.minimize(cost_fun, params, method = 'powell', scale_covar = False)
	
			return minresult.params['Ti'].value
		
		wrapped_fun = _uc.wrap(fun)
		Teq[i] = wrapped_fun(D47[i], D48[i], *D47_calib_coefs, *D48_calib_coefs)
	
	R = _cd.uarray(_np.concatenate((
		D47 - D4x_calib_function(Teq.n, D47_calib_coefs, ignore_calib_uncertainties = ignore_calib_uncertainties),
		D48 - D4x_calib_function(Teq.n, D48_calib_coefs, ignore_calib_uncertainties = ignore_calib_uncertainties),
	)))
	
	p = _np.zeros((N,))
	for k in range(N):
		r = R[k::N]
		z2 = r.m
		p[k] = 1-_chi2.cdf(z2, 2)

	return Teq, p


def joint_nearest_Teq(
	D47: _cd.uarray,
	D48: _cd.uarray,
	D47_calib_coefs: _cd.uarray = D47_calib_coefs,
	D48_calib_coefs: _cd.uarray = D48_calib_coefs,
	ignore_calib_uncertainties: bool = False,
):
	"""
	Returns a `correldata.uarray` of T values which are *jointly* closest (in the OGLS sense)
	to a sequence of (Δ<sub>47</sub>, Δ<sub>48</sub>) pairs. Also returns an array of
	corresponding p-values taking into account errors in Δ<sub>47</sub> and Δ<sub>48</sub>
	(and any covariance between the two) as well as errors in the Δ<sub>47</sub> and
	Δ<sub>48</sub> calibrations.

	Caution: the use of this function is **not generally recommended** except for
	experimentation purposes, because it is conceptually and numerically risky to *jointly*
	fit the sequence of `Teq` values, as opposed to fitting each of them individually,
	as done by the recommended function `nearest_Teq()`.
	
	This is both the slowest and most complete version of this calculation.
	It is expected to yield an `uarray` with reasonably accurate covariance between the
	`Teq` values, but also between `Teq` and all other variables. Usually, these `Teq`
	values are strongly correlated with the corresponding values of `D47`.
	
	A faster but incomplete and potentially less accurate version of this calculation is
	provided by `lazy_joint_nearest_Teq()`.
	"""

	N = D47.size
	N47 = D47_calib_coefs.size
	N48 = D48_calib_coefs.size
	
	def fun(j, *args):

		args = _np.array(args)
		D47_n = args[:N]
		D48_n = args[N:2*N]
		D47_calib_coefs_n = args[-N48-N47:-N48]
		D48_calib_coefs_n = args[-N48:]
			
		params = _lmfit.Parameters()
		a0, a2 = _D47_approx_calib_coefs
		for k in range(N):
			params.add(f'T{k}', value = ((D47_n[k] - a0) / a2)**-.5 - 273.15)
		
		D47_u = _cd.uarray(_uc.correlated_values(D47_n, D47.covar))
		D48_u = _cd.uarray(_uc.correlated_values(D48_n, D48.covar))
		D47_calib_coefs_u = _cd.uarray(_uc.correlated_values(D47_calib_coefs_n, D47_calib_coefs.covar))
		D48_calib_coefs_u = _cd.uarray(_uc.correlated_values(D48_calib_coefs_n, D48_calib_coefs.covar))

		def cost_fun(
			p,
			ignore_calib_uncertainties = ignore_calib_uncertainties,
		):
			T = _np.array([p[f'T{k}'] for k in range(N)])
			R = _cd.uarray(_np.concatenate((
				D47_u - D4x_calib_function(T, D47_calib_coefs_u, ignore_calib_uncertainties = ignore_calib_uncertainties),
				D48_u - D4x_calib_function(T, D48_calib_coefs_u, ignore_calib_uncertainties = ignore_calib_uncertainties),
			)))
			
			invS = _np.linalg.inv(R.covar)
			L = _cholesky(invS)

# 			R_n = _np.concatenate((
# 				D47_n - D4x_calib_function(T, D47_calib_coefs_n, ignore_calib_uncertainties = ignore_calib_uncertainties),
# 				D48_n - D4x_calib_function(T, D48_calib_coefs_n, ignore_calib_uncertainties = ignore_calib_uncertainties),
# 			))

			return L @ R.n
		
		minresult = _lmfit.minimize(cost_fun, params, method = 'least_squares', scale_covar = False, jac = '3-point')
		# slower but yields very similar results:
		# minresult = _lmfit.minimize(cost_fun, params, method = 'powell', scale_covar = False)

		return minresult.params[f'T{j}'].value
	
	wrapped_fun = _uc.wrap(fun)
	Teq = _cd.uarray([wrapped_fun(_, *D47, *D48, *D47_calib_coefs, *D48_calib_coefs) for _ in range(N)])
	
	R = _cd.uarray(_np.concatenate((
		D47 - D4x_calib_function(Teq.n, D47_calib_coefs, ignore_calib_uncertainties = ignore_calib_uncertainties),
		D48 - D4x_calib_function(Teq.n, D48_calib_coefs, ignore_calib_uncertainties = ignore_calib_uncertainties),
	)))
	
	p = _np.zeros((N,))
	for k in range(N):
		r = R[k::N]
		z2 = r.m
		p[k] = 1-_chi2.cdf(z2, 2)

	return Teq, p


def lazy_joint_nearest_Teq(
	D47: _cd.uarray,
	D48: _cd.uarray,
	D47_calib_function = D47_calib_function,
	D48_calib_function = D48_calib_function,
	ignore_calib_uncertainties: bool = False,
):
	"""
	Returns a `correldata.uarray` of T values which are *jointly* closest (in the OGLS sense)
	to a sequence of (Δ<sub>47</sub>, Δ<sub>48</sub>) pairs. Also returns an array of
	corresponding p-values taking into account errors in Δ<sub>47</sub> and Δ<sub>48</sub>
	(and any covariance between the two) as well as errors in the Δ<sub>47</sub> and
	Δ<sub>48</sub> calibrations.

	Caution: the use of this function is **not generally recommended** except for
	experimentation purposes, because it is conceptually and numerically risky to *jointly*
	fit the sequence of `Teq` values, as opposed to fitting each of them individually,
	as done by the recommended function `nearest_Teq()`.
	
	This is a faster but incomplete version of this calculation. It is expected to yield an
	`uarray` with roughly accurate covariance between the `Teq` values, but without computing
	the covariance with any other variables.
	
	A slower but complete and more accurate version of this calculation is provided by
	`joint_nearest_Teq()`.
	"""

	N = D47.size

	params = _lmfit.Parameters()
	a0, a2 = _D47_approx_calib_coefs
	for k in range(N):
		params.add(f'T{k}', value = ((D47[k].n - a0) / a2)**-.5 - 273.15)
	
	def cost_fun(
		p,
		ignore_calib_uncertainties = ignore_calib_uncertainties,
	):
		T = _np.array([p[f'T{k}'] for k in range(N)])
		R = _cd.uarray(_np.concatenate((
			D47 - D47_calib_function(T, ignore_calib_uncertainties = ignore_calib_uncertainties),
			D48 - D48_calib_function(T, ignore_calib_uncertainties = ignore_calib_uncertainties),
		)))
		
		invS = _np.linalg.inv(R.covar)
		L = _cholesky(invS)
		return L @ R.n
	
	minresult = _lmfit.minimize(cost_fun, params, method = 'least_squares', scale_covar = False, jac = '3-point')

	Teq = _cd.uarray([minresult.uvars[f'T{k}'] for k in range(N)])

	R = _cd.uarray(_np.concatenate((
		D47 - D47_calib_function(Teq.n, ignore_calib_uncertainties = ignore_calib_uncertainties),
		D48 - D48_calib_function(Teq.n, ignore_calib_uncertainties = ignore_calib_uncertainties),
	)))
	
	p = _np.zeros((N,))
	for k in range(N):
		r = R[k::N]
		z2 = r.m
		p[k] = 1-_chi2.cdf(z2, 2)

	return Teq, p


def projected_Teq(
	D47: _cd.uarray,
	D48: _cd.uarray,
	kinetic_slope: (float | _uc.UFloat),
	D47_calib_coefs: _cd.uarray = D47_calib_coefs,
	D48_calib_coefs: _cd.uarray = D48_calib_coefs,
	ignore_calib_uncertainties: bool = False,
):

	D47 = _cd.uarray(D47)
	D48 = _cd.uarray(D48)
	N = D47.size
	N47c = D47_calib_coefs.size
	N48c = D48_calib_coefs.size
	T = D47 * 0
	for k in range(N):

		# function to solve
		def fun(t, *args): # args = (D47, D48, kinetic_slope, *D47_calib_coefs, *D48_calib_coefs)
			return (
				args[1]
				- _np.sum([
					c / (t[0] + 273.15)**k
					for k,c in enumerate(args[-N48c:])
				])
				- args[2] * (
					args[0]
					- _np.sum([
						c / (t[0] + 273.15)**k
						for k,c in enumerate(args[-N47c-N48c:-N48c])
					])
				)
			)

		def g(*args):
			return _fsolve(fun, [100.], args = args)[0]
		
		wg = _uc.wrap(g)

		T[k] = wg(
			D47[k],
			D48[k],
			kinetic_slope,
			*D47_calib_coefs,
			*D48_calib_coefs,
		)
		
	return T

	
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


_typer.rich_utils.STYLE_HELPTEXT = ''

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

	data = _cd.read_data(datastring)

	Teq, p = nearest_Teq(data['D47'], data['D48'])
	data['eq_pvalue'] = p
	data['Teq'] = Teq

	if isinstance(kslope, str):
		kslope = kslope.split(')')[0]
		kslope = kslope.split('(')
		kslope = _uc.ufloat(float(kslope[0]), float(kslope[1]))

		Tkp = projected_Teq(data['D47'], data['D48'], kinetic_slope = kslope)

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

	out = _cd.data_string(
		data,
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

# _np.set_printoptions(precision = 4, linewidth = 1000)
# 
# X = _cd.uarray(_uc.correlated_values([0.25, 0.64], _np.eye(2)*25e-6))
# Y = _cd.uarray(_uc.correlated_values([0.155, 0.3], _np.eye(2)*225e-6))
# 
# T1 = lazy_nearest_Teq(X, Y, D47_calib_function, D48_calib_function)[0]
# print(T1)
# 
# T2 = nearest_Teq(X, Y, D47_calib_coefs, D48_calib_coefs)
# print(T2)
# 
# slope = _uc.ufloat(-1, 0.1)
# 
# T3 = projected_Teq(X, Y, slope, D47_calib_coefs, D48_calib_coefs)
# print(T3)
