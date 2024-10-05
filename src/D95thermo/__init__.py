"""
Estimate carbonate formation temperatures from dual clumped isotope measurements
"""

__author__    = 'Mathieu Daëron'
__contact__   = 'daeron@lsce.ipsl.fr'
__copyright__ = 'Copyright (c) 2024 Mathieu Daëron'
__license__   = 'MIT License - https://opensource.org/licenses/MIT'
__date__      = '2024-10-01'
__version__   = '0.1.0'

import numpy as _np
import ogls as _ogls
import uncertainties as _uc
import lmfit as _lmfit

from uncertainties import unumpy as _unp
from D47calib import OGLS23 as _D47approx
from D47calib import D47calib as _D47calib
from matplotlib import pyplot as _ppl
from matplotlib.patches import Ellipse as _Ellipse
from scipy.stats import chi2 as _chi2
from scipy.linalg import eigh as _eigh
from scipy.linalg import cholesky as _cholesky
from scipy.optimize import fsolve as _fsolve


def smart_type(x):
	'''
	Tries to convert string `x` to a float if it includes a decimal point, or
	to an integer if it does not. If both attempts fail, return the original
	string unchanged.
	'''
	try:
		y = float(x)
	except ValueError:
		return x
	if '.' not in x:
		return int(y)
	return y


_D47approx = _D47calib(
	samples = _D47approx.samples,
	T = _D47approx.T,
	sT = _D47approx.sT,
	D47 = _D47approx.D47,
	sD47 = _D47approx.sD47,
	degrees = [0,2],
)
_D47approx.regress()

def error_ellipses(
	X,
	Y,
	p = 0.95,
	ax = None,
	**kwargs,
):

	kwargs = dict(fc = 'None', ec = 'k', lw = 0.7) | kwargs

	r2 = _chi2.ppf(p, 2)
	out = []

	for x, y in zip(X, Y):

		val, vec = _eigh(_uc.covariance_matrix((x, y)))
		width, height = 2 * (val[:, None] * r2)**0.5
		angle = _np.degrees(_np.arctan2(*vec[::-1, 0]))

		if ax is None:
			ax = _ppl.gca()

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
	return out

		

def _compute_D48_calib_coefficients():
	"""
	Based on Fiebig et al. (2021)
	"""


	data = '''
	Sample	T	T_SE	D48	D48_2SE
	LGB-2	7.9	0.2	0.260	0.023
	DVH-2	33.7	0.2	0.246	0.023
	DHC2-8	33.7	0.2	0.237	0.015
	CA120	120	2	0.174	0.035
	CA170	170	2	0.168	0.031
	CA200	200	2	0.171	0.029
	CA250A	250	2	0.148	0.031
	CA250B	250	2	0.141	0.029
	CM351	726.85	10	0.120	0.017
	ETH1-1100	1100	10	0.118	0.022
	ETH2-1100	1100	10	0.123	0.022
	'''.split('\n')[2:-1]

	data = [l.split('\t') for l in data]
	data = [{
		'Sample': l[1],
		'T': float(l[2]),
		'sT': float(l[3]),
		'D48': float(l[4]),
		'sD48': float(l[5])/2,
	} for l in data]
	
	# specify D48 values with covariance
	D48 = _np.array(_uc.correlated_values(
		[l['D48'] for l in data],
		_np.diag([l['sD48']**2 for l in data])
	))

	# specify T values with covariance
	CM_T = _np.diag([l['sT']**2 for l in data])
	CM_T[1,2] = CM_T[1,1]
	CM_T[2,1] = CM_T[1,1]
	T = _np.array(_uc.correlated_values([l['T'] for l in data], CM_T))
	
	# D64 predictions with covariance
	a1 =  6.002
	a2 = -1.299e4
	a3 =  8.996e6
	a4 = -7.423e8
	
	D64_theory = (
		a1 / (273.15 + T)
		+ a2 / (273.15 + T)**2
		+ a3 / (273.15 + T)**3
		+ a4 / (273.15 + T)**4
	)

	# affine regression of the form D48 = b0 + b1 * D64_theory
	R = _ogls.Polynomial(
		X = [_.n for _ in D64_theory],
		sX = _np.array(_uc.covariance_matrix(D64_theory)),
		Y = [_.n for _ in D48],
		sY = _np.array(_uc.covariance_matrix(D48)),
		degrees = [0,1],
	)
	R.regress()
	
	b0, b1 = _uc.correlated_values(R.bfp.values(), R.bfp_CM)
	
	a0 = b0
	a1 *= b1
	a2 *= b1
	a3 *= b1
	a4 *= b1
	
	return _np.array([a0, a1, a2, a3, a4])

def D4x_calib_function(
	T,
	coefs,
	return_without_uncertainties = False,
	ignore_calib_uncertainties = False,
):
	"""
	If `return_without_uncertainties` is False, returns one or more ufloat values.
	In that case, if T is a ufloat or an array of ufloats, the resulting D4x ufloat
	values will account for this source of uncertainty, but if T is a float or an
	array of floats, the D4x ufloat values will only account for uncertainties in
	the calibration coefficients. If `return_without_uncertainties` is True, returns
	the D4x values without error propagation of any kind.
	"""
	if ignore_calib_uncertainties:
		coefs = _unp.nominal_values(coefs)
	D4x = _np.sum(
		[
		coefs[k] / (T + 273.15)**k
		for k in range(coefs.size)
		],
		axis = 0,
	)
	if return_without_uncertainties:
		return _unp.nominal_values(D48)
	return D4x

# D47_calib_coefs from OGLS23 (D47calib v1.3.1)
D47_calib_coefs = _np.array(_uc.correlated_values_norm(
	[
		(0.17437754366432887, 4.911105567257293e-3),
		( -18.14215245127414,    5.632326472234856),
		(42.65722989162373e3,   1.27712751715908e3),
	],
	[
		[ 1.        , -0.93797005,  0.8865771 ],
		[-0.93797005,  1.        , -0.98994249],
		[ 0.8865771 , -0.98994249,  1.        ],
	]
))


def D47_calib_function(
	T,
	coefs = D47_calib_coefs,
	return_without_uncertainties = False,
	ignore_calib_uncertainties = False,
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


D48_calib_coefs = _np.array(_uc.correlated_values_norm(
	[
		(1.244445077e-1, 4.830177847e-3),
		(   6.166790011, 3.866500673e-1),
		(-1.334665149e4, 8.368184562e2),
		( 9.242992826e6, 5.795241595e5),
		(-7.626804774e8, 4.781911778e7),
	],
	[
		[ 1.         , -0.701167327,  0.701167327, -0.701167327,  0.701167327],
		[-0.701167327,  1.         , -1.         ,  1.         , -1.         ],
		[ 0.701167327, -1.         ,  1.         , -1.         ,  1.         ],
		[-0.701167327,  1.         , -1.         ,  1.         , -1.         ],
		[ 0.701167327, -1.         ,  1.         , -1.         ,  1.         ],
	]
))


def D48_calib_function(
	T,
	coefs = D48_calib_coefs,
	return_without_uncertainties = False,
	ignore_calib_uncertainties = False,
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


def plot_D95_equilibrium(
	Tmin = 0,
	Tmax = 1000,
	NT = 101,
	Tmarkers = [0, 25, 100, 250, 1000],
	kwargs_Tmarkers = {},
	show_Tmarker_labels = True,
	kwargs_Tmarker_labels = {},
	show_Tmarker_ellipses = False,
	kwargs_Tmarker_ellipses = {},
	show_Tmarker_errorbars = False,
	kwargs_Tmarker_errorbars = {},
	show_eqline = True,
	kwargs_eqline = {},
	show_D47ci = True,
	kwargs_D47ci = {},
	show_D48ci = True,
	kwargs_D48ci = {},
	ci_pvalue = 0.95,
	ax = None,
	xlabel = '$Δ_{47}$   [‰]',
	ylabel = '$Δ_{48}$   [‰]',
	lw = 0.7,
):
	
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
			plot_elements['Tme'] = error_ellipses(
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
		Ti = Ti,
		Xe = Xe,
		Ye = Ye,
		Tm = Tmarkers,
		Xm = Xm,
		Ym = Ym,
	)
	
	return data, plot_elements


def nearest_Teq(
	X,
	Y,
	D47_calib_function = D47_calib_function,
	D48_calib_function = D48_calib_function,
	ignore_calib_uncertainties = False,
):
	"""
	Returns an array of T ufloats which are closest (in the least-squares sense)
	to each (x, y) pair, along with an array of corresponding p-values taking into
	account errors in X and Y (both of them being potentially covariant) and those
	in the D47 and D48 calibrations.
	"""

	N = X.size
	_X = _unp.nominal_values(X)

	params = _lmfit.Parameters()
	for k in range(N):
		params.add(
			f'T{k}',
			value = ((_X[k] - _D47approx.bfp['a0']) / _D47approx.bfp['a2'])**-.5 - 273.15,
		)
	
	def cost_fun(
		p,
		ignore_calib_uncertainties = ignore_calib_uncertainties,
	):
		T = _np.array([p[f'T{k}'] for k in range(N)])
		R = _np.concatenate((
			X - D47_calib_function(T, ignore_calib_uncertainties = ignore_calib_uncertainties),
			Y - D48_calib_function(T, ignore_calib_uncertainties = ignore_calib_uncertainties),
		))
		CMr = _np.array(_uc.covariance_matrix(R))
		invS = _np.linalg.solve(CMr, _np.eye(2*N))
		L = _cholesky(invS)
		return L @ _unp.nominal_values(R)
	
	model = _lmfit.Minimizer(cost_fun, params, scale_covar = False)
	minresult = model.minimize(method = 'least_squares')

	Teq = _np.array(_uc.correlated_values(
			(minresult.params[f'T{k}'].value for k in range(N)),
			minresult.covar,
	))

	R = _np.concatenate((
		X - D47_calib_function(_unp.nominal_values(Teq), return_without_uncertainties = ignore_calib_uncertainties),
		Y - D48_calib_function(_unp.nominal_values(Teq), return_without_uncertainties = ignore_calib_uncertainties),
	))
	CMr = _np.array(_uc.covariance_matrix(R))
	
	p = _np.zeros((N,))
	for k in range(N):
		r = _unp.nominal_values(R[k::N])
		cm = CMr[k::N,:][:,k::N]
		invS = _np.linalg.solve(cm, _np.eye(2))
		z = r.T @ invS @ r
		p[k] = 1-_chi2.cdf(z, 2)

	
	return Teq, p


def projected_Teq(
	X,
	Y,
	slope,
	D47_calib_coefs = D47_calib_coefs,
	D48_calib_coefs = D48_calib_coefs,
	ignore_calib_uncertainties = False,
):

	X = _np.asarray(X)
	Y = _np.asarray(Y)
	N = X.size
	N47c = D47_calib_coefs.size
	N48c = D48_calib_coefs.size
	T = X * 0
	for k in range(N):

		# function to solve
		def fun(t, *args): # args = (X, Y, slope, *D47_calib_coefs, *D48_calib_coefs)
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
			X[k],
			Y[k],
			slope,
			*D47_calib_coefs,
			*D48_calib_coefs,
		)
		
	return T


def T_ellipses(T, p = 0.95, ax = None, **kwargs):
	return error_ellipses(
		D47_calib_function(T),
		D48_calib_function(T),
		p = p,
		ax = ax,
		**kwargs,
	)


def save_array(
	X,
	varname,
	filename,
	labels = None,
	sep = ',',
	fmt_nv = '.9e',
	fmt_se = '.9e',
	fmt_cm = '.9f',
):
	if labels is None:
		labels = [str(k+1) for k in range(X.size)]
	with open(filename, 'w') as fid:
		fid.write(f'{sep}{varname}{sep}SE{sep}correl')
		nv = _unp.nominal_values(X)
		se = _unp.std_devs(X)
		cm = _np.array(_uc.correlation_matrix(X))
		for k in range(X.size):
			fid.write(f'\n{labels[k]}{sep}{nv[k]:{fmt_nv}}{sep}{se[k]:{fmt_se}}{sep}' + sep.join([f'{cm[j,k]:{fmt_cm}}' for j in range(X.size)]))
	
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


# def read_data(data, sep = ','):
# 	data = [[smart_type(e.strip()) for e in l.split(sep)] for l in data.split('\n')]
# 	N = len(data) - 1
# 	Ncol = _np.max([len(l) for l in data])
# 
# 	work = {}
# 	for j in range(Ncol):
# 		try:
# 			cf = data[0][j]
# 			if cf not in ['SE', 'correl', 'covar', '']:
# 				work[cf] = _np.array([l[j] for l in data[1:]])
# 				of = cf
# 			elif cf == 'SE':
# 				work[of+'_SE'] = _np.array([l[j] for l in data[1:]])
# 			elif cf == 'correl':
# 				work[of+'_correl'] = _np.array([l[j:j+N] for l in data[1:]])
# 			elif cf == 'covar':
# 				work[of+'_covar'] = _np.array([l[j:j+N] for l in data[1:]])
# 		except IndexError:
# 			pass
# 
# 	result = {}
# 	for k in work:
# 		if k.endswith('_SE') or k.endswith('_correl') or k.endswith('_covar'):
# 			continue
# 		if (k + '_covar') in work:
# 			if (k + '_SE') in work:
# 				raise KeyError(f'Too much information: both SE and covar are specified for variable "{k}".')
# 			result[k] = _np.array(_uc.correlated_values(work[k], work[k+'_covar']))
# 		elif (k + '_correl') in work:
# 			if (k + '_SE') in work:
# 				result[k] = _np.array(_uc.correlated_values_norm([*zip(work[k], work[k+'_SE'])], work[k+'_correl']))
# 			else:
# 				raise KeyError('Not enough information: Correl is specified without SE for variable "{k}".')
# 		elif (k + '_SE') in work:
# 			result[k] = _np.array(_uc.correlated_values_norm([*zip(work[k], work[k+'_SE'])], _np.eye(N)))
# 		else:
# 			result[k] = work[k]
# 
# 	return result

def read_data(data, sep = ','):
	data = [[smart_type(e.strip()) for e in l.split(sep)] for l in data.split('\n')]
	N = len(data) - 1

	values, se, correl, covar = {}, {}, {}, {}
	j = 0
	while j < len(data[0]):
		field = data[0][j]
		if not (
			field.startswith('SE_')
			or field.startswith('correl_')
			or field.startswith('covar_')
			or field == 'SE'
			or field == 'correl'
			or field == 'covar'
			or len(field) == 0
		):
			values[field] = _np.array([l[j] for l in data[1:]])
			j += 1
			oldfield = field
		elif field.startswith('SE_'):
			se[field[3:]] = _np.array([l[j] for l in data[1:]])
			j += 1
		elif field == 'SE':
			se[oldfield] = _np.array([l[j] for l in data[1:]])
			j += 1
		elif field == 'SE':
			se[oldfield] = _np.array([l[j] for l in data[1:]])
			j += N
		elif field.startswith('correl_'):
			correl[field[7:]] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field == 'correl':
			correl[oldfield] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field.startswith('covar_'):
			covar[field[6:]] = _np.array([l[j:j+N] for l in data[1:]])
			j += N
		elif field == 'covar':
			covar[oldfield] = _np.array([l[j:j+N] for l in data[1:]])
			j += N

	nakedvalues = {}
	for k in [_ for _ in values]:
		if (
			k not in se
			and k not in correl
			and k not in covar
		):
			nakedvalues[k] = values.pop(k)

	for k in covar:
		if k in values:
			if k in se:
				raise KeyError(f'Too much information: both SE and covar are specified for variable "{k}".')

	for k in correl:
		if k in values:
			if k not in se:
				raise KeyError(f'Not enough information: correl is specified without SE for variable "{k}".')
			if k in covar:
				raise KeyError(f'Too much information: both correl and covar are specified for variable "{k}".')
			covar[k] = _np.diag(se[k]) @ correl[k] @ _np.diag(se[k])
		else:
			for j1 in values:
				for j2 in values:
					if k == f'{j1}_{j2}':
						covar[f'{j1}_{j2}'] = _np.diag(se[j1]) @ correl[k] @ _np.diag(se[j2])
						covar[f'{j2}_{j1}'] = covar[f'{j1}_{j2}'].T

	for k in se:
		if k not in covar:
			covar[k] = _np.diag(se[k]**2)

	for k in [_ for _ in covar]:
		if k not in values:
			for j1 in values:
				for j2 in values:
					if k == f'{j1}_{j2}':
						covar[f'{j2}_{j1}'] = covar[f'{j1}_{j2}'].T

	X = _np.array([_ for k in values for _ in values[k]])
	CM = _np.zeros((X.size, X.size))
	for i, vi in enumerate(values):
		for j, vj in enumerate(values):
			if vi == vj:
				if vi in covar:
					CM[N*i:N*i+N,N*j:N*j+N] = covar[vi]
			else:
				if f'{vi}_{vj}' in covar:
					CM[N*i:N*i+N,N*j:N*j+N] = covar[f'{vi}_{vj}']
	
	corvalues = _np.array(_uc.correlated_values(X, CM))

	result = nakedvalues
	
	for i,k in enumerate(values):
		result[k] = corvalues[i*N:i*N+N]

	return result


def read_data_from_file(filename, sep = ','):
	with open(filename) as fid:
		return read_data(fid.read(), sep = sep)

if __name__ == '__main__':

# 	coefs = _compute_D48_calib_coefficients()
# 	correl = _uc.correlation_matrix(coefs)
# 	with open('D48_calib_coefs.csv', 'w') as fid:
# 		fid.write(f'coef,value,SE,correl')
# 		for k in range(coefs.size):
# 			fid.write(f'\n{"b"+str(k)},{coefs[k].n:.9e},{coefs[k].s:.9e},' + ','.join([f'{correl[j,k]:.9f}' for j in range(coefs.size)]))	

	slope = _uc.ufloat(-1, 0.1)
	p_cutoff = 0.05
	eq_color = (0,.5,.2)
	diseq_color = (1, 0, .4)

	data = read_data_from_file('example_data.csv')
	X = data['X']
	Y = data['Y']

	Teq, p = nearest_Teq(X, Y)

	Tp = projected_Teq(X[p < p_cutoff], Y[p < p_cutoff], slope)
	
	fig = _ppl.figure(figsize = (6.5,4.5))
	_ppl.title("“$Δ_{95}$ thermometry” ($95=47+48$)")
	
	plot_D95_equilibrium()

	error_ellipses(X, Y, ec = 'k')

	T_ellipses(Teq[p >= p_cutoff], ec = eq_color, fc = (*eq_color, 0.2))
	T_ellipses(Tp, ec = diseq_color, fc = (*diseq_color, 0.2))

	for x, y, t in zip(X[p < p_cutoff], Y[p < p_cutoff], Tp):
		v = _np.array([
			D47_calib_function(t).n - x.n,
			D48_calib_function(t).n - y.n,
		])
		i, j = 0.15, 0.85
		kw = dict(
			color = diseq_color,
			lw = 0,
			width = 0.001,
			head_width = 0.005,
		)
		_ppl.arrow(
			x.n + i * v[0],
			y.n + i * v[1],
			(j-i) * v[0],
			(j-i) * v[1],
			**kw,
		)
		for s in (-1.96, +1.96):
			i, j = 0.2, 0.85
			_t = projected_Teq([x], [y], slope + s * slope.s)[0]
			v = _np.array([
				D47_calib_function(_t).n - x.n,
				D48_calib_function(_t).n - y.n,
			])
			_ppl.arrow(
				x.n + i * v[0],
				y.n + i * v[1],
				(j-i) * v[0],
				(j-i) * v[1],
				alpha = 0.25,
				**kw,
			)

	for x, y, t, pv in zip(X[p >= p_cutoff], Y[p >= p_cutoff], Teq, p[p >= p_cutoff]):
		_ppl.text(
			x.n, y.n + 5*y.s,
			'($Δ_{47}, Δ_{48}$)\nobservation',
			ha = 'center', va = 'top', size = 8,
		)
		_ppl.text(
			x.n, y.n + 5.5*y.s,
			f'Equil. p-value = {pv:.2f}',
			ha = 'center', va = 'bottom', size = 8, color = eq_color,
		)
		_ppl.text(
			D47_calib_function(t).n + 4*D47_calib_function(t).s, D48_calib_function(t).n - 5 * D48_calib_function(t).s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = eq_color,
		)

	for x, y, t, pv in zip(X[p < p_cutoff], Y[p < p_cutoff], Tp, p[p < p_cutoff]):
		_ppl.text(
			x.n, y.n + 5*y.s,
			'($Δ_{47}, Δ_{48}$)\nobservation',
			ha = 'center', va = 'top', size = 8,
		)
		_ppl.text(
			x.n, y.n + 5.5*y.s,
			f'Equil. p-value = {pv:.0e}',
			ha = 'center', va = 'bottom', size = 8, color = diseq_color,
		)
		_ppl.text(
			D47_calib_function(t).n + D47_calib_function(t).s, D48_calib_function(t).n - 3 * D48_calib_function(t).s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = diseq_color,
		)
		m = 0.5
		_ppl.text(
			(m * x.n + (1-m) * D47_calib_function(t).n),
			(m * y.n + (1-m) * D48_calib_function(t).n),
			'disequilibrium slope\n(with uncertainty)\n',
			ha = 'left', va = 'bottom', size = 8, color = diseq_color,
		)

		_ppl.text(
			0.5, 0.02,
			"""
Inputs: $Δ_{47}$ and $Δ_{48}$ measurements, with arbitrary errors (covariance matrix).
Outputs: equilibrium p-values and T estimates with T covariance matrix fully accounting for $Δ_{47}$ and $Δ_{48}$
measurement uncertainties, $Δ_{47}$ and $Δ_{48}$ calibration uncertainties, and the disequilibrium slope uncertainty.""",
			size = 6.5, va = 'bottom', ha = 'center', transform = _ppl.gca().transAxes,
		)

	_ppl.text(
		1, 1.01, 'M. Daëron 2024-10',
		transform = _ppl.gca().transAxes,
		size = 6,
		alpha = 0.25,
		ha = 'right',
		va = 'bottom',
	)

	_ppl.axis('equal')
	_ppl.axis([0.15, 0.78, None, None])
	_ppl.savefig('example_plot.pdf')
	_ppl.savefig('example_plot.png', dpi = 150)
