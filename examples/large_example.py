from correldata import read_data_from_file, save_data_to_file, uarray, data_string
import numpy as _np
import uncertainties as _uc

from matplotlib import pyplot as _ppl
from D95thermo import *
import time


slope = _uc.ufloat(-1, 0.1)
p_cutoff = 0.05
eq_color = (0,.5,.2)
diseq_color = (1, 0, .4)

N = 12

X = _np.linspace(0.2, 0.65, N)
Y = _np.linspace(0.15, 0.25, N)
X = uarray(_uc.correlated_values(X, _np.diag([1.*.005**2]*N) + _np.ones((N,N))*0.*.005**2))
Y = uarray(_uc.correlated_values(Y, _np.diag([.015**2]*N)))

for k, funTeq in enumerate((
	nearest_Teq,
	joint_nearest_Teq,
	lazy_joint_nearest_Teq,
)):

	funTeqname = funTeq.__name__
	print(f'Start {funTeqname}()')
	t1 = time.time()
	Teq, p = funTeq(X, Y)
	t2 = time.time()
	Tp = projected_Teq(X, Y, slope)
	
	fig = _ppl.figure(figsize = (6.5,4.5))
	_ppl.title("“$Δ_{95}$ thermometry” ($47+48=95$)")
	
	plot_D95_equilibrium()
	
	error_ellipses(X, Y, ec = 'k')
	
	T_ellipses(Teq[p >= p_cutoff], ec = eq_color, fc = (*eq_color, 0.2))
	T_ellipses(Tp[p < p_cutoff], ec = diseq_color, fc = (*diseq_color, 0.2))
	
	for x, y, t in zip(X[p < p_cutoff], Y[p < p_cutoff], Tp[p < p_cutoff]):
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
	
	for x, y, t, pv in zip(X[p >= p_cutoff], Y[p >= p_cutoff], Teq[p >= p_cutoff], p[p >= p_cutoff]):
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
	
	for x, y, t, pv in zip(X[p < p_cutoff], Y[p < p_cutoff], Tp[p < p_cutoff], p[p < p_cutoff]):
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

	_ppl.text(
		.99, .02, f'computed using {funTeqname}() in {t2-t1:.2f} s',
		transform = _ppl.gca().transAxes,
		size = 8,
		ha = 'right',
		va = 'bottom',
	)
	
	_ppl.axis('equal')
	_ppl.axis([0.15, 0.78, None, None])
	_ppl.savefig(f'large_example_plot_{k}_{funTeqname}.pdf')

	data = dict(
		D47 = X,
		D48 = Y,
	)

	data['pvalue_eq'] = p
	data['Teq'] = Teq
# 	data['Tkp'] = Tp
	
# 	print(data_string(data))
	save_data_to_file(data, f'output_{k}_{funTeqname}.csv')
