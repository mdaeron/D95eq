from correldata import read_data_from_file, save_data_to_file, uarray, data_string
import numpy as _np
import uncertainties as _uc
from warnings import filterwarnings
filterwarnings('ignore', category = FutureWarning)

from matplotlib import pyplot as _ppl
from D95thermo import *
import time

E = Engine()

slope = _uc.ufloat(-1, 0.1)
p_cutoff = 0.05
eq_color = (0,.5,.2)
diseq_color = (1, 0, .4)

N = 8

X = _np.linspace(0.2, 0.65, N)
Y = _np.linspace(0.15, 0.25, N)
X = uarray(_uc.correlated_values(X, _np.diag([1.*.005**2]*N) + _np.ones((N,N))*0.*.005**2))
Y = uarray(_uc.correlated_values(Y, _np.diag([.015**2]*N)))

for k, funD47eq in enumerate((
	E.nearest_D47eq,
	E.joint_nearest_D47eq,
	E.lazy_joint_nearest_D47eq,
)):

	funD47eqname = funD47eq.__name__
	print(f'Start {funD47eqname}()')
	t1 = time.time()
	D47eq, D48eq, p = funD47eq(X, Y)
	t2 = time.time()
	Tp = E.projected_Teq(X, Y, slope)

	fig = _ppl.figure(figsize = (6.5,4.5))
	_ppl.title("“$Δ_{95}$ thermometry” ($47+48=95$)")

	E.plot_D95_equilibrium()

	conf_ellipse(X, Y, ec = 'k')

	conf_ellipse(D47eq[p >= p_cutoff], D48eq[p >= p_cutoff], ec = eq_color, fc = (*eq_color, 0.2))
	E.T_ellipse(Tp[p < p_cutoff], ec = diseq_color, fc = (*diseq_color, 0.2))

	for x, y, t in zip(X[p < p_cutoff], Y[p < p_cutoff], Tp[p < p_cutoff]):
		v = _np.array([
			E.D47_calib_function(t).n - x.n,
			E.D48_calib_function(t).n - y.n,
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
			_t = E.projected_Teq([x], [y], slope + s * slope.s)[0]
			v = _np.array([
				E.D47_calib_function(_t).n - x.n,
				E.D48_calib_function(_t).n - y.n,
			])
			_ppl.arrow(
				x.n + i * v[0],
				y.n + i * v[1],
				(j-i) * v[0],
				(j-i) * v[1],
				alpha = 0.25,
				**kw,
			)

	t = _uc.ufloat(0., 0.1)

	for x, y, xeq, yeq, pv in zip(
		X[p >= p_cutoff],
		Y[p >= p_cutoff],
		D47eq[p >= p_cutoff],
		D48eq[p >= p_cutoff],
		p[p >= p_cutoff],
	):
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
			xeq.n + 4*xeq.s,
			yeq.n - 5 * yeq.s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = eq_color,
		)

	for x, y, xeq, yeq, pv in zip(
		X[p < p_cutoff],
		Y[p < p_cutoff],
		D47eq[p >= p_cutoff],
		D48eq[p >= p_cutoff],
		p[p < p_cutoff],
	):
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
			E.D47_calib_function(t).n + E.D47_calib_function(t).s, E.D48_calib_function(t).n - 3 * E.D48_calib_function(t).s,
			f'T = {t.n:.1f}±{t.s:.1f}°C',
			ha = 'left', va = 'top', size = 8, color = diseq_color,
		)
		m = 0.5
		_ppl.text(
			(m * x.n + (1-m) * E.D47_calib_function(t).n),
			(m * y.n + (1-m) * E.D48_calib_function(t).n),
			'disequilibrium slope\n(with uncertainty)\n',
			ha = 'left', va = 'bottom', size = 8, color = diseq_color,
		)

	# 	_ppl.text(
	# 		0.5, 0.02,
	# 		"""
	# Inputs: $Δ_{47}$ and $Δ_{48}$ measurements, with arbitrary errors (covariance matrix).
	# Outputs: equilibrium p-values and T estimates with T covariance matrix fully accounting for $Δ_{47}$ and $Δ_{48}$
	# measurement uncertainties, $Δ_{47}$ and $Δ_{48}$ calibration uncertainties, and the disequilibrium slope uncertainty.""",
	# 		size = 6.5, va = 'bottom', ha = 'center', transform = _ppl.gca().transAxes,
	# 	)

	_ppl.text(
		1, 1.01, 'M. Daëron 2024-10',
		transform = _ppl.gca().transAxes,
		size = 6,
		alpha = 0.25,
		ha = 'right',
		va = 'bottom',
	)

	_ppl.text(
		.99, .02, f'computed using {funD47eqname}() in {t2-t1:.2f} s',
		transform = _ppl.gca().transAxes,
		size = 8,
		ha = 'right',
		va = 'bottom',
	)

	_ppl.axis('equal')
	_ppl.axis([0.15, 0.78, None, None])
	_ppl.savefig(f'large_example_plot_{k}_{funD47eqname}.pdf')

	data = dict(
		D47 = X,
		D48 = Y,
	)

	try:
		data['pvalue_eq'] = p
		data['Teq'] = Teq
	# 	data['Tkp'] = Tp

	# 	print(data_string(data))
		save_data_to_file(data, f'output_{k}_{funTeqname}.csv')
	except:
		pass
