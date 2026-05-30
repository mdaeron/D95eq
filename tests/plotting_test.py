from D95eq import *
from pylab import *

def test_plots():

	k = 0
	for D47_coefs in (None, 0.1, 10.0):
		for D48_coefs in (None, 0.1, 10.0):
			E = Engine(
				D47_coefs = None if D47_coefs is None else Engine.D47_calib_coefs * D47_coefs,
				D48_coefs = None if D48_coefs is None else Engine.D48_calib_coefs * D48_coefs,
			)

			for t in [
				19.0,
			]:
				k += 1
				fig = figure()
				ax = subplot(111)

				E.plot_D95_equilibrium(
					kwargs_Tmarkers = dict(mec = 'r'),
					show_Tmarker_labels = True,
					kwargs_Tmarker_labels = dict(color = 'r'),
					show_Tmarker_ellipses = True,
					kwargs_Tmarker_ellipses = dict(fc = (0,1,0,.1)),
					kwargs_eqline = dict(color = 'b'),
					kwargs_confidence = dict(fc = (1,0,1,.1)),
					confidence_pvalue = 0.9,
					xlabel = '$Δ_{47}$ (‰)',
					ylabel = '$Δ_{48}$ (‰)',
					lw = 0.25,
				)

				for p, ls in (
					(0.5, '-'),
					(0.95, '--'),
					(0.99, ':'),
				):
					E.T_ellipse(
						T = t,
						p = p,
						Tse = 2,
						ls = ls,
					)
				ax.autoscale_view()
				fig.savefig(f'tests/plotting_test_{k:03.0f}.pdf')
				close(fig)
