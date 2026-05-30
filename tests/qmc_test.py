from D95eq import *
from uncertainties import *
from pylab import *
from warnings import filterwarnings
filterwarnings('ignore', category = FutureWarning)

def test_Teq_pdf():
	E = Engine()
	k  =0
	for D47 in [
		ufloat(0.65, 0.01),
		ufloat(0.42, 0.05),
		ufloat(0.35, 0.003),
	]:
		k += 1
		fig = figure()
		ax1 = subplot(211)
		yticks([])
		ax2 = subplot(212, sharex = ax1)
		yticks([])
		for ignore_calib_uncertainties, ax in ((True, ax1), (False, ax2)):
			Ti, pdf, Tqmc = E.Teq_pdf(
				D47,
				ignore_calib_uncertainties = ignore_calib_uncertainties,
				run_qmc = True,
				N_qmc = 1024 if ignore_calib_uncertainties else 4096,
			)
			sca(ax)
			hist(
				Tqmc,
				bins = 50,
				density = True,
				histtype = 'stepfilled',
				color = (1,.8,0,0.5),
			)
			plot(
				Ti,
				pdf,
				'-' + {True: 'g', False: 'r'}[ignore_calib_uncertainties],
				label = {
					True: 'Ignoring calib uncertainties',
					False: 'Accounting for calib uncertainties',
				}[ignore_calib_uncertainties],
			)
			legend()
		xlabel('T')
		fig.savefig(f"tests/qmc_test_{k:03.0f}.pdf")
		close(fig)

if __name__ == '__main__':
	test_Teq_pdf()
