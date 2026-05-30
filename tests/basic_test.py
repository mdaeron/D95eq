from D95eq import *
from uncertainties import UFloat, ufloat
from correldata import uarray
import numpy as np

def test_convert_D47_to_T():
	E = Engine()
	for a in [ufloat(0.65, 0.01), ufloat(0.5, 0.01), ufloat(0.35, 0.003)]:
		t1 = E.T_as_function_of_D47(a.n, ignore_calib_uncertainties = True)
		t2 = E.T_as_function_of_D47(a.n, ignore_calib_uncertainties = False)
		t3 = E.T_as_function_of_D47(a, ignore_calib_uncertainties = True)
		t4 = E.T_as_function_of_D47(a, ignore_calib_uncertainties = False)
		print(f"""
For Δ47 = {a}, Teq = {t1:.2f}
	* {t1:.2f} considering no uncertainties
	* {t2:.2f} considering only calibration uncertainties
	* {t3:.2f} considering only measurement uncertainties
	* {t4:.2f} considering both sources of uncertainty""")



def test_convert_T_float_to_D4x():

	for D47_coefs in (None, 0.1, 10.0):
		for D48_coefs in (None, 0.1, 10.0):
			E = Engine(
				D47_coefs = None if D47_coefs is None else Engine.D47_calib_coefs * D47_coefs,
				D48_coefs = None if D48_coefs is None else Engine.D48_calib_coefs * D48_coefs,
			)

			for t in (0., 19., 100., 1000.):

					print(f"\nT = {t:.1f} °C")

					D47 = E.D47_calib_function(T = t)
					assert isinstance(D47, UFloat)
					print(f"    Δ47 = {D47:.4f} ‰")

					D48 = E.D48_calib_function(T = t)
					assert isinstance(D48, UFloat)
					print(f"    Δ48 = {D48:.4f} ‰")

def test_convert_T_array_to_D4x():

	for D47_coefs in (None, 0.1, 10.0):
		for D48_coefs in (None, 0.1, 10.0):
			for return_without_uncertainties in (True, False):
				E = Engine(
					D47_coefs = None if D47_coefs is None else Engine.D47_calib_coefs * D47_coefs,
					D48_coefs = None if D48_coefs is None else Engine.D48_calib_coefs * D48_coefs,
				)

				t = np.array((0., 19., 100., 1000.))

				print(f"\nD47_coefs = {D47_coefs}")
				print(f"D48_coefs = {D48_coefs}")
				print(f"return_without_uncertainties = {return_without_uncertainties}")
				print(f"T = [{', '.join((f'{x:.1f}' for x in t))}]")

				D47 = E.D47_calib_function(
					T = t,
					return_without_uncertainties = return_without_uncertainties,
				)
				if return_without_uncertainties:
					assert isinstance(D47, np.ndarray)
				else:
					assert isinstance(D47, uarray)
				print(f"    Δ47 = [{', '.join((f'{x:.4f}' for x in D47))}]")

				D48 = E.D48_calib_function(
					T = t,
					return_without_uncertainties = return_without_uncertainties,
				)
				if return_without_uncertainties:
					assert isinstance(D48, np.ndarray)
				else:
					assert isinstance(D48, uarray)
				print(f"    Δ48 = [{', '.join((f'{x:.4f}' for x in D48))}]")
