import numpy as _np
import correldata
from D95thermo import Engine
from D95thermo import _compute_D48_calib_coefficients


coefs = Engine.D47_calib_coefs
data = correldata.CorrelData({
	'degree' : _np.array([k for k,v in enumerate(coefs)]),
	'coef': correldata.uarray(coefs),
})
data.to_csv('D47_calib_coefs.csv', correl_format = 'z.9f')


coefs = _compute_D48_calib_coefficients()
data = correldata.CorrelData({
	'degree' : _np.array([k for k,v in enumerate(coefs)]),
	'coef': correldata.uarray(coefs),
})
data.to_csv('D48_calib_coefs.csv', correl_format = 'z.9f')
