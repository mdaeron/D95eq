import numpy as np

from typing import Callable
from numpy.typing import ArrayLike
from scipy.stats import chi2
from scipy.differentiate import derivative
from uncertainties import covariance_matrix

def confidence_band(
	t: ArrayLike,
	fx: Callable,
	fy: Callable,
	p: float = 0.95,
	dt: float = 1e-9,
):
	"""
	Return an (N, 2) array of (x, y) vertices outlining a confidence region, at a given p-value,
	for the central parametric curve ***C*** defined by `x = fx(t)` and `y = fy(t)`.

	This confidence region is defined as the union of confidence ellipses for all points along ***C***.

	**Arguments**
	* `t`: array of values over which to sample ***C***
	* `fx`: parametric function of `t` yielding x values of ***C*** as
	[UFloat](https://pythonhosted.org/uncertainties/tech_guide.html) values
	* `fy`: parametric function of `t` yielding y values of ***C*** as
	[UFloat](https://pythonhosted.org/uncertainties/tech_guide.html) values
	* `p`: p-value for the confidence region to return
	* `p`: p-value for the confidence region to return

	Returns a (N, 2) array of (x, y) vertices.
	"""

	# curve position & covariance
	curve      = lambda _t: np.array([fx(_t).n, fy(_t).n])
	def covariance(_t):
		return np.array(covariance_matrix((fx(_t), fy(_t))))
	# corresponding derivatives
	def deriv(_f, _t, _dt = dt):
		return (_f(float(_t) + _dt) - _f(float(_t) - _dt)) / (2 * _dt)
	mu_dot     = lambda _t: deriv(curve, _t)
	sigma_dot  = lambda _t: deriv(covariance, _t)

	# ellipse discretization
	def ellipse_points(mean, cov, chi2_val, n_pts = 120):
		phi  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
		unit = np.stack([np.cos(phi), np.sin(phi)], axis = 1)
		L    = np.linalg.cholesky(cov)
		return mean + np.sqrt(chi2_val) * (unit @ L.T)

	# find angular positions where a given ellipse is tangent to the union of ellipses
	def envelope_contact_angles(t, chi2_val, n_pts = 2000):
		mu    = curve(t)
		Sigma = covariance(t)
		L     = np.linalg.cholesky(Sigma)
		s     = np.sqrt(chi2_val)

		Lambda     = np.linalg.inv(Sigma)
		Sigma_d    = sigma_dot(t)
		Lambda_dot = -Lambda @ Sigma_d @ Lambda
		mu_d       = mu_dot(t)

		phi   = np.linspace(0, 2 * np.pi, n_pts, endpoint = False)
		u     = np.stack([np.cos(phi), np.sin(phi)], axis = 1)
		delta = s * (u @ L.T)

		term1 = -2.0 * (delta @ (Lambda @ mu_d))
		term2 = np.einsum('ni,ij,nj->n', delta, Lambda_dot, delta)
		dFdt  = term1 + term2

		signs     = np.sign(dFdt)
		crossings = np.where(np.diff(signs) != 0)[0]

		contact_pts = []
		for idx in crossings:
			phi0, phi1 = phi[idx], phi[idx + 1]
			f0,   f1   = dFdt[idx], dFdt[idx + 1]
			phi_c = phi0 - f0 * (phi1 - phi0) / (f1 - f0)
			u_c   = np.array([np.cos(phi_c), np.sin(phi_c)])
			pt    = mu + s * L @ u_c
			contact_pts.append(pt)

		return contact_pts

	# build the upper and lower limits of the envelope
	def build_envelope(ts, chi2_val, means):
		all_contacts = []
		all_t        = []

		for i, t in enumerate(ts):
			pts = envelope_contact_angles(t, chi2_val)
			for pt in pts:
				all_contacts.append(pt)
				all_t.append(i)

		if not all_contacts:
			return None, None

		pts   = np.array(all_contacts)
		t_idx = np.array(all_t)

		upper, lower = [], []

		for i, t in enumerate(ts):
			mask  = t_idx == i
			pts_t = pts[mask]
			if len(pts_t) == 0:
				continue

			i0, i1  = max(0, i - 1), min(len(ts) - 1, i + 1)
			tangent = means[i1] - means[i0]
			normal  = np.array([-tangent[1], tangent[0]])

			for pt in pts_t:
				side = np.dot(pt - means[i], normal)
				if side >= 0:
					upper.append((i, pt))
				else:
					lower.append((i, pt))

		upper.sort(key=lambda x: x[0])
		lower.sort(key=lambda x: x[0])

		upper_pts = np.array([p for _, p in upper])
		lower_pts = np.array([p for _, p in lower])

		return upper_pts, lower_pts

	# Trace the arc of the terminal ellipse that faces outward, running exactly
	# from upper_end to lower_end along the outward-facing side.
	# Strategy: parametrise the full ellipse by angle, find the angles
	# corresponding to upper_end and lower_end, then extract the arc between
	# them that passes through the outward direction.
	def terminal_cap(mean, cov, chi2_val, outward_tangent, upper_end, lower_end, n_pts = 200):
		L    = np.linalg.cholesky(cov)
		Linv = np.linalg.inv(L)
		s    = np.sqrt(chi2_val)

		# Map upper_end and lower_end back to angles in the unit circle
		def point_to_angle(pt):
			u = Linv @ (pt - mean) / s
			return np.arctan2(u[1], u[0])

		phi_upper = point_to_angle(upper_end)
		phi_lower = point_to_angle(lower_end)
		phi_out   = np.arctan2(outward_tangent[1], outward_tangent[0])

		# Normalise all angles relative to phi_upper, on [0, 2π)
		def normalise(phi, ref):
			return (phi - ref) % (2 * np.pi)

		phi_lower_n = normalise(phi_lower, phi_upper)
		phi_out_n   = normalise(phi_out,   phi_upper)

		# The outward arc from phi_upper to phi_lower passes through phi_out.
		# Determine direction: if phi_out_n < phi_lower_n, the outward arc goes
		# forward (increasing angle); otherwise it goes backward.
		if phi_out_n < phi_lower_n:
			# Forward arc: phi_upper → phi_upper + phi_lower_n
			phis = np.linspace(phi_upper, phi_upper + phi_lower_n, n_pts)
		else:
			# Backward arc: phi_upper → phi_upper - (2π - phi_lower_n)
			phis = np.linspace(phi_upper, phi_upper - (2 * np.pi - phi_lower_n), n_pts)

		u   = np.stack([np.cos(phis), np.sin(phis)], axis=1)
		arc = mean + s * (u @ L.T)

		return arc

	chi2_value = chi2.ppf(p, df = 2)
	means = curve(t).T
	covs = np.array([covariance(_) for _ in t])
	upper, lower = build_envelope(t, chi2_value, means)

	# Outward tangents at each tip: unit vector pointing away from curve interior
	tangent_start = means[0]  - means[1]
	tangent_end   = means[-1] - means[-2]

	cap_start = terminal_cap(
			means[0], covs[0], chi2_value, tangent_start,
			upper_end=lower[0],    # polygon arrives via lower[::-1], which ends at lower[0]
			lower_end=upper[0],    # polygon departs via upper, which starts at upper[0]
	)
	cap_end = terminal_cap(
			means[-1], covs[-1], chi2_value, tangent_end,
			upper_end=upper[-1],   # polygon arrives via upper, which ends at upper[-1]
			lower_end=lower[-1],   # polygon departs via lower[::-1], which starts at lower[-1]
	)

	band_x = np.concatenate([
		upper[:, 0],
		cap_end[:, 0],
		lower[::-1, 0],
		cap_start[:, 0],
	])
	band_y = np.concatenate([
			upper[:, 1],
			cap_end[:, 1],
			lower[::-1, 1],
			cap_start[:, 1],
	])

	return np.array([band_x, band_y]).T

confidence_band.__module__ = "D95thermo"
