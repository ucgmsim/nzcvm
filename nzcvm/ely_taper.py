"""Ely et al. (2010) near-surface velocity taper.

This module applies the near-surface velocity taper described by Ely et al.
(2010) to smoothly transition from a tomography-based velocity model to a
near-surface geotechnical layer (GTL) defined by a Vs30-based relation.

References
----------
Ely, G. P., Jordan, T. H., Small, P., & Maechling, P. J. (2010).
A Vs30-derived near-surface seismic velocity model.
*Abstracts, Annual Meeting of the Southern California Earthquake Center*, 174.
"""

import functools

import numpy as np
import xarray as xr

from nzcvm.qualities import Qualities, QualitiesSchema

# Brocher Vp/Vs relations, converted to accept and return m/s instead of km/s using sympy.
BROCHER_VP_COEFFS = xr.DataArray(
    np.array([-2.51e-11, 2.683e-07, -0.0008206, 2.0947, 940.9], dtype=np.float32),
    dims=["degree"],
    coords={"degree": [4, 3, 2, 1, 0]},
)
VP_FROM_VS_RELATION = functools.partial(xr.polyval, coeffs=BROCHER_VP_COEFFS)

BROCHER_DENSITY_COEFFS = xr.DataArray(
    np.array(
        [1.06e-16, -4.3e-12, 6.71e-08, -0.00047211, 1.6612, 0.0], dtype=np.float32
    ),
    dims=["degree"],
    coords={"degree": [5, 4, 3, 2, 1, 0]},
)
DENSITY_RELATION = functools.partial(xr.polyval, coeffs=BROCHER_DENSITY_COEFFS)


def _ely_vs_profile(
    depth: xr.DataArray,
    vs30: xr.DataArray,
    vp_at_z_t: xr.DataArray,
    vs_at_z_t: xr.DataArray,
    depth_t: float,
) -> xr.Dataset:
    """Compute the Ely GTL velocity profile at each depth value.

    Parameters
    ----------
    depth :
        Depth values (metres, positive downwards).  Values should satisfy
        ``0 <= depth <= depth_t``.
    vs30 :
        Site-average shear-wave velocity over the top 30 m (m s⁻¹).
    vp_at_z_t :
        P-wave velocity at the reference depth *depth_t* from the underlying
        tomography model (m s⁻¹).
    vs_at_z_t :
        S-wave velocity at the reference depth *depth_t* from the underlying
        tomography model (m s⁻¹).
    depth_t :
        Reference depth (metres) at which the taper meets the tomography.

    Returns
    -------
    Qualities
        Ely GTL velocities and derived densities at each depth point.
    """
    depth_norm = depth / depth_t
    depth_norm_sq = np.square(depth_norm)
    f = depth_norm + (2 / 3) * (depth_norm - depth_norm_sq)
    g = 0.5 - 5 * depth_norm + 1.5 * depth_norm_sq + 3 * np.sqrt(depth_norm)

    vs = f * vs_at_z_t + g * vs30
    vp_from_vs30 = VP_FROM_VS_RELATION(vs30)
    vp = f * vp_at_z_t + g * vp_from_vs30
    rho = DENSITY_RELATION(vp)
    # EMOD3D derives anelastic attenuation directly from velocity as
    # Qs = 50 * Vs and Qp = 100 * Vs (Vs in km/s).  The tapered vs here is in
    # m/s, so scale by 1/1000 to match the tomography and basin generators and
    # let Q follow the tapered velocity through the GTL.
    qs = 50.0 * vs / 1000.0
    qp = 100.0 * vs / 1000.0
    alpha = xr.full_like(rho, 1.0)

    return xr.Dataset(
        {"rho": rho, "vs": vs, "vp": vp, "qp": qp, "qs": qs, "alpha": alpha}
    )


def ely_vs_profile(
    depth: xr.DataArray,
    vs30: xr.DataArray,
    vp_at_z_t: xr.DataArray,
    vs_at_z_t: xr.DataArray,
    depth_t: float,
) -> Qualities:
    """Compute the Ely GTL velocity profile at each depth value.

    Parameters
    ----------
    depth :
        Depth values (metres, positive downwards).  Values should satisfy
        ``0 <= depth <= depth_t``.
    vs30 :
        Site-average shear-wave velocity over the top 30 m (m s⁻¹).
    vp_at_z_t :
        P-wave velocity at the reference depth *depth_t* from the underlying
        tomography model (m s⁻¹).
    vs_at_z_t :
        S-wave velocity at the reference depth *depth_t* from the underlying
        tomography model (m s⁻¹).
    depth_t :
        Reference depth (metres) at which the taper meets the tomography.

    Returns
    -------
    Qualities
        Ely GTL velocities and derived densities at each depth point.
    """
    dset = _ely_vs_profile(depth, vs30, vp_at_z_t, vs_at_z_t, depth_t)
    return QualitiesSchema.from_dataset(dset)
