from .config import Config as _BaseConfig
from .beams.gaussian import gauss as _gauss_base
from .beams.lg import lg as _lg_base
from .beams.hg import hg as _hg_base
from .beams.ig import ig as _ig_base
from .propagation.propagation import propagation as _prop_base
from .propagation.turbulence import turbulence as _turb_base
from .propagation.lens import (
    lens_phase as _lens_phase,
    apply_lens as _apply_lens,
    lens_focal_spot_size as _lens_focal_spot_size,
    lens_fft_propagation_to_focal as _lens_fft_propagation_to_focal
)
from .propagation.slm import (
    zernike_polynomial, zernike_name, noll_to_nm,
    generate_zernike_map, zernike_basis,
    slm_phase_from_zernike, fit_zernike_to_phase,
    compensate_turbulence_phase, apply_slm_correction,
    simulate_slm, adaptive_optic_correction
)

# --- Global configuration---
_default_config = None

def get_default_config():
    """Return the current global configuration."""
    global _default_config
    if _default_config is None:
        _default_config = Config()  #uses the class defined below
    return _default_config


def set_default_config(cfg):
    """Manually set a new global configuration."""
    global _default_config
    _default_config = cfg


class Config(_BaseConfig):
    """
    Global beam configuration class.
    Creating a new Config automatically sets it as the package default.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_default_config(self)  #update global default when created

    def __repr__(self):
        return super().__repr__()


# --- Ensure beam and propagation wrappers use current config automatically ---
def gauss(cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _gauss_base(cfg=cfg, **kwargs)


def lg(p, l, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _lg_base(p=p, l=l, cfg=cfg, **kwargs)


def hg(n, m, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _hg_base(n=n, m=m, cfg=cfg, **kwargs)


def ig(p, m, beam="e", cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _ig_base(p=p, m=m, beam=beam, cfg=cfg, **kwargs)


def propagation(E, z=None, cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    z = z or cfg.z_default
    return _prop_base(E, z=z, cfg=cfg, **kwargs)


def turbulence(cfg=None, **kwargs):
    cfg = cfg or get_default_config()
    return _turb_base(cfg=cfg, **kwargs)


# --- Lens functions ---
def lens_phase(f, cfg=None):
    """Generate thin lens phase function."""
    cfg = cfg or get_default_config()
    return _lens_phase(cfg=cfg, f=f)


def apply_lens(E, f, cfg=None):
    """Apply lens phase to an optical field."""
    cfg = cfg or get_default_config()
    return _apply_lens(E=E, cfg=cfg, f=f)


def lens_focal_spot_size(f, cfg=None, wavelength=None):
    """Calculate theoretical Airy disk size at focal plane."""
    cfg = cfg or get_default_config()
    return _lens_focal_spot_size(cfg=cfg, f=f, wavelength=wavelength)


def lens_fft_propagation_to_focal(E, f, z_prop=0, cfg=None):
    """Propagate through lens to focal plane."""
    cfg = cfg or get_default_config()
    return _lens_fft_propagation_to_focal(E=E, cfg=cfg, f=f, z_prop=z_prop)


# --- SLM/Zernike functions ---
def slm_phase(coeffs, cfg=None):
    """Generate SLM phase from Zernike coefficients."""
    cfg = cfg or get_default_config()
    X, Y = cfg.grid()
    return slm_phase_from_zernike(coefficients=coeffs, x=X, y=Y)


def slm_correct(E, turbulence_phase=None, zernike_coeffs=None, cfg=None):
    """Apply SLM phase correction."""
    cfg = cfg or get_default_config()
    return simulate_slm(E=E, cfg=cfg, turbulence_phase=turbulence_phase, zernike_coeffs=zernike_coeffs)
