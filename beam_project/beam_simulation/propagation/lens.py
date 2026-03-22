"""
Lens phase functions for focused beam simulation.
"""
import numpy as np
import math


def lens_phase(cfg, f):
    """
    Generate thin lens phase function.
    
    Phase for a thin lens: φ(r) = -k * r² / (2f)
    where k = 2π/λ, r = √(x² + y²), f = focal length
    
    Parameters
    ----------
    cfg : Config
        Configuration object with wavelength and grid info.
    f : float
        Focal length [m].
        
    Returns
    -------
    np.ndarray
        Lens phase screen [rad].
    """
    X, Y = cfg.grid()
    r_sq = X**2 + Y**2
    k = 2 * np.pi / cfg.wavelength
    
    # Thin lens phase
    phi = -k * r_sq / (2 * f)
    return phi


def apply_lens(E, cfg, f):
    """
    Apply lens phase to an optical field.
    
    Parameters
    ----------
    E : np.ndarray
        Complex input field.
    cfg : Config
        Configuration object.
    f : float
        Focal length [m].
        
    Returns
    -------
    np.ndarray
        Field with lens phase applied.
    """
    phi_lens = lens_phase(cfg, f)
    return E * np.exp(1j * phi_lens)


def lens_focal_spot_size(cfg, f, wavelength=None):
    """
    Calculate theoretical Airy disk diameter at focal plane.
    
    For a circular aperture, the first zero of J1 occurs at 1.22λf/D.
    
    Parameters
    ----------
    cfg : Config
        Configuration object.
    f : float
        Focal length [m].
    wavelength : float, optional
        Wavelength [m]. Uses cfg.wavelength if None.
        
    Returns
    -------
    float
        Airy disk radius (first zero) [m].
    """
    wavelength = wavelength or cfg.wavelength
    D = cfg.size_x * cfg.pixel_size  # Aperture size
    
    # Airy disk radius
    r_airy = 1.22 * wavelength * f / D
    return r_airy


def lens_fft_propagation_to_focal(E, cfg, f, z_prop=0):
    """
    Propagate through lens to focal plane using FFT method.
    
    The focal plane is at distance f from the lens.
    
    Parameters
    ----------
    E : np.ndarray
        Input field before lens.
    cfg : Config
        Configuration object.
    f : float
        Focal length [m].
    z_prop : float
        Additional propagation distance after focal plane [m].
        
    Returns
    -------
    np.ndarray
        Field at focal plane.
    """
    from .propagation import propagation
    
    # Apply lens
    E_lensed = apply_lens(E, cfg, f)
    
    # Propagate to focal plane (distance f)
    total_z = f + z_prop
    E_focal = propagation(E_lensed, z=total_z, cfg=cfg)
    
    return E_focal


def zoom_lens_system(cfg_in, cfg_out, f1, f2):
    """
    Create phase for a zoom lens system (two lens configuration).
    
    Parameters
    ----------
    cfg_in : Config
        Input grid configuration.
    cfg_out : Config
        Output grid configuration.
    f1 : float
        Focal length of first lens [m].
    f2 : float
        Focal length of second lens [m].
        
    Returns
    -------
    np.ndarray
        Combined lens phase.
    """
    # For now, return phase for single effective lens
    # In a real system, would need proper beam resizing
    f_eff = 1 / (1/f1 + 1/f2)
    return lens_phase(cfg_out, f_eff)
