"""
Zernike polynomials for SLM phase correction.

Zernike polynomials form an orthogonal basis over the unit circle,
commonly used to represent optical aberrations.

Noll Index Convention:
  1: Z1 = 1 (Piston)
  2: Z2 = 2ρcos(θ) (Tip/Tilt X)
  3: Z3 = 2ρsin(θ) (Tip/Tilt Y)
  4: Z4 = √3(2ρ²-1) (Defocus)
  5: Z5 = √6ρ²sin(2θ) (Astigmatism 45°)
  6: Z6 = √6ρ²cos(2θ) (Astigmatism 0°)
  7: Z7 = √8(3ρ²-2)ρsin(θ) (Coma X)
  8: Z8 = √8(3ρ²-2)ρcos(θ) (Coma Y)
  9: Z9 = √8ρ³sin(3θ) (Trefoil X)
 10: Z10 = √8ρ³cos(3θ) (Trefoil Y)
 11: Z11 = √5(6ρ⁴-6ρ²+1) (Spherical)
 ... and so on

References:
- Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence.
- Mahajan, V. N. (1994). Zernike polynomials and wavefront fitting.
"""
import numpy as np
from scipy.special import factorial


def normalize_rho(x, y):
    """
    Normalize radial coordinate to unit circle.
    
    Parameters
    ----------
    x, y : np.ndarray
        Cartesian coordinates [m].
        
    Returns
    -------
    rho : np.ndarray
        Normalized radial coordinate (0 to 1 on unit circle).
    """
    r = np.sqrt(x**2 + y**2)
    # Find max radius to normalize
    r_max = np.max(r)
    return r / r_max if r_max > 0 else r


def zernike_polynomial(n, m, rho, theta):
    """
    Compute Zernike polynomial Z_n^m(rho, theta).
    
    Uses radial and azimuthal indices (n, m) where:
    - n: radial order (non-negative integer)
    - m: azimuthal frequency (-n ≤ m ≤ n, same parity as n)
    
    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Azimuthal order.
    rho : np.ndarray
        Normalized radial coordinate (0 to 1).
    theta : np.ndarray
        Azimuthal angle [rad].
        
    Returns
    -------
    Z : np.ndarray
        Zernike polynomial values.
    """
    # Handle the sign of m
    if m >= 0:
        angular = np.cos(m * theta)
    else:
        m = abs(m)
        angular = np.sin(m * theta)
    
    # Compute radial polynomial
    radial = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coefficient = ((-1)**k * factorial(n - k) /
                      (factorial(k) * 
                       factorial((n + m) // 2 - k) * 
                       factorial((n - m) // 2 - k)))
        radial += coefficient * rho**(n - 2*k)
    
    return radial * angular


def noll_to_nm(noll_index):
    """
    Convert Noll index to (n, m) indices.
    
    Parameters
    ----------
    noll_index : int
        Zernike index in Noll notation (1-based).
        
    Returns
    -------
    n : int
        Radial order.
    m : int
        Azimuthal order.
    """
    n = 0
    m = 0
    noll = 1
    
    while noll < noll_index:
        m += 1
        if m > n:
            n += 1
            m = -n
        else:
            m = -m
        noll += 1
    
    return n, m


def zernike_name(noll_index):
    """
    Get the name of a Zernike polynomial by Noll index.
    
    Parameters
    ----------
    noll_index : int
        Zernike index in Noll notation.
        
    Returns
    -------
    name : str
        Human-readable name.
    """
    names = {
        1: "Piston",
        2: "Tip (X)",
        3: "Tip (Y)",
        4: "Defocus",
        5: "Astigmatism 45°",
        6: "Astigmatism 0°",
        7: "Coma X",
        8: "Coma Y",
        9: "Trefoil X",
        10: "Trefoil Y",
        11: "Spherical",
        12: "Trefoil 45°",
        13: "Trefoil 0°",
        14: "Secondary Astigmatism X",
        15: "Secondary Astigmatism Y",
        16: "Quadrafoil X",
        17: "Quadrafoil Y",
        18: "Secondary Coma X",
        19: "Secondary Coma Y",
        20: "Secondary Spherical",
        21: "Secondary Trefoil X",
        22: "Secondary Trefoil Y",
    }
    return names.get(noll_index, f"Z{noll_index}")


def generate_zernike_map(noll_index, x, y):
    """
    Generate a Zernike polynomial map over a grid.
    
    Parameters
    ----------
    noll_index : int
        Zernike index in Noll notation (1-based).
    x, y : np.ndarray
        Cartesian coordinates [m].
        
    Returns
    -------
    Z : np.ndarray
        Zernike polynomial values.
    """
    n, m = noll_to_nm(noll_index)
    
    # Convert to polar coordinates
    rho = normalize_rho(x, y)
    theta = np.arctan2(y, x)
    
    # Compute polynomial
    return zernike_polynomial(n, m, rho, theta)


def zernike_basis(n_max, x, y):
    """
    Generate multiple Zernike basis functions.
    
    Parameters
    ----------
    n_max : int
        Maximum radial order.
    x, y : np.ndarray
        Cartesian coordinates [m].
        
    Returns
    -------
    basis : dict
        Dictionary mapping Noll index to Zernike map.
    """
    basis = {}
    noll = 1
    n = 0
    m = -n
    
    while n <= n_max:
        n, m = noll_to_nm(noll)
        if n > n_max:
            break
        basis[noll] = generate_zernike_map(noll, x, y)
        noll += 1
    
    return basis


def slm_phase_from_zernike(coefficients, x, y):
    """
    Generate SLM phase from Zernike coefficients.
    
    The SLM applies a phase-only modulation, so we take the 
    residual phase after removing 2π multiples.
    
    Parameters
    ----------
    coefficients : dict
        Dictionary mapping Noll index to coefficient (in waves).
    x, y : np.ndarray
        Cartesian coordinates [m].
        
    Returns
    -------
    phase : np.ndarray
        Phase to apply to SLM [rad], wrapped to [0, 2π].
    """
    # Generate all needed Zernike maps
    zernike_sum = np.zeros_like(x, dtype=np.float64)
    
    for noll_index, coeff in coefficients.items():
        if abs(coeff) > 1e-10:  # Skip near-zero coefficients
            Z = generate_zernike_map(noll_index, x, y)
            zernike_sum += coeff * Z
    
    # Convert waves to radians and wrap to [0, 2π]
    phase = 2 * np.pi * zernike_sum
    phase = np.mod(phase, 2 * np.pi)  # Wrap to [0, 2π]
    
    return phase


def fit_zernike_to_phase(phase, x, y, n_max=10):
    """
    Fit Zernike coefficients to a measured phase.
    
    Parameters
    ----------
    phase : np.ndarray
        Measured phase [rad].
    x, y : np.ndarray
        Cartesian coordinates [m].
    n_max : int
        Maximum radial order to fit.
        
    Returns
    -------
    coefficients : dict
        Fitted Zernike coefficients in waves.
    """
    from numpy.linalg import lstsq
    
    # Create mask for unit circle
    rho = normalize_rho(x, y)
    mask = rho <= 1.0
    
    # Build design matrix
    noll_count = 0
    for noll in range(1, 100):
        n, m = noll_to_nm(noll)
        if n > n_max:
            break
        noll_count += 1
    
    # Compute basis functions
    basis_funcs = []
    for noll in range(1, noll_count + 1):
        Z = generate_zernike_map(noll, x, y)
        Z_flat = Z[mask].flatten()
        basis_funcs.append(Z_flat)
    
    # Stack into matrix
    A = np.column_stack(basis_funcs)
    
    # Fit (convert phase from radians to waves)
    phase_flat = (phase[mask].flatten() / (2 * np.pi))
    
    # Solve least squares
    coeffs, _, _, _ = lstsq(A, phase_flat, rcond=None)
    
    # Convert to dictionary
    coefficients = {i+1: coeffs[i] for i in range(len(coeffs))}
    
    return coefficients


def compensate_turbulence_phase(turbulence_phase, x, y, n_max=10):
    """
    Generate SLM phase to compensate for turbulence.
    
    This applies a conjugate phase correction.
    
    Parameters
    ----------
    turbulence_phase : np.ndarray
        Turbulence phase screen [rad].
    x, y : np.ndarray
        Cartesian coordinates [m].
    n_max : int
        Maximum Zernike order for correction.
        
    Returns
    -------
    correction_phase : np.ndarray
        SLM phase to compensate [rad].
    """
    # Fit Zernike coefficients to turbulence phase
    coefficients = fit_zernike_to_phase(turbulence_phase, x, y, n_max)
    
    # Generate correction phase
    correction = slm_phase_from_zernike(coefficients, x, y)
    
    return correction


def apply_slm_correction(E, correction_phase):
    """
    Apply SLM phase correction to a field.
    
    Parameters
    ----------
    E : np.ndarray
        Complex input field.
    correction_phase : np.ndarray
        SLM phase to apply [rad].
        
    Returns
    -------
    E_corrected : np.ndarray
        Field after SLM correction.
    """
    return E * np.exp(1j * correction_phase)


def simulate_slm(E, cfg, turbulence_phase=None, zernike_coeffs=None):
    """
    Simulate SLM phase correction with optional turbulence.
    
    Parameters
    ----------
    E : np.ndarray
        Input complex field.
    cfg : Config
        Configuration object.
    turbulence_phase : np.ndarray, optional
        Turbulence phase screen [rad].
    zernike_coeffs : dict, optional
        Zernike coefficients for SLM. If None, auto-fit from turbulence.
        
    Returns
    -------
    E_corrected : np.ndarray
        Corrected field.
    X, Y : np.ndarray
        Grid coordinates for reference.
    """
    X, Y = cfg.grid()
    
    # Determine correction phase
    if zernike_coeffs is not None:
        correction_phase = slm_phase_from_zernike(zernike_coeffs, X, Y)
    elif turbulence_phase is not None:
        correction_phase = compensate_turbulence_phase(turbulence_phase, X, Y)
    else:
        # No correction
        return E, X, Y
    
    # Apply correction
    E_corrected = apply_slm_correction(E, correction_phase)
    
    return E_corrected, X, Y


def adaptive_optic_correction(E, cfg, Cn2_estimate, n_modes=10):
    """
    Simulate adaptive optics correction with estimated turbulence.
    
    Parameters
    ----------
    E : np.ndarray
        Input field (distorted by turbulence).
    cfg : Config
        Configuration object.
    Cn2_estimate : float
        Estimated Cn2 value.
    n_modes : int
        Number of Zernike modes to correct.
        
    Returns
    -------
    E_corrected : np.ndarray
        Corrected field.
    """
    X, Y = cfg.grid()
    
    # Generate wavefront sensor measurement (simulated)
    # In practice, would use Shack-Hartmann or interferometer
    phase_measured = np.angle(E)
    
    # Fit to Zernikes
    coeffs = fit_zernike_to_phase(phase_measured, X, Y, n_max=n_modes)
    
    # Generate correction
    correction_phase = slm_phase_from_zernike(coeffs, X, Y)
    
    # Apply correction
    E_corrected = apply_slm_correction(E, correction_phase)
    
    return E_corrected
