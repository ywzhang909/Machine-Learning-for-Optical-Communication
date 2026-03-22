"""
Pytest configuration and shared fixtures for beam_simulation tests.
"""
import os
import sys
import pytest
import numpy as np

# Add beam_project to path for imports
beam_project_path = os.path.join(os.path.dirname(__file__), '..')
if beam_project_path not in sys.path:
    sys.path.insert(0, beam_project_path)

import beam_simulation as bs


@pytest.fixture(scope="session")
def output_dir():
    """Create and return the test output directory for visualizations."""
    test_output = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(test_output, exist_ok=True)
    return test_output


@pytest.fixture(scope="session")
def default_config():
    """Default configuration for beam simulation (moderate resolution)."""
    return bs.Config(
        wavelength=810e-9,  # 810 nm
        size_x=256,
        size_y=256,
        pixel_size=8e-6,    # 8 µm
        w0=0.45e-3,        # 0.45 mm beam waist
        z_default=1e-2,    # 1 cm
        Cn2=3e-13,         # Moderate turbulence
        l_max=1e-1,
        l_min=1e-3
    )


@pytest.fixture(scope="session")
def small_config():
    """Small configuration for fast tests."""
    return bs.Config(
        wavelength=810e-9,
        size_x=64,
        size_y=64,
        pixel_size=8e-6,
        w0=0.45e-3,
        z_default=1e-2,
        Cn2=3e-13,
        l_max=1e-1,
        l_min=1e-3
    )


@pytest.fixture(scope="session")
def high_res_config():
    """High resolution configuration for detailed visualization (512x512)."""
    return bs.Config(
        wavelength=810e-9,
        size_x=512,
        size_y=512,
        pixel_size=4e-6,    # 4 µm for finer detail
        w0=0.5e-3,         # 0.5 mm beam waist
        z_default=5e-2,
        Cn2=5e-14,
        l_max=0.5,
        l_min=1e-3
    )


@pytest.fixture
def lg_beam(default_config):
    """Generate a Laguerre-Gaussian beam for testing."""
    return bs.lg(p=0, l=1, cfg=default_config)


@pytest.fixture
def hg_beam(default_config):
    """Generate a Hermite-Gaussian beam for testing."""
    return bs.hg(n=2, m=2, cfg=default_config)


@pytest.fixture
def ig_beam(default_config):
    """Generate an Ince-Gaussian beam for testing."""
    return bs.ig(p=4, m=2, beam="e", cfg=default_config)


@pytest.fixture
def turbulence_screen(default_config):
    """Generate a turbulence phase screen."""
    return bs.turbulence(cfg=default_config, Cn2=38e-12)
