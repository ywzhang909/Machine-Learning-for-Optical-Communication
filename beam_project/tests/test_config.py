"""
Tests for beam_simulation Config class.
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import beam_simulation as bs


class TestConfig:
    """Test suite for Config class."""

    def test_config_creation(self, default_config):
        """Test that Config can be created with default parameters."""
        assert default_config is not None
        assert default_config.wavelength == 810e-9
        assert default_config.size_x == 256
        assert default_config.size_y == 256
        assert default_config.pixel_size == 8e-6
        assert default_config.w0 == 0.45e-3

    def test_config_custom_parameters(self):
        """Test Config with custom parameters."""
        cfg = bs.Config(
            wavelength=633e-9,
            size_x=512,
            size_y=512,
            pixel_size=5e-6,
            w0=1e-3
        )
        assert cfg.wavelength == 633e-9
        assert cfg.size_x == 512
        assert cfg.size_y == 512
        assert cfg.pixel_size == 5e-6
        assert cfg.w0 == 1e-3

    def test_config_grid_shape(self, default_config):
        """Test that grid returns correct shape."""
        X, Y = default_config.grid()
        assert X.shape == (256, 256)
        assert Y.shape == (256, 256)
        assert X.shape == Y.shape

    def test_config_grid_symmetry(self, default_config):
        """Test that grid is centered at origin."""
        X, Y = default_config.grid()
        # Center pixel should be at origin
        center = default_config.size_x // 2
        assert np.isclose(X[center, center], 0.0)
        assert np.isclose(Y[center, center], 0.0)

    def test_config_grid_spacing(self, small_config):
        """Test grid spacing matches pixel size."""
        X, Y = small_config.grid()
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        assert np.isclose(dx, small_config.pixel_size)
        assert np.isclose(dy, small_config.pixel_size)

    def test_config_grid_extent(self, small_config):
        """Test grid extent covers expected physical size."""
        X, Y = small_config.grid()
        extent_x = X.max() - X.min()
        extent_y = Y.max() - Y.min()
        expected_x = (small_config.size_x - 1) * small_config.pixel_size
        expected_y = (small_config.size_y - 1) * small_config.pixel_size
        assert np.isclose(extent_x, expected_x)
        assert np.isclose(extent_y, expected_y)

    def test_config_repr(self, default_config):
        """Test Config string representation."""
        repr_str = repr(default_config)
        assert "Beam Simulation Configuration" in repr_str
        assert "Wavelength" in repr_str
        assert "nm" in repr_str

    def test_config_as_default(self, small_config):
        """Test that creating Config sets it as default."""
        cfg = bs.Config(wavelength=500e-9, size_x=100, size_y=100)
        default = bs.get_default_config()
        assert default.wavelength == 500e-9
        assert default.size_x == 100


class TestConfigVisualization:
    """Visualization tests for Config grid."""

    def test_grid_visualization(self, small_config, output_dir):
        """Visualize the grid X and Y coordinates."""
        X, Y = small_config.grid()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # X coordinate
        im1 = axes[0].imshow(X * 1e3, cmap='RdBu')
        axes[0].set_title('X Coordinate (mm)')
        axes[0].set_xlabel('Pixel X')
        axes[0].set_ylabel('Pixel Y')
        plt.colorbar(im1, ax=axes[0], label='X (mm)')
        
        # Y coordinate
        im2 = axes[1].imshow(Y * 1e3, cmap='RdBu')
        axes[1].set_title('Y Coordinate (mm)')
        axes[1].set_xlabel('Pixel X')
        axes[1].set_ylabel('Pixel Y')
        plt.colorbar(im2, ax=axes[1], label='Y (mm)')
        
        # Radial distance
        R = np.sqrt(X**2 + Y**2)
        im3 = axes[2].imshow(R * 1e3, cmap='viridis')
        axes[2].set_title('Radial Distance (mm)')
        axes[2].set_xlabel('Pixel X')
        axes[2].set_ylabel('Pixel Y')
        plt.colorbar(im3, ax=axes[2], label='R (mm)')
        
        plt.suptitle(f'Config Grid: {small_config.size_x}x{small_config.size_y}, '
                     f'pixel={small_config.pixel_size*1e6:.0f}µm')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_config_grid.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath), f"Visualization not saved: {filepath}"

    def test_grid_cross_section(self, default_config, output_dir):
        """Visualize cross-sections of the grid."""
        X, Y = default_config.grid()
        center = default_config.size_x // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # X cross-section
        axes[0].plot(X[center, :] * 1e3, 'b-', linewidth=1.5)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Origin')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('X coordinate (mm)')
        axes[0].set_title('X Cross-section at Center')
        axes[0].grid(True, alpha=0.3)
        
        # Radial profile
        R = np.sqrt(X**2 + Y**2)
        r_flat = R[center, :]
        intensity = np.exp(-2 * (r_flat / default_config.w0)**2)
        axes[1].plot(r_flat * 1e3, intensity, 'b-', linewidth=1.5, label='Gaussian envelope')
        axes[1].axvline(x=default_config.w0 * 1e3, color='r', linestyle='--', 
                        alpha=0.5, label=f'w0={default_config.w0*1e3:.3f}mm')
        axes[1].set_xlabel('r (mm)')
        axes[1].set_ylabel('Normalized Intensity')
        axes[1].set_title('Gaussian Envelope Profile')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Config Grid Analysis')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_config_cross_section.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath), f"Visualization not saved: {filepath}"
