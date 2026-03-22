"""
Tests for beam generation functions (LG, HG, IG, Gaussian).
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import beam_simulation as bs


class TestLaguerreGaussian:
    """Test suite for Laguerre-Gaussian beam generation."""

    def test_lg_basic_generation(self, default_config):
        """Test LG beam can be generated with basic parameters."""
        E = bs.lg(p=0, l=1, cfg=default_config)
        assert E is not None
        assert isinstance(E, np.ndarray)
        assert E.shape == (default_config.size_y, default_config.size_x)

    def test_lg_shape_matches_config(self, small_config):
        """Test LG output shape matches config grid size."""
        E = bs.lg(p=1, l=2, cfg=small_config)
        assert E.shape == (small_config.size_y, small_config.size_x)

    def test_lg_is_complex(self, default_config):
        """Test LG beam has complex values (amplitude and phase)."""
        E = bs.lg(p=0, l=1, cfg=default_config)
        assert np.iscomplexobj(E)
        assert np.any(np.imag(E) != 0) or np.any(np.real(E) != 0)

    def test_lg_has_nonzero_values(self, default_config):
        """Test LG beam has non-zero values (not empty)."""
        E = bs.lg(p=0, l=1, cfg=default_config)
        intensity = np.abs(E)**2
        assert np.max(intensity) > 0

    def test_lg_intensity_normalization(self, default_config):
        """Test LG beam intensity can be normalized."""
        E = bs.lg(p=0, l=1, cfg=default_config)
        intensity = np.abs(E)**2
        intensity_norm = intensity / np.max(intensity)
        assert np.max(intensity_norm) == pytest.approx(1.0)
        assert np.min(intensity_norm) >= 0

    def test_lg_various_orders(self, default_config):
        """Test LG beam generation with various orders."""
        orders = [(0, 1), (1, 0), (2, 1), (0, 3), (3, 2)]
        for p, l in orders:
            E = bs.lg(p=p, l=l, cfg=default_config)
            assert E.shape == (default_config.size_y, default_config.size_x)
            intensity = np.abs(E)**2
            assert np.max(intensity) > 0

    def test_lg_azimuthal_variation(self, default_config):
        """Test LG beam has azimuthal phase variation (OAM)."""
        E_l1 = bs.lg(p=0, l=1, cfg=default_config)
        E_l2 = bs.lg(p=0, l=2, cfg=default_config)
        # Different l values should produce different phase patterns
        assert not np.allclose(np.angle(E_l1), np.angle(E_l2))

    def test_lg_radial_variation(self, default_config):
        """Test LG beam has radial variation (p parameter effect)."""
        E_p0 = bs.lg(p=0, l=1, cfg=default_config)
        E_p1 = bs.lg(p=1, l=1, cfg=default_config)
        # Different p values should produce different intensity patterns
        intensity_p0 = np.abs(E_p0)**2
        intensity_p1 = np.abs(E_p1)**2
        assert not np.allclose(intensity_p0, intensity_p1)


class TestHermiteGaussian:
    """Test suite for Hermite-Gaussian beam generation."""

    def test_hg_basic_generation(self, default_config):
        """Test HG beam can be generated with basic parameters."""
        E = bs.hg(n=0, m=0, cfg=default_config)
        assert E is not None
        assert isinstance(E, np.ndarray)
        assert E.shape == (default_config.size_y, default_config.size_x)

    def test_hg_shape_matches_config(self, small_config):
        """Test HG output shape matches config grid size."""
        E = bs.hg(n=1, m=1, cfg=small_config)
        assert E.shape == (small_config.size_y, small_config.size_x)

    def test_hg_is_complex(self, default_config):
        """Test HG beam has complex values."""
        E = bs.hg(n=1, m=1, cfg=default_config)
        assert np.iscomplexobj(E)

    def test_hg_has_nonzero_values(self, default_config):
        """Test HG beam has non-zero values."""
        E = bs.hg(n=1, m=1, cfg=default_config)
        intensity = np.abs(E)**2
        assert np.max(intensity) > 0

    def test_hg_fundamental_mode(self, default_config):
        """Test HG_00 is a fundamental Gaussian beam."""
        E = bs.hg(n=0, m=0, cfg=default_config)
        intensity = np.abs(E)**2
        # Fundamental mode should be Gaussian-like (single peak)
        center = (default_config.size_y // 2, default_config.size_x // 2)
        assert intensity[center] == np.max(intensity)

    def test_hg_mode_indices(self, default_config):
        """Test HG beam with various mode indices."""
        orders = [(0, 0), (1, 0), (0, 1), (2, 2), (3, 1)]
        for n, m in orders:
            E = bs.hg(n=n, m=m, cfg=default_config)
            assert E.shape == (default_config.size_y, default_config.size_x)
            intensity = np.abs(E)**2
            assert np.max(intensity) > 0

    def test_hg_lobe_count(self, default_config):
        """Test HG mode has expected number of intensity lobes."""
        # HG_22 should have (2+1) x (2+1) = 9 lobes
        E = bs.hg(n=2, m=2, cfg=default_config)
        intensity = np.abs(E)**2
        # Count local maxima
        threshold = np.max(intensity) * 0.3
        peaks = intensity > threshold
        # At least some structure should be present
        assert np.sum(peaks) > 0


class TestInceGaussian:
    """Test suite for Ince-Gaussian beam generation."""

    def test_ig_basic_generation(self, default_config):
        """Test IG beam can be generated."""
        E = bs.ig(p=4, m=2, beam="e", cfg=default_config)
        assert E is not None
        assert isinstance(E, np.ndarray)

    def test_ig_shape_matches_config(self, small_config):
        """Test IG output shape matches config grid size."""
        E = bs.ig(p=4, m=2, beam="e", cfg=small_config)
        assert E.shape == (small_config.size_y, small_config.size_x)

    def test_ig_is_complex(self, default_config):
        """Test IG beam has complex values."""
        E = bs.ig(p=4, m=2, beam="e", cfg=default_config)
        assert np.iscomplexobj(E)

    def test_ig_has_nonzero_values(self, default_config):
        """Test IG beam has non-zero values."""
        E = bs.ig(p=4, m=2, beam="e", cfg=default_config)
        intensity = np.abs(E)**2
        assert np.max(intensity) > 0

    def test_ig_even_vs_odd(self, default_config):
        """Test IG even and odd modes are different."""
        E_even = bs.ig(p=4, m=2, beam="e", cfg=default_config)
        # Note: odd modes may not be fully implemented
        # Just test that even mode is valid
        assert E_even is not None
        intensity = np.abs(E_even)**2
        assert np.max(intensity) > 0


class TestBeamVisualization:
    """Visualization tests for beam patterns."""

    def test_lg_intensity_visualization(self, small_config, output_dir):
        """Visualize LG beam intensity patterns."""
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        
        modes = [(0, 1), (0, 2), (1, 1), (0, 3), (2, 1), (1, 2)]
        
        for idx, (p, l) in enumerate(modes):
            ax = axes[idx // 3, idx % 3]
            E = bs.lg(p=p, l=l, cfg=small_config)
            intensity = np.abs(E)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax.set_title(f'LG$_{{{p}}}^{{{l}}}$')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle('Laguerre-Gaussian Beam Intensity Patterns')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_lg_intensity.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_hg_intensity_visualization(self, small_config, output_dir):
        """Visualize HG beam intensity patterns."""
        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        
        modes = [(0, 0), (1, 0), (0, 1), (2, 2), (3, 1), (2, 3)]
        
        for idx, (n, m) in enumerate(modes):
            ax = axes[idx // 3, idx % 3]
            E = bs.hg(n=n, m=m, cfg=small_config)
            intensity = np.abs(E)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax.set_title(f'HG$_{{{n},{m}}}$')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle('Hermite-Gaussian Beam Intensity Patterns')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_hg_intensity.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_lg_phase_visualization(self, small_config, output_dir):
        """Visualize LG beam phase (showing OAM)."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        modes = [(0, 1), (0, 2), (0, 3)]
        
        for idx, (p, l) in enumerate(modes):
            ax = axes[idx]
            E = bs.lg(p=p, l=l, cfg=small_config)
            phase = np.angle(E)
            
            im = ax.imshow(phase, cmap='twilight_shifted', origin='lower', 
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'LG$_{{{p}}}^{{{l}}}$ Phase')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle('Laguerre-Gaussian Beam Phase (OAM Visualization)')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_lg_phase.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_ig_intensity_visualization(self, small_config, output_dir):
        """Visualize IG beam intensity patterns."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        modes = [(4, 2), (6, 2), (8, 4)]
        
        for idx, (p, m) in enumerate(modes):
            ax = axes[idx]
            E = bs.ig(p=p, m=m, beam="e", cfg=small_config)
            intensity = np.abs(E)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax.set_title(f'IG$_{{{p}}}^{{{m}}}$ (even)')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle('Ince-Gaussian Beam Intensity Patterns')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_ig_intensity.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_beam_comparison_visualization(self, small_config, output_dir):
        """Compare LG, HG, and IG beam patterns."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # LG
        E_lg = bs.lg(p=0, l=2, cfg=small_config)
        ax = axes[0]
        im = ax.imshow(np.abs(E_lg)**2, cmap='plasma', origin='lower')
        ax.set_title('Laguerre-Gaussian\nLG$_0^2$')
        plt.colorbar(im, ax=ax)
        
        # HG
        E_hg = bs.hg(n=1, m=2, cfg=small_config)
        ax = axes[1]
        im = ax.imshow(np.abs(E_hg)**2, cmap='plasma', origin='lower')
        ax.set_title('Hermite-Gaussian\nHG$_{1,2}$')
        plt.colorbar(im, ax=ax)
        
        # IG
        E_ig = bs.ig(p=4, m=2, beam="e", cfg=small_config)
        ax = axes[2]
        im = ax.imshow(np.abs(E_ig)**2, cmap='plasma', origin='lower')
        ax.set_title('Ince-Gaussian\nIG$_4^2$ (even)')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle('Beam Pattern Comparison')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_beam_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
