"""
Tests for turbulence phase screen generation.
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import beam_simulation as bs


class TestTurbulence:
    """Test suite for turbulence generation."""

    def test_turbulence_basic_generation(self, default_config):
        """Test turbulence can be generated with default parameters."""
        phi = bs.turbulence(cfg=default_config)
        assert phi is not None
        assert isinstance(phi, np.ndarray)

    def test_turbulence_shape_matches_config(self, small_config):
        """Test turbulence output shape matches config grid size."""
        phi = bs.turbulence(cfg=small_config)
        assert phi.shape == (small_config.size_y, small_config.size_x)

    def test_turbulence_is_real(self, default_config):
        """Test turbulence phase screen is real-valued."""
        phi = bs.turbulence(cfg=default_config)
        assert np.isrealobj(phi) or np.allclose(np.imag(phi), 0)

    def test_turbulence_has_variation(self, default_config):
        """Test turbulence has spatial variation."""
        phi = bs.turbulence(cfg=default_config)
        # Should have non-zero variance
        assert np.var(phi) > 0
        # Should span positive and negative values (approximately centered at 0)
        assert phi.min() < 0 or phi.max() > 0

    def test_turbulence_various_Cn2(self, small_config):
        """Test turbulence generation with various Cn2 values."""
        Cn2_values = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11]
        
        for Cn2 in Cn2_values:
            phi = bs.turbulence(cfg=small_config, Cn2=Cn2)
            assert phi.shape == (small_config.size_y, small_config.size_x)
            assert np.var(phi) > 0

    def test_turbulence_stronger_Cn2_larger_variance(self, small_config):
        """Test stronger turbulence (higher Cn2) has larger phase variance."""
        phi_weak = bs.turbulence(cfg=small_config, Cn2=1e-15)
        phi_strong = bs.turbulence(cfg=small_config, Cn2=1e-12)
        
        var_weak = np.var(phi_weak)
        var_strong = np.var(phi_strong)
        
        # Stronger turbulence should generally have larger variance
        assert var_strong > var_weak

    def test_turbulence_is_random(self, default_config):
        """Test turbulence phase screens are different (random)."""
        phi1 = bs.turbulence(cfg=default_config)
        phi2 = bs.turbulence(cfg=default_config)
        
        # Should be different realizations
        assert not np.allclose(phi1, phi2)

    def test_turbulence_reproducible_with_seed(self, small_config):
        """Test turbulence is reproducible when using numpy seed."""
        # Note: The turbulence function doesn't have a seed parameter
        # This test just verifies the behavior
        np.random.seed(42)
        phi1 = bs.turbulence(cfg=small_config, Cn2=1e-13)
        
        np.random.seed(42)
        phi2 = bs.turbulence(cfg=small_config, Cn2=1e-13)
        
        # Should be reproducible with same random state
        assert np.allclose(phi1, phi2)

    def test_turbulence_various_scales(self, small_config):
        """Test turbulence with various inner/outer scale parameters."""
        l_min_values = [1e-4, 1e-3, 1e-2]
        l_max_values = [0.1, 1.0, 10.0]
        
        for l_min in l_min_values[:2]:  # Only test a few combinations
            for l_max in l_max_values[:2]:
                if l_max > l_min:  # Ensure l_max > l_min
                    phi = bs.turbulence(cfg=small_config, l_min=l_min, l_max=l_max)
                    assert phi.shape == (small_config.size_y, small_config.size_x)

    def test_turbulence_applied_to_beam(self, lg_beam, default_config):
        """Test turbulence can be applied to a beam."""
        phi_turb = bs.turbulence(cfg=default_config, Cn2=38e-12)
        E_turb = lg_beam * np.exp(1j * phi_turb)
        
        assert E_turb.shape == lg_beam.shape
        assert np.iscomplexobj(E_turb)

    def test_turbulence_center_values(self, default_config):
        """Test turbulence phase at center is well-defined."""
        phi = bs.turbulence(cfg=default_config)
        center = default_config.size_x // 2
        # Center should have a finite value
        assert np.isfinite(phi[center, center])


class TestTurbulenceVisualization:
    """Visualization tests for turbulence."""

    def test_turbulence_single_realization(self, small_config, output_dir):
        """Visualize a single turbulence phase screen."""
        phi = bs.turbulence(cfg=small_config, Cn2=38e-12)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Phase screen
        ax = axes[0]
        im = ax.imshow(phi, cmap='RdBu', origin='lower')
        ax.set_title('Turbulence Phase Screen')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        # Histogram
        ax = axes[1]
        ax.hist(phi.flatten(), bins=50, density=True, alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='r', linestyle='--', label='Mean ≈ 0')
        ax.set_xlabel('Phase (rad)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Phase Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Turbulence Phase Screen (Cn2=38×10⁻¹², '
                     f'{small_config.size_x}×{small_config.size_y})')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_turbulence_single.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_turbulence_multiple_Cn2(self, small_config, output_dir):
        """Visualize turbulence at various strength levels."""
        Cn2_values = [1e-15, 1e-14, 1e-13, 5e-13, 1e-12]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, Cn2 in enumerate(Cn2_values):
            phi = bs.turbulence(cfg=small_config, Cn2=Cn2)
            
            ax = axes[idx]
            im = ax.imshow(phi, cmap='RdBu', origin='lower',
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Cn² = {Cn2:.0e} m⁻²/³')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        # Remove extra subplot
        axes[-1].axis('off')
        
        plt.suptitle('Turbulence Phase Screens at Various Strengths')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_turbulence_Cn2_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_turbulence_applied_to_lg_beam(self, small_config, output_dir):
        """Visualize turbulence effect on LG beam."""
        E_lg = bs.lg(p=0, l=2, cfg=small_config)
        
        # Clean beam
        intensity_clean = np.abs(E_lg)**2
        
        # Turbulence strengths
        Cn2_values = [1e-14, 5e-14, 1e-13, 5e-13]
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        
        # Clean beam
        ax = axes[0, 0]
        im = ax.imshow(intensity_clean, cmap='plasma', origin='lower')
        ax.set_title('Clean Beam')
        plt.colorbar(im, ax=ax)
        
        # Apply turbulence at different levels
        for idx, Cn2 in enumerate(Cn2_values):
            phi_turb = bs.turbulence(cfg=small_config, Cn2=Cn2)
            E_turb = E_lg * np.exp(1j * phi_turb)
            intensity_turb = np.abs(E_turb)**2
            
            # Intensity
            ax = axes[0, idx + 1]
            im = ax.imshow(intensity_turb, cmap='plasma', origin='lower')
            ax.set_title(f'Cn² = {Cn2:.0e}')
            plt.colorbar(im, ax=ax)
            
            # Cross-section
            ax = axes[1, idx + 1]
            center = small_config.size_x // 2
            x = np.arange(small_config.size_x) - center
            ax.plot(x, intensity_clean[center, :] / np.max(intensity_clean), 
                   'b-', alpha=0.5, label='Clean', linewidth=1)
            ax.plot(x, intensity_turb[center, :] / np.max(intensity_turb), 
                   'r-', alpha=0.8, label='Turbulent', linewidth=1)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Intensity (norm.)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        axes[1, 0].axis('off')
        
        plt.suptitle('Turbulence Effect on LG$_0^2$ Beam')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_turbulence_lg_effect.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_turbulence_propagation_effect(self, small_config, output_dir):
        """Visualize beam propagation with turbulence."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        
        phi_turb = bs.turbulence(cfg=small_config, Cn2=38e-12)
        E_turb = E_lg * np.exp(1j * phi_turb)
        
        distances = [1e-3, 10e-3, 30e-3, 50e-3]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for idx, z in enumerate(distances):
            # Propagate clean beam
            E_clean_prop = bs.propagation(E_lg, z=z, cfg=small_config)
            intensity_clean = np.abs(E_clean_prop)**2
            
            # Propagate turbulent beam
            E_turb_prop = bs.propagation(E_turb, z=z, cfg=small_config)
            intensity_turb = np.abs(E_turb_prop)**2
            
            # Clean
            ax = axes[0, idx]
            im = ax.imshow(intensity_clean, cmap='plasma', origin='lower')
            ax.set_title(f'Clean z={z*1e3:.0f}mm')
            plt.colorbar(im, ax=ax)
            
            # Turbulent
            ax = axes[1, idx]
            im = ax.imshow(intensity_turb, cmap='plasma', origin='lower')
            ax.set_title(f'Turbulent z={z*1e3:.0f}mm')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Beam Propagation: Clean vs Turbulent')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_turbulence_propagation.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_turbulence_spectrum_analysis(self, small_config, output_dir):
        """Visualize turbulence phase power spectrum."""
        phi = bs.turbulence(cfg=small_config, Cn2=38e-12)
        
        # Compute 2D FFT to get spatial frequency spectrum
        spectrum = np.fft.fftshift(np.fft.fft2(phi))
        power_spectrum = np.abs(spectrum)**2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Phase screen
        ax = axes[0]
        im = ax.imshow(phi, cmap='RdBu', origin='lower')
        ax.set_title('Phase Screen')
        plt.colorbar(im, ax=ax)
        
        # Power spectrum (log scale)
        ax = axes[1]
        im = ax.imshow(np.log10(power_spectrum + 1e-10), cmap='viridis', origin='lower')
        ax.set_title('Power Spectrum (log)')
        plt.colorbar(im, ax=ax)
        
        # Radial average of spectrum
        ax = axes[2]
        freq = np.fft.fftfreq(small_config.size_x, small_config.pixel_size)
        freq = np.fft.fftshift(freq)
        
        # Compute radial profile
        FX, FY = np.meshgrid(freq, freq)
        R = np.sqrt(FX**2 + FY**2)
        
        # Bin the spectrum radially
        max_freq = np.max(freq)
        n_bins = 50
        bins = np.linspace(0, max_freq, n_bins)
        radial_profile = np.zeros(n_bins - 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        for i in range(n_bins - 1):
            mask = (R >= bins[i]) & (R < bins[i + 1])
            radial_profile[i] = np.mean(power_spectrum[mask])
        
        ax.loglog(bin_centers, radial_profile, 'b-', linewidth=1.5)
        ax.set_xlabel('Spatial Frequency (1/m)')
        ax.set_ylabel('Power')
        ax.set_title('Radial Power Spectrum')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Turbulence Phase Statistics')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_turbulence_spectrum.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
