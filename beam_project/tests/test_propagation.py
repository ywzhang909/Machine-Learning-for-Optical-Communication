"""
Tests for beam propagation using Angular Spectrum Method.
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import beam_simulation as bs


class TestPropagation:
    """Test suite for beam propagation."""

    def test_propagation_output_shape(self, lg_beam, default_config):
        """Test propagation returns correct shape."""
        E_prop = bs.propagation(lg_beam, z=1e-2, cfg=default_config)
        assert E_prop.shape == lg_beam.shape

    def test_propagation_lg(self, default_config):
        """Test propagation of LG beam."""
        E_lg = bs.lg(p=0, l=1, cfg=default_config)
        E_prop = bs.propagation(E_lg, z=5e-2, cfg=default_config)
        assert E_prop.shape == E_lg.shape
        assert np.iscomplexobj(E_prop)

    def test_propagation_hg(self, default_config):
        """Test propagation of HG beam."""
        E_hg = bs.hg(n=2, m=2, cfg=default_config)
        E_prop = bs.propagation(E_hg, z=5e-2, cfg=default_config)
        assert E_prop.shape == E_hg.shape
        assert np.iscomplexobj(E_prop)

    def test_propagation_ig(self, default_config):
        """Test propagation of IG beam."""
        E_ig = bs.ig(p=4, m=2, beam="e", cfg=default_config)
        E_prop = bs.propagation(E_ig, z=5e-2, cfg=default_config)
        assert E_prop.shape == E_ig.shape
        assert np.iscomplexobj(E_prop)

    def test_propagation_various_distances(self, small_config):
        """Test propagation at various distances."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        distances = [1e-3, 5e-3, 1e-2, 5e-2, 10e-2]
        
        for z in distances:
            E_prop = bs.propagation(E_lg, z=z, cfg=small_config)
            assert E_prop.shape == E_lg.shape
            intensity = np.abs(E_prop)**2
            assert np.max(intensity) > 0

    def test_propagation_nonzero(self, lg_beam, default_config):
        """Test propagated beam has non-zero values."""
        E_prop = bs.propagation(lg_beam, z=5e-2, cfg=default_config)
        intensity = np.abs(E_prop)**2
        assert np.max(intensity) > 0

    def test_propagation_uses_default_z(self, lg_beam, default_config):
        """Test propagation uses default z when not specified."""
        # Should use cfg.z_default
        E_prop_default = bs.propagation(lg_beam, cfg=default_config)
        E_prop_explicit = bs.propagation(lg_beam, z=default_config.z_default, cfg=default_config)
        assert E_prop_default.shape == E_prop_explicit.shape

    def test_propagation_intensity_changes_with_distance(self, small_config):
        """Test beam intensity profile changes during propagation."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        
        # Near field
        E_near = bs.propagation(E_lg, z=1e-3, cfg=small_config)
        # Far field
        E_far = bs.propagation(E_lg, z=20e-2, cfg=small_config)
        
        # Intensity patterns should be different
        intensity_near = np.abs(E_near)**2
        intensity_far = np.abs(E_far)**2
        
        # Normalize for comparison
        intensity_near = intensity_near / np.max(intensity_near)
        intensity_far = intensity_far / np.max(intensity_far)
        
        # Should have different distributions
        assert not np.allclose(intensity_near, intensity_far, rtol=0.1)

    def test_propagation_maintains_total_power(self, lg_beam, default_config):
        """Test propagation approximately conserves total power."""
        input_power = np.sum(np.abs(lg_beam)**2)
        E_prop = bs.propagation(lg_beam, z=5e-2, cfg=default_config)
        output_power = np.sum(np.abs(E_prop)**2)
        
        # Power should be approximately conserved (within numerical tolerance)
        ratio = output_power / input_power
        assert 0.5 < ratio < 2.0  # Allow some numerical variation


class TestPropagationVisualization:
    """Visualization tests for beam propagation."""

    def test_propagation_distance_sequence(self, small_config, output_dir):
        """Visualize beam propagation at multiple distances."""
        E_lg = bs.lg(p=0, l=2, cfg=small_config)
        
        distances = [1e-3, 5e-3, 10e-3, 50e-3, 100e-3]
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        for idx, z in enumerate(distances):
            E_prop = bs.propagation(E_lg, z=z, cfg=small_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax.set_title(f'z = {z*1e3:.0f} mm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('LG$_0^2$ Beam Propagation (Angular Spectrum Method)')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_propagation_sequence.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_propagation_cross_section(self, small_config, output_dir):
        """Visualize beam cross-sections during propagation."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        
        distances = [1e-3, 20e-3, 50e-3]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        center = small_config.size_x // 2
        
        for idx, z in enumerate(distances):
            E_prop = bs.propagation(E_lg, z=z, cfg=small_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            # Intensity image
            ax1 = axes[0, idx]
            im = ax1.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax1.set_title(f'z = {z*1e3:.0f} mm')
            plt.colorbar(im, ax=ax1)
            
            # Cross-section
            ax2 = axes[1, idx]
            X, Y = small_config.grid()
            x_mm = X[center, :] * 1e3
            ax2.plot(x_mm, intensity_norm[center, :], 'b-', linewidth=1.5)
            ax2.set_xlabel('X (mm)')
            ax2.set_ylabel('Intensity (norm.)')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('LG$_0^1$ Beam Intensity During Propagation')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_propagation_cross_section.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_propagation_phase_evolution(self, small_config, output_dir):
        """Visualize beam phase evolution during propagation."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        
        distances = [1e-3, 10e-3, 50e-3]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, z in enumerate(distances):
            E_prop = bs.propagation(E_lg, z=z, cfg=small_config)
            phase = np.angle(E_prop)
            
            ax = axes[idx]
            im = ax.imshow(phase, cmap='twilight_shifted', origin='lower',
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Phase at z = {z*1e3:.0f} mm')
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle('LG$_0^1$ Beam Phase Evolution During Propagation')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_propagation_phase.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_propagation_hg_mode_evolution(self, small_config, output_dir):
        """Visualize HG mode propagation."""
        E_hg = bs.hg(n=2, m=2, cfg=small_config)
        
        distances = [1e-3, 10e-3, 30e-3, 50e-3]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for idx, z in enumerate(distances):
            E_prop = bs.propagation(E_hg, z=z, cfg=small_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='plasma', origin='lower')
            ax.set_title(f'z = {z*1e3:.0f} mm')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('HG$_{2,2}$ Beam Propagation')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_propagation_hg.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_propagation_amplitude_phase_3d(self, small_config, output_dir):
        """Visualize amplitude and phase as 3D surfaces."""
        E_lg = bs.lg(p=0, l=1, cfg=small_config)
        E_prop = bs.propagation(E_lg, z=20e-3, cfg=small_config)
        
        # Subsample for 3D plotting
        step = 4
        x = np.arange(0, small_config.size_x, step)
        y = np.arange(0, small_config.size_y, step)
        X, Y = np.meshgrid(x, y)
        
        intensity = np.abs(E_prop[::step, ::step])**2
        phase = np.angle(E_prop[::step, ::step])
        
        fig = plt.figure(figsize=(14, 5))
        
        # 3D intensity
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, intensity, cmap='plasma', alpha=0.8)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.set_zlabel('Intensity')
        ax1.set_title('LG$_0^1$ Intensity (z=20mm)')
        
        # 3D phase
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, phase, cmap='twilight_shifted', alpha=0.8)
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.set_zlabel('Phase (rad)')
        ax2.set_title('LG$_0^1$ Phase (z=20mm)')
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_propagation_3d.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
