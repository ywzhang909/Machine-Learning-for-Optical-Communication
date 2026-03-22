"""
High-resolution tests for Gaussian beam atmospheric propagation.
重点测试高斯光的大气传播
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import beam_simulation as bs


class TestGaussianAtmosphericPropagation:
    """高斯光大气传播测试套件"""

    def test_gaussian_basic_propagation(self, high_res_config):
        """Test Gaussian beam propagation."""
        E = bs.gauss(cfg=high_res_config)
        E_prop = bs.propagation(E, z=10e-2, cfg=high_res_config)
        assert E_prop.shape == E.shape
        assert np.max(np.abs(E_prop)**2) > 0

    def test_gaussian_with_turbulence(self, high_res_config):
        """Test Gaussian beam with turbulence applied."""
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        E_prop = bs.propagation(E_turb, z=20e-2, cfg=high_res_config)
        assert E_prop.shape == E.shape
        intensity = np.abs(E_prop)**2
        assert np.max(intensity) > 0

    def test_gaussian_various_turbulence_levels(self, high_res_config):
        """Test Gaussian beam with various turbulence levels."""
        Cn2_levels = [1e-15, 1e-14, 1e-13, 5e-13, 1e-12, 5e-12]
        for Cn2 in Cn2_levels:
            E = bs.gauss(cfg=high_res_config)
            phi = bs.turbulence(cfg=high_res_config, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi)
            E_prop = bs.propagation(E_turb, z=10e-2, cfg=high_res_config)
            assert E_prop.shape == E.shape

    def test_gaussian_various_distances(self, high_res_config):
        """Test Gaussian beam propagation at various distances."""
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        distances = [1e-3, 5e-3, 10e-2, 20e-2, 50e-2, 100e-2]
        for z in distances:
            E_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            assert E_prop.shape == E.shape


class TestGaussianVisualization:
    """高斯光传播可视化测试 (高分辨率)"""

    def test_gaussian_clean_propagation_sequence(self, high_res_config, output_dir):
        """Visualize Gaussian beam clean propagation at multiple distances.
        高斯光无湍流传播序列
        """
        E = bs.gauss(cfg=high_res_config)
        distances = [0, 5e-3, 10e-2, 20e-2, 50e-2, 100e-2]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, z in enumerate(distances):
            if z == 0:
                E_prop = E
            else:
                E_prop = bs.propagation(E, z=z, cfg=high_res_config)
            
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1])
            ax.set_title(f'z = {z*1e3:.0f} mm', fontsize=12)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Normalized Intensity')
        
        plt.suptitle('Gaussian Beam Clean Propagation\n(No Turbulence)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_clean_propagation.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_turbulent_propagation_sequence(self, high_res_config, output_dir):
        """Visualize Gaussian beam propagation with turbulence.
        高斯光湍流传播序列
        """
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        distances = [0, 5e-3, 10e-2, 20e-2, 50e-2, 100e-2]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, z in enumerate(distances):
            if z == 0:
                E_prop = E_turb
            else:
                E_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1])
            ax.set_title(f'z = {z*1e3:.0f} mm', fontsize=12)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Normalized Intensity')
        
        plt.suptitle(f'Gaussian Beam Turbulent Propagation\n(Cn² = 3.8×10⁻¹² m⁻²/³)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_turbulent_propagation.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_clean_vs_turbulent_comparison(self, high_res_config, output_dir):
        """Compare clean and turbulent propagation side by side.
        清洁与湍流传播对比
        """
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        distances = [10e-2, 30e-2, 50e-2]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for idx, z in enumerate(distances):
            # Clean
            E_clean_prop = bs.propagation(E, z=z, cfg=high_res_config)
            intensity_clean = np.abs(E_clean_prop)**2
            intensity_clean_norm = intensity_clean / np.max(intensity_clean)
            
            ax = axes[0, idx]
            im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'Clean Beam @ z = {z*1e3:.0f} mm', fontsize=12)
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Turbulent
            E_turb_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            intensity_turb = np.abs(E_turb_prop)**2
            intensity_turb_norm = intensity_turb / np.max(intensity_turb)
            
            ax = axes[1, idx]
            im = ax.imshow(intensity_turb_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'Turbulent Beam @ z = {z*1e3:.0f} mm', fontsize=12)
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle('Gaussian Beam: Clean vs Turbulent Propagation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_clean_vs_turbulent.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_various_turbulence_levels(self, high_res_config, output_dir):
        """Visualize Gaussian beam at various turbulence levels.
        不同湍流强度对比
        """
        E = bs.gauss(cfg=high_res_config)
        
        Cn2_levels = [1e-15, 1e-14, 1e-13, 5e-13, 1e-12, 38e-12]
        z = 30e-2  # 30 cm propagation
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, Cn2 in enumerate(Cn2_levels):
            phi = bs.turbulence(cfg=high_res_config, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi)
            E_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'Cn² = {Cn2:.1e} m⁻²/³', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Gaussian Beam @ z = {z*1e3:.0f} mm - Various Turbulence Levels',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_various_turbulence.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_cross_section_comparison(self, high_res_config, output_dir):
        """Visualize intensity cross-sections (clean vs turbulent).
        强度剖面对比
        """
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        z = 50e-2  # 50 cm
        
        E_clean = bs.propagation(E, z=z, cfg=high_res_config)
        E_turbulent = bs.propagation(E_turb, z=z, cfg=high_res_config)
        
        intensity_clean = np.abs(E_clean)**2 / np.max(np.abs(E_clean)**2)
        intensity_turb = np.abs(E_turbulent)**2 / np.max(np.abs(E_turbulent)**2)
        
        center = high_res_config.size_x // 2
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cross-section plot
        ax = axes[0]
        x = (np.arange(high_res_config.size_x) - center) * high_res_config.pixel_size * 1e3
        
        ax.plot(x, intensity_clean[center, :], 'b-', linewidth=2, label='Clean', alpha=0.8)
        ax.plot(x, intensity_turb[center, :], 'r-', linewidth=2, label='Turbulent', alpha=0.8)
        ax.fill_between(x, intensity_clean[center, :], alpha=0.3, color='blue')
        ax.fill_between(x, intensity_turb[center, :], alpha=0.3, color='red')
        
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)
        ax.set_title(f'Intensity Cross-Section @ z = {z*1e3:.0f} mm', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.1)
        
        # 2D comparison
        ax = axes[1]
        # Create a side-by-side comparison
        combined = np.hstack([intensity_clean, np.ones((high_res_config.size_x, 5)), intensity_turb])
        im = ax.imshow(combined, cmap='hot', origin='lower', aspect='auto',
                       extent=[-2.1, 2.1, -1, 1])
        ax.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=1.02, color='white', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Y Position (mm)', fontsize=12)
        ax.set_title('Clean | Turbulent', fontsize=14)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Gaussian Beam Comparison @ z = {z*1e3:.0f} mm', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_cross_section.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_long_range_propagation(self, high_res_config, output_dir):
        """Visualize long-range propagation through turbulence.
        长距离湍流传播
        """
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        distances = [10e-2, 50e-2, 100e-2, 200e-2, 500e-2]
        
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        
        for idx, z in enumerate(distances):
            E_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'z = {z*1e3:.0f} mm\n({z:.1f} m)', fontsize=11)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='I')
        
        plt.suptitle('Long-Range Gaussian Beam Propagation (Cn² = 3.8×10⁻¹²)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_long_range.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_spot_size_evolution(self, high_res_config, output_dir):
        """Analyze beam spot size evolution through turbulence.
        光斑尺寸演变分析
        """
        E = bs.gauss(cfg=high_res_config)
        phi = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        E_turb = E * np.exp(1j * phi)
        
        distances = np.linspace(0, 100e-2, 21)  # 0 to 1m
        spot_sizes_clean = []
        spot_sizes_turb = []
        
        center = high_res_config.size_x // 2
        
        for z in distances:
            # Clean
            E_clean = bs.propagation(E, z=z, cfg=high_res_config)
            intensity_clean = np.abs(E_clean)**2
            
            # RMS spot size (weighted by intensity)
            X, Y = high_res_config.grid()
            r_sq = X**2 + Y**2
            total_power = np.sum(intensity_clean)
            if total_power > 0:
                spot_size_clean = np.sqrt(np.sum(r_sq * intensity_clean) / total_power)
            else:
                spot_size_clean = 0
            spot_sizes_clean.append(spot_size_clean * 1e3)  # mm
            
            # Turbulent
            E_turb_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            intensity_turb = np.abs(E_turb_prop)**2
            
            total_power = np.sum(intensity_turb)
            if total_power > 0:
                spot_size_turb = np.sqrt(np.sum(r_sq * intensity_turb) / total_power)
            else:
                spot_size_turb = 0
            spot_sizes_turb.append(spot_size_turb * 1e3)  # mm
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(distances * 1e3, spot_sizes_clean, 'b-o', linewidth=2, 
                markersize=6, label='Clean Beam', alpha=0.8)
        ax.plot(distances * 1e3, spot_sizes_turb, 'r-s', linewidth=2, 
                markersize=6, label='Turbulent Beam', alpha=0.8)
        
        ax.set_xlabel('Propagation Distance (mm)', fontsize=12)
        ax.set_ylabel('RMS Spot Size (mm)', fontsize=12)
        ax.set_title('Beam Spot Size Evolution During Propagation', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Gaussian Beam Spot Size: Clean vs Turbulent (Cn² = 3.8×10⁻¹²)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_spot_size_evolution.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_peak_intensity_fluctuation(self, high_res_config, output_dir):
        """Analyze peak intensity fluctuations due to turbulence.
        峰值强度起伏分析
        """
        n_realizations = 50
        z = 50e-2  # 50 cm propagation
        Cn2 = 38e-12
        
        peak_intensities = []
        
        E = bs.gauss(cfg=high_res_config)
        
        for _ in range(n_realizations):
            phi = bs.turbulence(cfg=high_res_config, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi)
            E_prop = bs.propagation(E_turb, z=z, cfg=high_res_config)
            
            intensity = np.abs(E_prop)**2
            peak_intensities.append(np.max(intensity))
        
        peak_intensities = np.array(peak_intensities)
        peak_intensities_norm = peak_intensities / np.mean(peak_intensities)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Time series
        ax = axes[0]
        ax.plot(range(n_realizations), peak_intensities_norm, 'b-o', markersize=4)
        ax.axhline(y=1.0, color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Realization Index', fontsize=12)
        ax.set_ylabel('Normalized Peak Intensity', fontsize=12)
        ax.set_title('Peak Intensity Fluctuations', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histogram
        ax = axes[1]
        ax.hist(peak_intensities_norm, bins=20, density=True, alpha=0.7, 
               color='steelblue', edgecolor='black')
        ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(x=np.mean(peak_intensities_norm), color='orange', linestyle='-', 
                  linewidth=2, label=f'Actual Mean = {np.mean(peak_intensities_norm):.2f}')
        ax.set_xlabel('Normalized Peak Intensity', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title('Intensity Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Statistics box
        ax = axes[2]
        ax.axis('off')
        
        stats_text = f"""
        Turbulence Statistics
        
        Cn² = {Cn2:.1e} m⁻²/³
        Propagation Distance = {z*1e3:.0f} mm
        Number of Realizations = {n_realizations}
        
        Peak Intensity Statistics:
        ─────────────────────────
        Mean: {np.mean(peak_intensities_norm):.3f}
        Std Dev: {np.std(peak_intensities_norm):.3f}
        Min: {np.min(peak_intensities_norm):.3f}
        Max: {np.max(peak_intensities_norm):.3f}
        
        Scintillation Index:
        σ_I² = {np.var(peak_intensities_norm):.3f}
        """
        
        ax.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Gaussian Beam Scintillation Analysis',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'gaussian_scintillation.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_gaussian_detailed_comparison_grid(self, high_res_config, output_dir):
        """Create a detailed comparison grid of Gaussian beam propagation.
        高斯光传播详细对比网格
        """
        E = bs.gauss(cfg=high_res_config)
        phi_weak = bs.turbulence(cfg=high_res_config, Cn2=1e-14)
        phi_medium = bs.turbulence(cfg=high_res_config, Cn2=38e-13)
        phi_strong = bs.turbulence(cfg=high_res_config, Cn2=38e-12)
        
        E_turb_weak = E * np.exp(1j * phi_weak)
        E_turb_medium = E * np.exp(1j * phi_medium)
        E_turb_strong = E * np.exp(1j * phi_strong)
        
        z_levels = [10e-2, 50e-2, 100e-2]
        turbulence_labels = ['Weak\n(Cn²=10⁻¹⁴)', 'Medium\n(Cn²=3.8×10⁻¹³)', 
                           'Strong\n(Cn²=3.8×10⁻¹²)']
        
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Clean beam column
        for idx, z in enumerate(z_levels):
            E_clean = bs.propagation(E, z=z, cfg=high_res_config)
            intensity = np.abs(E_clean)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = fig.add_subplot(gs[idx, 0])
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'Clean @ {z*1e3:.0f}mm', fontsize=10)
            if idx == 0:
                ax.text(-1.5, 0, 'Clean', fontsize=12, fontweight='bold', 
                       rotation=90, va='center')
        
        # Weak turbulence
        for idx, z in enumerate(z_levels):
            E_prop = bs.propagation(E_turb_weak, z=z, cfg=high_res_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = fig.add_subplot(gs[idx, 1])
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'z={z*1e3:.0f}mm', fontsize=10)
            if idx == 0:
                ax.text(-1.5, 0, 'Weak', fontsize=12, fontweight='bold',
                       rotation=90, va='center')
        
        # Medium turbulence
        for idx, z in enumerate(z_levels):
            E_prop = bs.propagation(E_turb_medium, z=z, cfg=high_res_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = fig.add_subplot(gs[idx, 2])
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'z={z*1e3:.0f}mm', fontsize=10)
            if idx == 0:
                ax.text(-1.5, 0, 'Medium', fontsize=12, fontweight='bold',
                       rotation=90, va='center')
        
        # Strong turbulence
        for idx, z in enumerate(z_levels):
            E_prop = bs.propagation(E_turb_strong, z=z, cfg=high_res_config)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = fig.add_subplot(gs[idx, 3])
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-1, 1, -1, 1], vmin=0, vmax=1)
            ax.set_title(f'z={z*1e3:.0f}mm', fontsize=10)
            if idx == 0:
                ax.text(-1.5, 0, 'Strong', fontsize=12, fontweight='bold',
                       rotation=90, va='center')
        
        plt.suptitle('Gaussian Beam Atmospheric Propagation\nComprehensive Comparison',
                    fontsize=16, fontweight='bold')
        
        filepath = os.path.join(output_dir, 'gaussian_comprehensive_comparison.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
