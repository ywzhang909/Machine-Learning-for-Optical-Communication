"""
Focused Gaussian beam through turbulence - 聚焦高斯光大气传播
Tests focal spot images and phase distribution at 1000mm with Cn2 from 10^-18 to 10^-12
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import beam_simulation as bs


class TestFocusedBeamAtmosphere:
    """聚焦高斯光大气传播测试"""

    def test_focused_beam_propagation_to_1000mm(self):
        """Test focused beam propagation to 1000mm focal distance."""
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,  # 1m default
            Cn2=1e-14,
            l_max=0.5,
            l_min=1e-3
        )
        
        E = bs.gauss(cfg=cfg)
        
        # Focus at 1000mm
        z_focal = 1000e-3  # 1000mm = 1m
        E_focal = bs.propagation(E, z=z_focal, cfg=cfg)
        
        assert E_focal.shape == E.shape
        intensity = np.abs(E_focal)**2
        assert np.max(intensity) > 0

    def test_focused_beam_with_various_turbulence(self):
        """Test focused beam at 1000mm with various Cn2 values."""
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        z_focal = 1000e-3  # 1000mm
        
        for Cn2 in Cn2_levels:
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
            
            assert E_focal.shape == E.shape
            intensity = np.abs(E_focal)**2
            assert np.max(intensity) > 0


class TestFocusedBeamVisualization:
    """聚焦高斯光焦点分析可视化"""

    def test_focal_spot_various_Cn2(self, output_dir):
        """Visualize focal spot at 1000mm for Cn2 from 10^-18 to 10^-12.
        1000mm焦点处不同湍流强度的光斑图像
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3  # 1000mm
        Cn2_levels = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 5e-13, 1e-12]
        
        # Clean focal spot
        E = bs.gauss(cfg=cfg)
        E_clean_focal = bs.propagation(E, z=z_focal, cfg=cfg)
        intensity_clean = np.abs(E_clean_focal)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        fig, axes = plt.subplots(2, 5, figsize=(22, 10))
        
        # First: Clean focal spot
        ax = axes[0, 0]
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Clean Focus\n(no turbulence)', fontsize=11)
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Turbulent focal spots
        for idx, Cn2 in enumerate(Cn2_levels):
            ax = axes[(idx + 1) // 5, (idx + 1) % 5]
            
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn² = {Cn2:.0e}', fontsize=11)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Focal Spot at z = 1000 mm (f={z_focal*1e3:.0f}mm)\n'
                    f'Gaussian Beam (w₀=0.5mm, λ=810nm, 512×512 @ 4µm)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_spot_various_Cn2.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_phase_distribution(self, output_dir):
        """Visualize phase distribution at focal plane.
        焦点处相位分布
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3  # 1000mm
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        for idx, Cn2 in enumerate(Cn2_levels):
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
            
            # Phase at focal plane
            phase = np.angle(E_focal)
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            # Intensity
            ax = axes[0, idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Intensity @ Focus\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Phase
            ax = axes[1, idx]
            im = ax.imshow(phase, cmap='twilight_shifted', origin='lower',
                          extent=[-extent, extent, -extent, extent],
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Phase @ Focus\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle(f'Focal Plane Analysis at z = 1000 mm\n'
                    f'Gaussian Beam Focused through Turbulence',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_phase_distribution.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_intensity_cross_section(self, output_dir):
        """Visualize intensity cross-sections at focal plane.
        焦点处强度剖面
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3  # 1000mm
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        center = cfg.size_x // 2
        x = (np.arange(cfg.size_x) - center) * cfg.pixel_size * 1e3  # mm
        
        # Clean
        E = bs.gauss(cfg=cfg)
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        ax.plot(x, intensity_clean_norm[center, :], 'k-', linewidth=3, 
               label='Clean (no turbulence)', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Cn2_levels)))
        
        for idx, Cn2 in enumerate(Cn2_levels):
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax.plot(x, intensity_norm[center, :], linewidth=2, 
                   label=f'Cn² = {Cn2:.0e}', color=colors[idx], alpha=0.8)
            ax.fill_between(x, intensity_norm[center, :], alpha=0.15, color=colors[idx])
        
        ax.set_xlabel('X Position (mm)', fontsize=14)
        ax.set_ylabel('Normalized Intensity', fontsize=14)
        ax.set_title(f'Focal Spot Cross-Section @ z = {z_focal*1e3:.0f} mm\n'
                    f'Gaussian Beam (w₀=0.5mm, λ=810nm)', fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1.15)
        
        # Add reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(1.8, 0.52, 'FWHM', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_cross_section_Cn2.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_spot_size_vs_turbulence(self, output_dir):
        """Analyze focal spot size vs turbulence strength.
        焦点光斑尺寸与湍流强度关系
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3
        n_realizations = 30
        Cn2_levels = np.logspace(-18, -12, 13)  # 10^-18 to 10^-12
        
        # Clean reference
        E = bs.gauss(cfg=cfg)
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        
        # Calculate clean spot size (FWHM)
        center = cfg.size_x // 2
        profile_clean = intensity_clean[center, :]
        max_val = np.max(profile_clean)
        fwhm_clean = np.sum(profile_clean > max_val/2) * cfg.pixel_size * 1e3
        
        spot_sizes = []
        spot_sizes_std = []
        
        for Cn2 in Cn2_levels:
            sizes = []
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                
                # FWHM spot size
                profile = intensity[center, :]
                max_val = np.max(profile)
                if max_val > 0:
                    fwhm = np.sum(profile > max_val/2) * cfg.pixel_size * 1e3
                else:
                    fwhm = 0
                sizes.append(fwhm)
            
            spot_sizes.append(np.mean(sizes))
            spot_sizes_std.append(np.std(sizes))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Theoretical limit
        theoretical_min = fwhm_clean
        ax.axhline(y=theoretical_min, color='green', linestyle='--', 
                  linewidth=2, label=f'Clean focus (FWHM={theoretical_min:.3f}mm)')
        
        ax.errorbar(Cn2_levels, spot_sizes, yerr=spot_sizes_std, 
                   fmt='o-', linewidth=2, markersize=8, capsize=5,
                   color='red', ecolor='orange', label='Turbulent focus')
        
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=14)
        ax.set_ylabel('FWHM Spot Size (mm)', fontsize=14)
        ax.set_title(f'Focal Spot Size vs Turbulence Strength\n'
                    f'z = {z_focal*1e3:.0f}mm, {n_realizations} realizations each',
                    fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add reference markers
        ax.axvline(x=1e-15, color='blue', linestyle=':', alpha=0.5)
        ax.axvline(x=1e-14, color='blue', linestyle=':', alpha=0.5)
        ax.axvline(x=1e-13, color='blue', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_spot_size_vs_Cn2.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_spot_peak_intensity_distribution(self, output_dir):
        """Analyze peak intensity distribution at focus.
        焦点处峰值强度分布
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3
        n_realizations = 100
        Cn2_levels = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12]
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        peak_clean = np.max(np.abs(E_clean)**2)
        
        fig, axes = plt.subplots(1, len(Cn2_levels) + 1, figsize=(20, 5))
        
        # Clean
        ax = axes[0]
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title(f'Clean\nPeak={peak_clean:.2e}', fontsize=11)
        plt.colorbar(im, ax=ax)
        
        peak_intensities = []
        
        for idx, Cn2 in enumerate(Cn2_levels):
            peaks = []
            E_turb_list = []
            
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                peaks.append(np.max(intensity))
                E_turb_list.append(E_focal)
            
            peaks_normalized = np.array(peaks) / peak_clean
            peak_intensities.append(peaks_normalized)
            
            # Show one example
            ax = axes[idx + 1]
            intensity = np.abs(E_turb_list[0])**2
            intensity_norm = intensity / np.max(intensity)
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            
            mean_peak = np.mean(peaks_normalized)
            std_peak = np.std(peaks_normalized)
            ax.set_title(f'Cn²={Cn2:.0e}\nμ={mean_peak:.2f}, σ={std_peak:.2f}', fontsize=10)
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Focal Peak Intensity Distribution @ z = 1000mm\n'
                    f'(normalized to clean peak = 1.0)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_peak_intensity_distribution.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_spot_histogram(self, output_dir):
        """Histogram of focal spot metrics.
        焦点光斑统计直方图
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3
        n_realizations = 100
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        peak_clean = np.max(np.abs(E_clean)**2)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        for idx, Cn2 in enumerate(Cn2_levels):
            peaks = []
            fwhm_sizes = []
            center = cfg.size_x // 2
            
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                peaks.append(np.max(intensity) / peak_clean)
                
                # FWHM
                profile = intensity[center, :]
                max_val = np.max(profile)
                if max_val > 0:
                    fwhm = np.sum(profile > max_val/2) * cfg.pixel_size * 1e3
                else:
                    fwhm = 0
                fwhm_sizes.append(fwhm)
            
            peaks = np.array(peaks)
            fwhm_sizes = np.array(fwhm_sizes)
            
            # Peak intensity histogram
            ax = axes[0, idx]
            ax.hist(peaks, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Clean peak')
            ax.axvline(x=np.mean(peaks), color='orange', linestyle='-', linewidth=2, 
                      label=f'Mean={np.mean(peaks):.2f}')
            ax.set_xlabel('Normalized Peak Intensity', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'Cn² = {Cn2:.0e}\nPeak Intensity Distribution', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # FWHM histogram
            ax = axes[1, idx]
            ax.hist(fwhm_sizes, bins=20, density=True, alpha=0.7, color='coral', edgecolor='black')
            ax.axvline(x=np.mean(fwhm_sizes), color='darkred', linestyle='-', linewidth=2,
                      label=f'Mean={np.mean(fwhm_sizes):.3f}mm')
            ax.set_xlabel('FWHM Spot Size (mm)', fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(f'Cn² = {Cn2:.0e}\nSpot Size Distribution', fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Focal Spot Statistics @ z = 1000mm\n'
                    f'n={n_realizations} realizations per turbulence level',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_spot_histogram.png')
        plt.savefig(filepath, dpi=250, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_spot_3d_surface(self, output_dir):
        """3D surface plot of focal spots.
        焦点光斑3D表面图
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        E = bs.gauss(cfg=cfg)
        
        # Subsample for 3D plotting
        step = 8
        x = np.arange(0, cfg.size_x, step) * cfg.pixel_size * 1e3  # mm
        y = np.arange(0, cfg.size_y, step) * cfg.pixel_size * 1e3
        X, Y = np.meshgrid(x, y)
        
        fig = plt.figure(figsize=(18, 6))
        
        # Clean
        ax = fig.add_subplot(131, projection='3d')
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        intensity_clean = np.abs(E_clean[::step, ::step])**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        surf = ax.plot_surface(X, Y, intensity_clean_norm, cmap='hot', 
                              linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Intensity')
        ax.set_title('Clean Focus')
        ax.set_zlim(0, 1)
        
        # Medium turbulence
        ax = fig.add_subplot(132, projection='3d')
        phi_turb = bs.turbulence(cfg=cfg, Cn2=1e-14)
        E_turb = E * np.exp(1j * phi_turb)
        E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
        
        intensity = np.abs(E_focal[::step, ::step])**2
        intensity_norm = intensity / np.max(intensity)
        
        surf = ax.plot_surface(X, Y, intensity_norm, cmap='hot',
                              linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Intensity')
        ax.set_title(f'Cn² = 1e-14')
        ax.set_zlim(0, 1)
        
        # Strong turbulence
        ax = fig.add_subplot(133, projection='3d')
        phi_turb = bs.turbulence(cfg=cfg, Cn2=1e-12)
        E_turb = E * np.exp(1j * phi_turb)
        E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
        
        intensity = np.abs(E_focal[::step, ::step])**2
        intensity_norm = intensity / np.max(intensity)
        
        surf = ax.plot_surface(X, Y, intensity_norm, cmap='hot',
                              linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Intensity')
        ax.set_title(f'Cn² = 1e-12')
        ax.set_zlim(0, 1)
        
        plt.suptitle(f'3D Focal Spot Surface @ z = 1000mm', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'focal_spot_3d.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_full_Cn2_range_analysis(self, output_dir):
        """Comprehensive analysis from Cn2 = 10^-18 to 10^-12.
        完整湍流范围分析
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=512,
            size_y=512,
            pixel_size=4e-6,
            w0=0.5e-3,
            z_default=1.0,
            l_max=0.5,
            l_min=1e-3
        )
        
        z_focal = 1000e-3
        n_realizations = 20
        Cn2_levels = np.logspace(-18, -12, 7)
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.propagation(E, z=z_focal, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        peak_clean = np.max(intensity_clean)
        
        center = cfg.size_x // 2
        profile_clean = intensity_clean[center, :]
        fwhm_clean = np.sum(profile_clean > np.max(profile_clean)/2) * cfg.pixel_size * 1e3
        
        results = {'Cn2': [], 'mean_peak': [], 'std_peak': [], 
                   'mean_fwhm': [], 'std_fwhm': [], 'mean_images': []}
        
        for Cn2 in Cn2_levels:
            peaks = []
            fwhms = []
            images = []
            
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.propagation(E_turb, z=z_focal, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                peaks.append(np.max(intensity) / peak_clean)
                images.append(intensity)
                
                # FWHM
                profile = intensity[center, :]
                max_val = np.max(profile)
                if max_val > 0:
                    fwhm = np.sum(profile > max_val/2) * cfg.pixel_size * 1e3
                else:
                    fwhm = 0
                fwhms.append(fwhm)
            
            results['Cn2'].append(Cn2)
            results['mean_peak'].append(np.mean(peaks))
            results['std_peak'].append(np.std(peaks))
            results['mean_fwhm'].append(np.mean(fwhms))
            results['std_fwhm'].append(np.std(fwhms))
            results['mean_images'].append(np.mean(images, axis=0))
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 7, figure=fig, hspace=0.35, wspace=0.3)
        
        # Row 1: Sample focal spots
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        for idx, Cn2 in enumerate(Cn2_levels):
            ax = fig.add_subplot(gs[0, idx])
            mean_img = results['mean_images'][idx]
            mean_img_norm = mean_img / np.max(mean_img) if idx > 0 else mean_img / peak_clean
            
            im = ax.imshow(mean_img_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn²={Cn2:.0e}', fontsize=9)
            ax.set_xlabel('X (mm)', fontsize=8)
            ax.set_ylabel('Y (mm)', fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: Peak intensity vs Cn2
        ax = fig.add_subplot(gs[1, :3])
        ax.errorbar(results['Cn2'], results['mean_peak'], yerr=results['std_peak'],
                   fmt='o-', linewidth=2, markersize=10, capsize=5,
                   color='blue', ecolor='lightblue', label='Mean ± Std')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Clean peak')
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=12)
        ax.set_ylabel('Normalized Peak Intensity', fontsize=12)
        ax.set_title('Peak Intensity vs Turbulence Strength', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 2: FWHM vs Cn2
        ax = fig.add_subplot(gs[1, 4:])
        ax.errorbar(results['Cn2'], results['mean_fwhm'], yerr=results['std_fwhm'],
                   fmt='s-', linewidth=2, markersize=10, capsize=5,
                   color='red', ecolor='lightcoral', label='Mean ± Std')
        ax.axhline(y=fwhm_clean, color='green', linestyle='--', linewidth=2, 
                  label=f'Clean (FWHM={fwhm_clean:.3f}mm)')
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=12)
        ax.set_ylabel('FWHM Spot Size (mm)', fontsize=12)
        ax.set_title('Spot Size vs Turbulence Strength', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 3: Cross-sections
        ax = fig.add_subplot(gs[2, :])
        x = (np.arange(cfg.size_x) - center) * cfg.pixel_size * 1e3
        
        ax.plot(x, profile_clean / np.max(profile_clean), 'k-', linewidth=3, 
               label='Clean', alpha=0.9)
        
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(Cn2_levels)))
        for idx, Cn2 in enumerate(Cn2_levels):
            mean_img = results['mean_images'][idx]
            profile = mean_img[center, :] / np.max(mean_img)
            ax.plot(x, profile, linewidth=2, color=colors[idx],
                   label=f'Cn²={Cn2:.0e}', alpha=0.8)
        
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)
        ax.set_title('Mean Intensity Cross-Sections @ Focal Plane', fontsize=12)
        ax.legend(fontsize=9, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1.15)
        
        plt.suptitle(f'Focused Gaussian Beam Atmospheric Propagation Analysis\n'
                    f'z = 1000mm | w₀ = 0.5mm | λ = 810nm | {n_realizations} realizations',
                    fontsize=14, fontweight='bold')
        
        filepath = os.path.join(output_dir, 'focal_full_Cn2_analysis.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
