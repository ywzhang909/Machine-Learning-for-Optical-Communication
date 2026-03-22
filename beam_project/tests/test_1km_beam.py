"""
Long-range focused Gaussian beam at 1km - 1km长距离聚焦光大气传播
Tests focal spot images and phase distribution at 1000m with Cn2 from 10^-18 to 10^-12
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import beam_simulation as bs


class Test1kmFocusedBeam:
    """1km聚焦光束测试"""

    def test_1km_propagation(self):
        """Test 1km propagation basic functionality."""
        # For 1km propagation, we need larger grid
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,    # Larger grid for long distance
            size_y=1024,
            pixel_size=10e-6,  # 10µm pixels
            w0=10e-3,       # Larger beam waist for 1km
            z_default=1000.0,
            Cn2=1e-14,
            l_max=1.0,      # Larger outer scale for long path
            l_min=1e-3
        )
        
        E = bs.gauss(cfg=cfg)
        E_prop = bs.propagation(E, z=1000.0, cfg=cfg)
        
        assert E_prop.shape == E.shape
        assert np.max(np.abs(E_prop)**2) > 0

    def test_1km_various_turbulence(self):
        """Test 1km propagation with various Cn2."""
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        
        for Cn2 in Cn2_levels:
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_prop = bs.propagation(E_turb, z=1000.0, cfg=cfg)
            
            assert E_prop.shape == E.shape
            intensity = np.abs(E_prop)**2
            assert np.max(intensity) > 0


class Test1kmFocusedVisualization:
    """1km聚焦光束可视化"""

    def test_1km_focal_spot_various_Cn2(self, output_dir):
        """Visualize 1km focal spot for Cn2 from 10^-18 to 10^-12.
        1km焦点处不同湍流强度的光斑图像
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,  # 10µm pixel
            w0=10e-3,         # 10mm beam waist
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0  # 1km = 1000m
        Cn2_levels = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 5e-13, 1e-12]
        
        # Clean propagation
        E = bs.gauss(cfg=cfg)
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        fig, axes = plt.subplots(2, 5, figsize=(24, 11))
        
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2  # mm
        
        # First: Clean
        ax = axes[0, 0]
        im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Clean (No Turbulence)\nz = 1000m', fontsize=11)
        ax.set_xlabel('X (mm)', fontsize=10)
        ax.set_ylabel('Y (mm)', fontsize=10)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Turbulent focal spots
        for idx, Cn2 in enumerate(Cn2_levels):
            ax = axes[(idx + 1) // 5, (idx + 1) % 5]
            
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn² = {Cn2:.0e}', fontsize=11)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Focused Gaussian Beam @ z = 1000m (1km)\n'
                    f'w₀ = {cfg.w0*1e3:.0f}mm, λ = {cfg.wavelength*1e9:.0f}nm, '
                    f'{cfg.size_x}×{cfg.size_y} @ {cfg.pixel_size*1e6:.0f}µm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_focal_spot_various_Cn2.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_phase_distribution(self, output_dir):
        """Visualize phase distribution at 1km focal plane.
        1km焦点处相位分布
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        for idx, Cn2 in enumerate(Cn2_levels):
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
            
            # Phase
            phase = np.angle(E_focal)
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            # Intensity
            ax = axes[0, idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Intensity @ 1km\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Phase
            ax = axes[1, idx]
            im = ax.imshow(phase, cmap='twilight_shifted', origin='lower',
                          extent=[-extent, extent, -extent, extent],
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Phase @ 1km\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle(f'Focal Plane Analysis at z = 1000m (1km)\n'
                    f'Gaussian Beam Through Atmospheric Turbulence',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_focal_phase_distribution.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_cross_section_comparison(self, output_dir):
        """Compare intensity cross-sections at 1km.
        1km处强度剖面对比
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        center = cfg.size_x // 2
        x = (np.arange(cfg.size_x) - center) * cfg.pixel_size * 1e3  # mm
        
        # Clean
        E = bs.gauss(cfg=cfg)
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        ax.plot(x, intensity_clean_norm[center, :], 'k-', linewidth=3, 
               label='Clean (no turbulence)', alpha=0.9)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Cn2_levels)))
        
        for idx, Cn2 in enumerate(Cn2_levels):
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax.plot(x, intensity_norm[center, :], linewidth=2.5, 
                   label=f'Cn² = {Cn2:.0e}', color=colors[idx], alpha=0.85)
            ax.fill_between(x, intensity_norm[center, :], alpha=0.12, color=colors[idx])
        
        ax.set_xlabel('X Position (mm)', fontsize=14)
        ax.set_ylabel('Normalized Intensity', fontsize=14)
        ax.set_title(f'Intensity Cross-Section @ z = 1000m (1km)\n'
                    f'Focused Gaussian Beam (w₀={cfg.w0*1e3:.0f}mm, λ={cfg.wavelength*1e9:.0f}nm)',
                    fontsize=14)
        ax.legend(fontsize=11, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-50, 50)
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_cross_section_Cn2.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_spot_size_vs_turbulence(self, output_dir):
        """Analyze 1km spot size vs turbulence strength.
        1km光斑尺寸与湍流强度关系
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        n_realizations = 20
        Cn2_levels = np.logspace(-18, -12, 13)
        
        # Clean reference
        E = bs.gauss(cfg=cfg)
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
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
                E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
                
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
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Theoretical limit
        ax.axhline(y=fwhm_clean, color='green', linestyle='--', 
                  linewidth=2.5, label=f'Clean beam (FWHM={fwhm_clean:.1f}mm)')
        
        ax.errorbar(Cn2_levels, spot_sizes, yerr=spot_sizes_std, 
                   fmt='o-', linewidth=2.5, markersize=10, capsize=6,
                   color='red', ecolor='orange', label='With turbulence (mean±std)')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=14)
        ax.set_ylabel('FWHM Spot Size (mm)', fontsize=14)
        ax.set_title(f'Focal Spot Size vs Turbulence Strength @ 1km\n'
                    f'z = {z_distance:.0f}m, {n_realizations} realizations, '
                    f'w₀ = {cfg.w0*1e3:.0f}mm',
                    fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_spot_size_vs_Cn2.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_full_Cn2_analysis(self, output_dir):
        """Comprehensive 1km analysis from Cn2 = 10^-18 to 10^-12.
        1km完整湍流范围分析
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        n_realizations = 15
        Cn2_levels = np.logspace(-18, -12, 7)
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
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
                E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
                
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
        fig = plt.figure(figsize=(22, 14))
        gs = GridSpec(3, 7, figure=fig, hspace=0.35, wspace=0.35)
        
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        # Row 1: Sample focal spots
        for idx, Cn2 in enumerate(Cn2_levels):
            ax = fig.add_subplot(gs[0, idx])
            mean_img = results['mean_images'][idx]
            if idx == 0:
                mean_img_norm = mean_img / peak_clean
            else:
                mean_img_norm = mean_img / np.max(mean_img)
            
            im = ax.imshow(mean_img_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn²={Cn2:.0e}', fontsize=10)
            ax.set_xlabel('X (mm)', fontsize=9)
            ax.set_ylabel('Y (mm)', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 2: Peak intensity vs Cn2
        ax = fig.add_subplot(gs[1, :3])
        ax.errorbar(results['Cn2'], results['mean_peak'], yerr=results['std_peak'],
                   fmt='o-', linewidth=2.5, markersize=12, capsize=6,
                   color='blue', ecolor='lightblue', label='Mean ± Std')
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2.5, label='Clean peak')
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=13)
        ax.set_ylabel('Normalized Peak Intensity', fontsize=13)
        ax.set_title('Peak Intensity vs Turbulence @ 1km', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 2: FWHM vs Cn2
        ax = fig.add_subplot(gs[1, 4:])
        ax.errorbar(results['Cn2'], results['mean_fwhm'], yerr=results['std_fwhm'],
                   fmt='s-', linewidth=2.5, markersize=12, capsize=6,
                   color='red', ecolor='lightcoral', label='Mean ± Std')
        ax.axhline(y=fwhm_clean, color='green', linestyle='--', linewidth=2.5, 
                  label=f'Clean (FWHM={fwhm_clean:.1f}mm)')
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=13)
        ax.set_ylabel('FWHM Spot Size (mm)', fontsize=13)
        ax.set_title('Spot Size vs Turbulence @ 1km', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 3: Cross-sections
        ax = fig.add_subplot(gs[2, :])
        x = (np.arange(cfg.size_x) - center) * cfg.pixel_size * 1e3
        
        ax.plot(x, profile_clean / np.max(profile_clean), 'k-', linewidth=3.5, 
               label='Clean', alpha=0.95)
        
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(Cn2_levels)))
        for idx, Cn2 in enumerate(Cn2_levels):
            mean_img = results['mean_images'][idx]
            profile = mean_img[center, :] / np.max(mean_img)
            ax.plot(x, profile, linewidth=2.5, color=colors[idx],
                   label=f'Cn²={Cn2:.0e}', alpha=0.85)
        
        ax.set_xlabel('X Position (mm)', fontsize=13)
        ax.set_ylabel('Normalized Intensity', fontsize=13)
        ax.set_title('Mean Intensity Cross-Sections @ 1km Focal Plane', fontsize=13)
        ax.legend(fontsize=9, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-80, 80)
        ax.set_ylim(0, 1.15)
        
        plt.suptitle(f'Focused Gaussian Beam @ 1km (z = 1000m)\n'
                    f'w₀ = {cfg.w0*1e3:.0f}mm | λ = {cfg.wavelength*1e9:.0f}nm | '
                    f'{cfg.size_x}×{cfg.size_y} @ {cfg.pixel_size*1e6:.0f}µm | '
                    f'{n_realizations} realizations per Cn²',
                    fontsize=15, fontweight='bold')
        
        filepath = os.path.join(output_dir, '1km_full_Cn2_analysis.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_longitudinal_propagation(self, output_dir):
        """Visualize beam evolution over 1km path.
        1km路径上的光束演变
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_levels = [100.0, 300.0, 500.0, 700.0, 900.0, 1000.0]  # m
        Cn2 = 1e-14  # Moderate turbulence
        
        E = bs.gauss(cfg=cfg)
        phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
        E_turb = E * np.exp(1j * phi_turb)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        for idx, z in enumerate(z_levels):
            E_prop = bs.propagation(E_turb, z=z, cfg=cfg)
            intensity = np.abs(E_prop)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[idx // 3, idx % 3]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'z = {z:.0f}m', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Gaussian Beam Propagation through Turbulence (Cn² = {Cn2:.0e})\n'
                    f'w₀ = {cfg.w0*1e3:.0f}mm, λ = {cfg.wavelength*1e9:.0f}nm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_longitudinal_propagation.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_clean_vs_turbulent(self, output_dir):
        """Compare clean and turbulent propagation at 1km.
        1km清洁与湍流传播对比
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        E = bs.gauss(cfg=cfg)
        
        # Clean
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        fig, axes = plt.subplots(2, 4, figsize=(22, 11))
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        # Clean
        ax = axes[0, 0]
        im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Clean Beam\n(No Turbulence)', fontsize=11)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax)
        
        # Turbulent
        for idx, Cn2 in enumerate(Cn2_levels):
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[0, idx + 1]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn² = {Cn2:.0e}', fontsize=11)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax)
        
        # Cross-sections
        center = cfg.size_x // 2
        x = (np.arange(cfg.size_x) - center) * cfg.pixel_size * 1e3
        
        ax = axes[1, 0]
        ax.plot(x, intensity_clean_norm[center, :], 'k-', linewidth=2.5, 
               label='Clean', alpha=0.9)
        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Intensity', fontsize=11)
        ax.set_title('Clean Cross-Section', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-60, 60)
        
        for idx, Cn2 in enumerate(Cn2_levels):
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[1, idx + 1]
            ax.plot(x, intensity_norm[center, :], 'r-', linewidth=2, alpha=0.8)
            ax.plot(x, intensity_clean_norm[center, :], 'k--', linewidth=1.5, alpha=0.5)
            ax.set_xlabel('X (mm)', fontsize=11)
            ax.set_ylabel('Intensity', fontsize=11)
            ax.set_title(f'Cn² = {Cn2:.0e}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-60, 60)
            ax.set_ylim(0, 1.1)
        
        plt.suptitle(f'Clean vs Turbulent Propagation @ 1km (z = 1000m)\n'
                    f'w₀ = {cfg.w0*1e3:.0f}mm, λ = {cfg.wavelength*1e9:.0f}nm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_clean_vs_turbulent.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_1km_scintillation_analysis(self, output_dir):
        """Analyze scintillation at 1km.
        1km闪烁分析
        """
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=1024,
            size_y=1024,
            pixel_size=10e-6,
            w0=10e-3,
            z_default=1000.0,
            l_max=1.0,
            l_min=1e-3
        )
        
        z_distance = 1000.0
        n_realizations = 50
        Cn2_levels = [1e-16, 1e-15, 1e-14, 1e-13, 1e-12]
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.propagation(E, z=z_distance, cfg=cfg)
        peak_clean = np.max(np.abs(E_clean)**2)
        
        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        extent = cfg.size_x * cfg.pixel_size * 1e3 / 2
        
        for idx, Cn2 in enumerate(Cn2_levels):
            peaks = []
            E_samples = []
            
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.propagation(E_turb, z=z_distance, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                peaks.append(np.max(intensity) / peak_clean)
                E_samples.append(E_focal)
            
            peaks = np.array(peaks)
            
            # Sample image
            ax = axes[0, idx]
            intensity = np.abs(E_samples[0])**2
            intensity_norm = intensity / np.max(intensity)
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            mean_peak = np.mean(peaks)
            std_peak = np.std(peaks)
            scint_index = np.var(peaks) / (mean_peak ** 2)
            ax.set_title(f'Cn²={Cn2:.0e}\nσ_I²={scint_index:.2f}', fontsize=10)
            plt.colorbar(im, ax=ax)
            
            # Histogram
            ax = axes[1, idx]
            ax.hist(peaks, bins=15, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='black')
            ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Clean')
            ax.axvline(x=mean_peak, color='orange', linestyle='-', linewidth=2, 
                      label=f'μ={mean_peak:.2f}')
            ax.set_xlabel('Norm. Peak Intensity', fontsize=10)
            ax.set_ylabel('PDF', fontsize=10)
            ax.set_title(f'Distribution', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Scintillation Analysis @ 1km (z = 1000m)\n'
                    f'n = {n_realizations} realizations',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '1km_scintillation.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
