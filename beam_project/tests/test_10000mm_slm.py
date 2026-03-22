"""
Focused beam at 10000mm with turbulence and SLM correction - 10000mm聚焦光束与SLM校正
Tests focal spot, cross-section, and phase distribution at 10000mm with various Cn2 values
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import beam_simulation as bs


class Test10000mmFocusedBeam:
    """10000mm聚焦光束测试"""

    def test_10000mm_focal_with_turbulence(self):
        """Test 10000mm focal with turbulence."""
        f = 10000e-3  # 10000mm = 10m
        size = 1024
        pixel = 15e-6
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=200e-3,  # 200mm beam waist
            l_max=5.0,
            l_min=1e-3
        )
        
        E = bs.gauss(cfg=cfg)
        phi_turb = bs.turbulence(cfg=cfg, Cn2=1e-14)
        E_turb = E * np.exp(1j * phi_turb)
        E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
        
        assert E_focal.shape == E.shape
        assert np.max(np.abs(E_focal)**2) > 0

    def test_10000mm_various_Cn2(self):
        """Test 10000mm focal at various Cn2."""
        f = 10000e-3
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        
        for Cn2 in Cn2_levels:
            cfg = bs.Config(
                wavelength=810e-9,
                size_x=512, size_y=512,
                pixel_size=20e-6,
                w0=200e-3
            )
            
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
            
            assert E_focal.shape == E.shape


class Test10000mmVisualization:
    """10000mm聚焦可视化测试"""

    def test_10000mm_focal_spot_various_Cn2(self, output_dir):
        """Visualize 10000mm focal spot at various Cn2 levels.
        10000mm焦距下不同Cn2的光斑图像
        """
        f = 10000e-3  # 10000mm = 10m
        size = 1024
        pixel = 15e-6
        w0 = 200e-3  # 200mm beam waist
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0,
            l_max=5.0,
            l_min=1e-3
        )
        
        # Clean focal spot
        E = bs.gauss(cfg=cfg)
        E_clean = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        Cn2_levels = [1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 5e-13, 1e-12]
        
        fig, axes = plt.subplots(2, 5, figsize=(24, 11))
        
        extent = size * pixel * 1e3 / 2  # mm
        
        # First: Clean focal spot
        ax = axes[0, 0]
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
            E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Cn² = {Cn2:.0e}', fontsize=11)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle(f'Focused Gaussian Beam @ f = {f*1e3:.0f} mm (10m)\n'
                    f'λ = {cfg.wavelength*1e9:.0f}nm, w₀ = {w0*1e3:.0f}mm, '
                    f'{size}×{size} @ {pixel*1e6:.0f}µm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '10000mm_focal_spot_various_Cn2.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_10000mm_cross_section(self, output_dir):
        """Visualize 10000mm focal cross-sections at various Cn2.
        10000mm焦距下不同Cn2的强度剖面
        """
        f = 10000e-3
        size = 1024
        pixel = 15e-6
        w0 = 200e-3
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0
        )
        
        Cn2_levels = [1e-18, 1e-16, 1e-14, 1e-12]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3  # mm
        
        # Clean
        E = bs.gauss(cfg=cfg)
        E_clean = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        ax.plot(x, intensity_clean_norm[center, :], 'k-', linewidth=3, 
               label='Clean (no turbulence)', alpha=0.9)
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(Cn2_levels)))
        
        for idx, Cn2 in enumerate(Cn2_levels):
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax.plot(x, intensity_norm[center, :], linewidth=2.5,
                   label=f'Cn² = {Cn2:.0e}', color=colors[idx], alpha=0.85)
            ax.fill_between(x, intensity_norm[center, :], alpha=0.1, color=colors[idx])
        
        ax.set_xlabel('X Position (mm)', fontsize=14)
        ax.set_ylabel('Normalized Intensity', fontsize=14)
        ax.set_title(f'Focal Spot Cross-Section @ f = {f*1e3:.0f} mm (10m)\n'
                    f'λ = {cfg.wavelength*1e9:.0f}nm, w₀ = {w0*1e3:.0f}mm',
                    fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '10000mm_cross_section_various_Cn2.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_10000mm_phase_distribution(self, output_dir):
        """Visualize 10000mm focal phase at various Cn2.
        10000mm焦距下不同Cn2的相位分布
        """
        f = 10000e-3
        size = 1024
        pixel = 15e-6
        w0 = 200e-3
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0
        )
        
        Cn2_levels = [1e-16, 1e-14, 1e-12]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        extent = size * pixel * 1e3 / 2
        
        for idx, Cn2 in enumerate(Cn2_levels):
            E = bs.gauss(cfg=cfg)
            phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
            E_turb = E * np.exp(1j * phi_turb)
            E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
            
            # Intensity
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            ax = axes[0, idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'Intensity @ Focus\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Intensity')
            
            # Phase
            phase = np.angle(E_focal)
            ax = axes[1, idx]
            im = ax.imshow(phase, cmap='twilight_shifted', origin='lower',
                          extent=[-extent, extent, -extent, extent],
                          vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Phase @ Focus\nCn² = {Cn2:.0e}', fontsize=12)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle(f'Focal Plane @ f = {f*1e3:.0f} mm (10m) - Intensity and Phase\n'
                    f'λ = {cfg.wavelength*1e9:.0f}nm, w₀ = {w0*1e3:.0f}mm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, '10000mm_phase_distribution.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_10000mm_comprehensive_analysis(self, output_dir):
        """Comprehensive analysis of 10000mm focus with various Cn2.
        10000mm焦距综合分析
        """
        f = 10000e-3
        size = 1024
        pixel = 15e-6
        w0 = 200e-3
        n_realizations = 15
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0
        )
        
        Cn2_levels = np.logspace(-18, -12, 7)
        
        E = bs.gauss(cfg=cfg)
        
        # Clean reference
        E_clean = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        intensity_clean = np.abs(E_clean)**2
        peak_clean = np.max(intensity_clean)
        
        center = size // 2
        profile_clean = intensity_clean[center, :]
        fwhm_clean = np.sum(profile_clean > np.max(profile_clean)/2) * pixel * 1e3
        
        results = {'Cn2': [], 'mean_peak': [], 'std_peak': [],
                   'mean_fwhm': [], 'std_fwhm': [], 'mean_images': []}
        
        for Cn2 in Cn2_levels:
            peaks = []
            fwhms = []
            images = []
            
            for _ in range(n_realizations):
                phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
                E_turb = E * np.exp(1j * phi_turb)
                E_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
                
                intensity = np.abs(E_focal)**2
                peaks.append(np.max(intensity) / peak_clean)
                images.append(intensity)
                
                # FWHM
                profile = intensity[center, :]
                max_val = np.max(profile)
                if max_val > 0:
                    fwhm = np.sum(profile > max_val/2) * pixel * 1e3
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
        
        extent = size * pixel * 1e3 / 2
        
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
        ax.set_title('Peak Intensity vs Turbulence', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 2: FWHM vs Cn2
        ax = fig.add_subplot(gs[1, 4:])
        ax.errorbar(results['Cn2'], results['mean_fwhm'], yerr=results['std_fwhm'],
                   fmt='s-', linewidth=2.5, markersize=12, capsize=6,
                   color='red', ecolor='lightcoral', label='Mean ± Std')
        ax.axhline(y=fwhm_clean, color='green', linestyle='--', linewidth=2.5,
                  label=f'Clean (FWHM={fwhm_clean:.2f}mm)')
        ax.set_xscale('log')
        ax.set_xlabel('Cn² (m⁻²/³)', fontsize=13)
        ax.set_ylabel('FWHM Spot Size (mm)', fontsize=13)
        ax.set_title('Spot Size vs Turbulence', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        # Row 3: Cross-sections
        ax = fig.add_subplot(gs[2, :])
        x = (np.arange(size) - center) * pixel * 1e3
        
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
        ax.set_title('Mean Intensity Cross-Sections @ Focal Plane', fontsize=13)
        ax.legend(fontsize=9, ncol=4, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 1.15)
        
        plt.suptitle(f'Focused Gaussian Beam @ f = {f*1e3:.0f} mm (10m)\n'
                    f'λ = {cfg.wavelength*1e9:.0f}nm | w₀ = {w0*1e3:.0f}mm | '
                    f'{n_realizations} realizations per Cn²',
                    fontsize=15, fontweight='bold')
        
        filepath = os.path.join(output_dir, '10000mm_comprehensive_analysis.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)


class TestSLMCorrection:
    """SLM相位校正测试"""

    def test_zernike_basis_generation(self):
        """Test Zernike basis generation."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=10e-6)
        X, Y = cfg.grid()
        
        # Test generating a few Zernike modes
        from beam_simulation.propagation.slm import generate_zernike_map, zernike_name
        
        for noll_idx in [1, 2, 4, 11]:  # Piston, Tilt, Defocus, Spherical
            Z = generate_zernike_map(noll_idx, X, Y)
            assert Z.shape == (256, 256)
            assert zernike_name(noll_idx) is not None

    def test_slm_phase_generation(self):
        """Test SLM phase from Zernike coefficients."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=10e-6)
        X, Y = cfg.grid()
        
        from beam_simulation.propagation.slm import slm_phase_from_zernike
        
        coeffs = {1: 0, 2: 0.5, 3: 0.3, 4: -0.1}  # Some coefficients
        phase = slm_phase_from_zernike(coeffs, X, Y)
        
        assert phase.shape == (256, 256)
        assert np.min(phase) >= 0
        assert np.max(phase) <= 2 * np.pi

    def test_slm_correction_simulation(self):
        """Test SLM correction simulation."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=10e-6)
        
        E = bs.gauss(cfg=cfg)
        phi_turb = bs.turbulence(cfg=cfg, Cn2=1e-14)
        
        from beam_simulation.propagation.slm import simulate_slm
        
        E_corrected, X, Y = simulate_slm(
            E=E,
            cfg=cfg,
            turbulence_phase=phi_turb,
            zernike_coeffs={2: 0.1, 3: 0.1, 4: 0.05}  # Simple correction
        )
        
        assert E_corrected.shape == E.shape

    def test_slm_correction_visualization(self, output_dir):
        """Visualize SLM phase correction effect.
        SLM相位校正效果图
        """
        f = 10000e-3  # 10m focal length
        size = 512
        pixel = 20e-6
        w0 = 150e-3
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0
        )
        
        X, Y = cfg.grid()
        
        # Turbulence phase
        Cn2 = 1e-14
        phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
        
        # Fit Zernike coefficients
        from beam_simulation.propagation.slm import fit_zernike_to_phase, slm_phase_from_zernike
        
        coeffs = fit_zernike_to_phase(phi_turb, X, Y, n_max=15)
        
        # Generate correction phase
        phi_correction = slm_phase_from_zernike(coeffs, X, Y)
        
        # Apply to beam
        E = bs.gauss(cfg=cfg)
        E_turb = E * np.exp(1j * phi_turb)
        E_corrected = E_turb * np.exp(1j * phi_correction)
        
        # Propagate to focus
        E_turb_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
        E_corrected_focal = bs.lens_fft_propagation_to_focal(E_corrected, f=f, cfg=cfg)
        E_clean_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        
        intensity_turb = np.abs(E_turb_focal)**2
        intensity_corrected = np.abs(E_corrected_focal)**2
        intensity_clean = np.abs(E_clean_focal)**2
        
        # Normalize
        intensity_turb_norm = intensity_turb / np.max(intensity_clean)
        intensity_corrected_norm = intensity_corrected / np.max(intensity_clean)
        intensity_clean_norm = intensity_clean / np.max(intensity_clean)
        
        extent = size * pixel * 1e3 / 2
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: Focal spots
        ax = axes[0, 0]
        im = ax.imshow(intensity_clean_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Clean Focus', fontsize=12)
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.imshow(intensity_turb_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title(f'Turbulent (Cn²={Cn2:.0e})', fontsize=12)
        plt.colorbar(im, ax=ax)
        
        ax = axes[0, 2]
        im = ax.imshow(intensity_corrected_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('SLM Corrected', fontsize=12)
        plt.colorbar(im, ax=ax)
        
        # Row 2: Cross-sections
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3
        
        ax = axes[1, 0]
        ax.plot(x, intensity_clean_norm[center, :], 'g-', linewidth=2.5, label='Clean')
        ax.plot(x, intensity_turb_norm[center, :], 'r-', linewidth=2, alpha=0.7, label='Turbulent')
        ax.plot(x, intensity_corrected_norm[center, :], 'b-', linewidth=2, alpha=0.8, label='Corrected')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Cross-Section Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-15, 15)
        
        # Zernike coefficients
        ax = axes[1, 1]
        coeff_items = sorted(coeffs.items(), key=lambda x: x[0])
        indices = [c[0] for c in coeff_items]
        values = [c[1] for c in coeff_items]
        names = [bs.zernike_name(i) for i in indices]
        bars = ax.bar(range(len(indices)), values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([f'Z{i}\n{n}' for i, n in zip(indices, names)], fontsize=8)
        ax.set_xlabel('Zernike Mode')
        ax.set_ylabel('Coefficient (waves)')
        ax.set_title('Fitted Zernike Coefficients')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Phase maps
        ax = axes[1, 2]
        im = ax.imshow(phi_turb, cmap='RdBu', origin='lower',
                      extent=[-extent, extent, -extent, extent],
                      vmin=-np.pi, vmax=np.pi)
        ax.set_title('Turbulence Phase')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        plt.suptitle(f'SLM Phase Correction @ f = {f*1e3:.0f} mm\n'
                    f'λ = {cfg.wavelength*1e9:.0f}nm, Zernike modes 1-{15}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'slm_correction_effect.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_slm_correction_various_modes(self, output_dir):
        """Visualize SLM correction with various Zernike mode counts.
        不同Zernike模式数的SLM校正效果
        """
        f = 10000e-3
        size = 512
        pixel = 20e-6
        w0 = 150e-3
        Cn2 = 1e-14
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=w0
        )
        
        X, Y = cfg.grid()
        
        from beam_simulation.propagation.slm import fit_zernike_to_phase, slm_phase_from_zernike
        
        # Turbulence phase
        phi_turb = bs.turbulence(cfg=cfg, Cn2=Cn2)
        
        # Clean focus
        E = bs.gauss(cfg=cfg)
        E_clean_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        intensity_clean = np.abs(E_clean_focal)**2
        
        # Turbulent focus
        E_turb = E * np.exp(1j * phi_turb)
        E_turb_focal = bs.lens_fft_propagation_to_focal(E_turb, f=f, cfg=cfg)
        intensity_turb = np.abs(E_turb_focal)**2
        
        mode_counts = [3, 6, 10, 15, 21]
        
        fig, axes = plt.subplots(2, len(mode_counts) + 2, figsize=(22, 8))
        extent = size * pixel * 1e3 / 2
        
        # Clean
        ax = axes[0, 0]
        im = ax.imshow(intensity_clean / np.max(intensity_clean), cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Clean', fontsize=10)
        plt.colorbar(im, ax=ax)
        
        # Turbulent
        ax = axes[0, 1]
        im = ax.imshow(intensity_turb / np.max(intensity_clean), cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title('Turbulent', fontsize=10)
        plt.colorbar(im, ax=ax)
        
        # Various corrections
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3
        
        for idx, n_modes in enumerate(mode_counts):
            # Fit and correct
            coeffs = fit_zernike_to_phase(phi_turb, X, Y, n_max=n_modes)
            phi_correction = slm_phase_from_zernike(coeffs, X, Y)
            E_corrected = E_turb * np.exp(1j * phi_correction)
            E_corrected_focal = bs.lens_fft_propagation_to_focal(E_corrected, f=f, cfg=cfg)
            intensity_corrected = np.abs(E_corrected_focal)**2
            intensity_norm = intensity_corrected / np.max(intensity_clean)
            
            # Image
            ax = axes[0, idx + 2]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'{n_modes} modes', fontsize=10)
            plt.colorbar(im, ax=ax)
            
            # Cross-section
            ax = axes[1, idx + 2]
            ax.plot(x, intensity_clean[center, :] / np.max(intensity_clean), 'g-', 
                   linewidth=2, alpha=0.7, label='Clean')
            ax.plot(x, intensity_turb[center, :] / np.max(intensity_clean), 'r--', 
                   linewidth=1.5, alpha=0.5, label='Turb')
            ax.plot(x, intensity_norm[center, :], 'b-', linewidth=2, label='Corrected')
            ax.set_xlabel('X (mm)', fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-15, 15)
            ax.set_ylim(0, 1.2)
            if idx == 0:
                ax.legend(fontsize=7)
        
        plt.suptitle(f'SLM Correction with Various Zernike Modes @ f = {f*1e3:.0f}mm\n'
                    f'Cn² = {Cn2:.0e}',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'slm_various_modes.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
