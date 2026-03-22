"""
Ideal focused beam simulation - 理想聚焦光束仿真
Tests ideal focused spot at various focal lengths from 10mm to 1km
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import beam_simulation as bs


class TestIdealFocusedSpot:
    """理想聚焦光斑测试"""

    def test_lens_phase_generation(self):
        """Test lens phase can be generated."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=10e-6)
        f = 100e-3  # 100mm focal length
        phi = bs.lens_phase(f=f, cfg=cfg)
        assert phi.shape == (256, 256)
        assert np.max(np.abs(phi)) > 0

    def test_apply_lens(self):
        """Test lens can be applied to beam."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=10e-6)
        E = bs.gauss(cfg=cfg)
        f = 50e-3  # 50mm focal length
        E_lensed = bs.apply_lens(E, f=f, cfg=cfg)
        assert E_lensed.shape == E.shape
        assert np.max(np.abs(E_lensed)) > 0

    def test_ideal_focus_short_distance(self):
        """Test ideal focus at short distance (10mm)."""
        cfg = bs.Config(wavelength=810e-9, size_x=256, size_y=256, pixel_size=5e-6)
        E = bs.gauss(cfg=cfg)
        f = 10e-3  # 10mm focal length
        E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        assert E_focal.shape == E.shape
        intensity = np.abs(E_focal)**2
        assert np.max(intensity) > 0


class TestIdealFocusVisualization:
    """理想聚焦可视化测试"""

    def test_ideal_focus_various_focal_lengths(self, output_dir):
        """Visualize ideal focused spots at various focal lengths from 10mm to 1km.
        不同焦距下的理想聚焦光斑 (10mm ~ 1km)
        """
        # Use different configurations for different focal lengths
        focal_lengths = [
            (10e-3, 256, 5e-6),    # 10mm - short focal, small pixels
            (50e-3, 256, 8e-6),    # 50mm
            (100e-3, 256, 10e-6),  # 100mm
            (500e-3, 512, 10e-6),   # 500mm
            (1000e-3, 512, 10e-6), # 1000mm (1m)
            (10000e-3, 1024, 10e-6), # 10000mm (10m)
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (f, size, pixel) in enumerate(focal_lengths):
            w0_default = 0.45e-3  # Default beam waist
            w0 = w0_default if idx == 0 else f/20  # Adaptive beam waist
            cfg = bs.Config(
                wavelength=810e-9,
                size_x=size, size_y=size,
                pixel_size=pixel,
                w0=w0
            )
            
            E = bs.gauss(cfg=cfg)
            E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            extent = size * pixel * 1e3 / 2  # mm
            
            ax = axes[idx]
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'f = {f*1e3:.0f} mm\n{size}×{size} @ {pixel*1e6:.0f}µm', fontsize=11)
            ax.set_xlabel('X (mm)', fontsize=10)
            ax.set_ylabel('Y (mm)', fontsize=10)
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.suptitle('Ideal Focused Gaussian Beam Spots\n(various focal lengths from 10mm to 10m)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ideal_focus_various_lengths.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_ideal_focus_cross_sections(self, output_dir):
        """Visualize intensity cross-sections at various focal lengths.
        不同焦距下的强度剖面
        """
        focal_lengths = [50e-3, 100e-3, 500e-3, 1000e-3]  # mm
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(focal_lengths)))
        
        for idx, f in enumerate(focal_lengths):
            # Adaptive config
            size = 512 if f > 200e-3 else 256
            pixel = 10e-6 if f > 100e-3 else 8e-6
            
            cfg = bs.Config(
                wavelength=810e-9,
                size_x=size, size_y=size,
                pixel_size=pixel
            )
            
            E = bs.gauss(cfg=cfg)
            E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            center = size // 2
            x = (np.arange(size) - center) * pixel * 1e3  # mm
            
            ax.plot(x, intensity_norm[center, :], linewidth=2.5,
                   label=f'f = {f*1e3:.0f} mm', color=colors[idx], alpha=0.85)
        
        ax.set_xlabel('X Position (mm)', fontsize=14)
        ax.set_ylabel('Normalized Intensity', fontsize=14)
        ax.set_title('Ideal Focal Spot Cross-Sections\n(Gaussian beam at various focal lengths)', fontsize=14)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ideal_focus_cross_sections.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_ideal_focus_theoretical_comparison(self, output_dir):
        """Compare simulated spot with theoretical Airy disk.
        模拟光斑与理论Airy斑对比
        """
        f = 500e-3  # 500mm focal length
        size = 512
        pixel = 8e-6
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel
        )
        
        # Simulate
        E = bs.gauss(cfg=cfg)
        E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        intensity_sim = np.abs(E_focal)**2
        intensity_sim_norm = intensity_sim / np.max(intensity_sim)
        
        # Calculate theoretical Airy disk
        airy_radius = bs.lens_focal_spot_size(f=f, cfg=cfg) * 1e3  # mm
        
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3  # mm
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 2D spot
        extent = size * pixel * 1e3 / 2
        ax = axes[0]
        im = ax.imshow(intensity_sim_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        # Draw Airy disk circles
        circle1 = plt.Circle((0, 0), airy_radius, fill=False, color='cyan', 
                             linewidth=2, linestyle='--', label='Airy radius')
        circle2 = plt.Circle((0, 0), 2*airy_radius, fill=False, color='cyan',
                             linewidth=2, linestyle=':', label='2× Airy radius')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.set_title(f'Simulated Focal Spot @ f={f*1e3:.0f}mm', fontsize=12)
        ax.legend(fontsize=10)
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Cross-section
        ax = axes[1]
        ax.plot(x, intensity_sim_norm[center, :], 'b-', linewidth=2.5, 
               label='Simulated', alpha=0.8)
        
        # Theoretical Airy pattern (simplified)
        r = np.abs(x)
        r_norm = r / (airy_radius + 1e-10)
        # Airy function approximation
        airy = np.zeros_like(r_norm)
        mask = r_norm > 0
        airy[mask] = (2 * np.abs(np.sinc(r_norm[mask])) )**2
        airy = airy / np.max(airy)  # Normalize
        
        ax.plot(x, airy, 'r--', linewidth=2, label='Theoretical Airy', alpha=0.8)
        ax.axvline(x=airy_radius, color='green', linestyle='--', alpha=0.5,
                  label=f'Airy radius = {airy_radius:.3f}mm')
        ax.axvline(x=2*airy_radius, color='orange', linestyle=':', alpha=0.5,
                  label=f'2× radius')
        
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)
        ax.set_title('Cross-Section Comparison', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5*airy_radius, 5*airy_radius)
        ax.set_ylim(0, 1.1)
        
        plt.suptitle(f'Ideal Focus: Simulation vs Theory @ f={f*1e3:.0f}mm\n'
                    f'λ={cfg.wavelength*1e9:.0f}nm, D={size*pixel*1e3:.1f}mm',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ideal_focus_theoretical.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_short_focal_length_10mm(self, output_dir):
        """Visualize ideal focus at 10mm focal length.
        10mm短焦距聚焦
        """
        f = 10e-3  # 10mm
        size = 256
        pixel = 3e-6  # High resolution for small scale
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=0.5e-3  # Match small focal length
        )
        
        E = bs.gauss(cfg=cfg)
        E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        
        intensity = np.abs(E_focal)**2
        intensity_norm = intensity / np.max(intensity)
        
        extent = size * pixel * 1e3 / 2  # mm
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 2D intensity
        ax = axes[0]
        im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title(f'Focal Spot @ f = {f*1e3:.0f} mm\n'
                    f'{size}×{size} @ {pixel*1e6:.0f}µm', fontsize=12)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Cross-section X
        ax = axes[1]
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3
        ax.plot(x, intensity_norm[center, :], 'b-', linewidth=2)
        ax.fill_between(x, intensity_norm[center, :], alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Horizontal Cross-Section')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-extent, extent)
        
        # 3D surface - need to create 3D projection axes
        ax3d = fig.add_subplot(133, projection='3d')
        step = 4
        X = x[::step]
        Y = x[::step]
        XX, YY = np.meshgrid(X, Y)
        Z = intensity_norm[::step, ::step]
        ax3d.plot_surface(XX, YY, Z, cmap='hot', alpha=0.9)
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Intensity')
        ax3d.set_title('3D Surface')
        
        plt.suptitle(f'Short Focal Length Focus: f = {f*1e3:.0f} mm\n'
                    f'λ = {cfg.wavelength*1e9:.0f} nm, w₀ = {cfg.w0*1e3:.2f} mm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ideal_focus_10mm.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_long_focal_length_1km(self, output_dir):
        """Visualize ideal focus at 1km focal length.
        1km长焦距聚焦
        """
        f = 1000e-3  # 1km = 1m
        size = 1024
        pixel = 20e-6  # Large pixels for wide field
        
        cfg = bs.Config(
            wavelength=810e-9,
            size_x=size, size_y=size,
            pixel_size=pixel,
            w0=50e-3  # Large beam waist for long focal
        )
        
        E = bs.gauss(cfg=cfg)
        E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
        
        intensity = np.abs(E_focal)**2
        intensity_norm = intensity / np.max(intensity)
        
        extent = size * pixel * 1e3 / 2  # mm
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 2D intensity
        ax = axes[0]
        im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                      extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
        ax.set_title(f'Focal Spot @ f = {f*1e3:.0f} mm (1km)\n'
                    f'{size}×{size} @ {pixel*1e6:.0f}µm', fontsize=12)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Cross-section
        ax = axes[1]
        center = size // 2
        x = (np.arange(size) - center) * pixel * 1e3
        ax.plot(x, intensity_norm[center, :], 'b-', linewidth=1.5)
        ax.fill_between(x, intensity_norm[center, :], alpha=0.3)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title('Horizontal Cross-Section')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
        
        # Zoomed view
        ax = axes[2]
        zoom_center = center
        zoom_range = 100  # pixels
        x_zoom = (np.arange(zoom_range*2) - zoom_range) * pixel * 1e3
        intensity_zoom = intensity_norm[center, center-zoom_range:center+zoom_range]
        ax.plot(x_zoom, intensity_zoom, 'r-', linewidth=2)
        ax.fill_between(x_zoom, intensity_zoom, alpha=0.3, color='red')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Intensity')
        ax.set_title('Zoomed Cross-Section (±1mm)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1, 1)
        
        plt.suptitle(f'Long Focal Length Focus: f = {f*1e3:.0f} mm (1km)\n'
                    f'λ = {cfg.wavelength*1e9:.0f} nm, w₀ = {cfg.w0*1e3:.0f} mm, D = {size*pixel*1e3:.0f} mm',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ideal_focus_1km.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_focal_length_comparison_grid(self, output_dir):
        """Create comprehensive comparison grid of focal spots.
        焦距对比综合网格
        """
        # Different focal lengths
        cases = [
            (10e-3, 256, 3e-6, 0.5e-3),   # 10mm
            (50e-3, 256, 5e-6, 2e-3),     # 50mm  
            (100e-3, 256, 8e-6, 5e-3),    # 100mm
            (500e-3, 512, 10e-6, 20e-3), # 500mm
            (1000e-3, 512, 15e-6, 40e-3), # 1000mm
            (10000e-3, 1024, 20e-6, 200e-3), # 10000mm (10m)
        ]
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 6, figure=fig, hspace=0.35, wspace=0.25)
        
        results = []
        
        for idx, (f, size, pixel, w0) in enumerate(cases):
            cfg = bs.Config(
                wavelength=810e-9,
                size_x=size, size_y=size,
                pixel_size=pixel,
                w0=w0
            )
            
            E = bs.gauss(cfg=cfg)
            E_focal = bs.lens_fft_propagation_to_focal(E, f=f, cfg=cfg)
            
            intensity = np.abs(E_focal)**2
            intensity_norm = intensity / np.max(intensity)
            
            # Calculate spot size (FWHM)
            center = size // 2
            profile = intensity_norm[center, :]
            fwhm = np.sum(profile > 0.5) * pixel * 1e3  # mm
            
            results.append((f, intensity_norm, size, pixel, fwhm))
            
            extent = size * pixel * 1e3 / 2
            
            # Image
            ax = fig.add_subplot(gs[0, idx])
            im = ax.imshow(intensity_norm, cmap='hot', origin='lower',
                          extent=[-extent, extent, -extent, extent], vmin=0, vmax=1)
            ax.set_title(f'f={f*1e3:.0f}mm\nFWHM={fwhm:.3f}mm', fontsize=9)
            ax.set_xlabel('X (mm)', fontsize=8)
            ax.set_ylabel('Y (mm)', fontsize=8)
        
        # Row 2: Log scale images
        for idx, (f, intensity, size, pixel, fwhm) in enumerate(results):
            ax = fig.add_subplot(gs[1, idx])
            intensity_log = np.log10(intensity + 1e-10)
            im = ax.imshow(intensity_log, cmap='viridis', origin='lower')
            ax.set_title(f'Log scale', fontsize=9)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
        
        # Row 3: Scaled cross-sections
        ax = fig.add_subplot(gs[2, :])
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(cases)))
        
        for idx, (f, intensity, size, pixel, fwhm) in enumerate(results):
            center = size // 2
            x = (np.arange(size) - center) * pixel * 1e3
            profile = intensity[center, :]
            
            # Normalize x to same scale
            if f < 100e-3:
                scale = 5  # mm
            elif f < 1000e-3:
                scale = 2  # mm
            else:
                scale = 0.5  # mm
            
            mask = np.abs(x) < scale
            x_scaled = x[mask]
            profile_scaled = profile[mask]
            
            ax.plot(x_scaled, profile_scaled, linewidth=2, color=colors[idx],
                   label=f'f={f*1e3:.0f}mm (FWHM={fwhm:.3f}mm)', alpha=0.85)
        
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)
        ax.set_title('Cross-Sections (scaled to same spatial extent)', fontsize=12)
        ax.legend(fontsize=9, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 1.1)
        
        plt.suptitle('Ideal Gaussian Beam Focal Spots\nComparison of Various Focal Lengths (10mm ~ 10m)',
                    fontsize=15, fontweight='bold')
        
        filepath = os.path.join(output_dir, 'ideal_focus_comprehensive.png')
        plt.savefig(filepath, dpi=180, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
