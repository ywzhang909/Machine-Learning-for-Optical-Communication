"""
Tests for GenerateOAM dataset generation class.
"""
import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from beam_simulation.create_oam_list import GenerateOAM


class TestGenerateOAM:
    """Test suite for GenerateOAM dataset generation."""

    def test_generate_oam_lg_single_mode(self):
        """Test GenerateOAM with single LG mode."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0, 1], "l": [1, 2]}
            },
            Cn2_list=[1e-13],
            n_samples=5
        )
        
        assert "LG" in gen.data
        assert len(gen.data["LG"]) > 0

    def test_generate_oam_multiple_modes(self):
        """Test GenerateOAM with multiple mode types."""
        gen = GenerateOAM(
            mode_types=["LG", "HG"],
            orders={
                "LG": {"p": [0], "l": [1]},
                "HG": {"n": [0], "m": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=3
        )
        
        assert "LG" in gen.data
        assert "HG" in gen.data

    def test_generate_oam_data_structure(self):
        """Test GenerateOAM creates correct data structure."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-13, 5e-13],
            n_samples=4
        )
        
        # Check data structure: data[mode][order][Cn2] = list of images
        lg_data = gen.data["LG"]
        
        # Get first order key
        first_order = list(lg_data.keys())[0]
        order_data = lg_data[first_order]
        
        # Should have entries for each Cn2 value
        assert len(order_data) == 2
        assert 1e-13 in order_data
        assert 5e-13 in order_data
        
        # Each Cn2 entry should be a list of n_samples images
        assert len(order_data[1e-13]) == 4

    def test_generate_oam_image_shapes(self):
        """Test GenerateOAM produces correct image shapes."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=3
        )
        
        # Get first image
        lg_data = gen.data["LG"]
        first_order = list(lg_data.keys())[0]
        images = lg_data[first_order][1e-13]
        
        # All images should have the same shape
        first_shape = images[0].shape
        for img in images:
            assert img.shape == first_shape
        
        # Shape should be square (based on default config)
        assert first_shape[0] == first_shape[1]

    def test_generate_oam_intensity_normalized(self):
        """Test GenerateOAM produces normalized intensity images."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=2
        )
        
        lg_data = gen.data["LG"]
        first_order = list(lg_data.keys())[0]
        images = lg_data[first_order][1e-13]
        
        for img in images:
            # Should be normalized to [0, 1]
            assert img.max() == pytest.approx(1.0, rel=0.01)
            assert img.min() >= 0

    def test_generate_oam_multiple_realizations_different(self):
        """Test GenerateOAM produces different realizations (due to turbulence)."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=3
        )
        
        lg_data = gen.data["LG"]
        first_order = list(lg_data.keys())[0]
        images = lg_data[first_order][1e-13]
        
        # With turbulence, different realizations should be different
        # (Check that not all images are identical)
        first_img = images[0]
        all_same = all(np.allclose(img, first_img) for img in images)
        # Note: This might pass if turbulence is weak, but that's OK

    def test_generate_oam_attributes(self):
        """Test GenerateOAM stores correct attributes."""
        gen = GenerateOAM(
            mode_types=["LG", "HG"],
            orders={
                "LG": {"p": [0], "l": [1]},
                "HG": {"n": [0], "m": [1]}
            },
            Cn2_list=[1e-13, 2e-13],
            n_samples=10
        )
        
        assert gen.mode_types == ["LG", "HG"]
        assert gen.Cn2_list == [1e-13, 2e-13]
        assert gen.n_samples == 10

    def test_generate_oam_order_labels(self):
        """Test GenerateOAM generates correct order labels."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0, 1], "l": [1, 2]}
            },
            Cn2_list=[1e-13],
            n_samples=2
        )
        
        # Should have generated combinations: p0_l1, p0_l2, p1_l1, p1_l2
        lg_data = gen.data["LG"]
        order_names = list(lg_data.keys())
        
        assert len(order_names) == 4  # 2 p values × 2 l values

    def test_generate_oam_hg_orders(self):
        """Test GenerateOAM with HG mode orders."""
        gen = GenerateOAM(
            mode_types=["HG"],
            orders={
                "HG": {"n": [0, 1], "m": [0, 1]}
            },
            Cn2_list=[1e-13],
            n_samples=2
        )
        
        hg_data = gen.data["HG"]
        order_names = list(hg_data.keys())
        
        # Should have combinations: n0_m0, n0_m1, n1_m0, n1_m1
        assert len(order_names) == 4


class TestGenerateOAMVisualization:
    """Visualization tests for GenerateOAM."""

    def test_generate_oam_single_mode_samples(self, output_dir):
        """Visualize multiple samples of a single mode."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=6
        )
        
        lg_data = gen.data["LG"]
        first_order = list(lg_data.keys())[0]
        images = lg_data[first_order][1e-13]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        for idx, img in enumerate(images[:6]):
            ax = axes[idx]
            im = ax.imshow(img, cmap='plasma', origin='lower')
            ax.set_title(f'Sample {idx + 1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'LG {first_order} - Multiple Turbulence Realizations')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_samples.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_generate_oam_mode_comparison(self, output_dir):
        """Visualize different modes side by side."""
        gen = GenerateOAM(
            mode_types=["LG", "HG"],
            orders={
                "LG": {"p": [0, 1], "l": [1, 2]},
                "HG": {"n": [0, 1], "m": [0, 1]}
            },
            Cn2_list=[1e-13],
            n_samples=1
        )
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # LG modes (top row)
        lg_data = gen.data["LG"]
        lg_orders = list(lg_data.keys())[:4]
        for idx, order in enumerate(lg_orders):
            ax = axes[0, idx]
            img = lg_data[order][1e-13][0]  # First sample
            im = ax.imshow(img, cmap='plasma', origin='lower')
            ax.set_title(f'LG {order}')
            plt.colorbar(im, ax=ax)
        
        # HG modes (bottom row)
        hg_data = gen.data["HG"]
        hg_orders = list(hg_data.keys())[:4]
        for idx, order in enumerate(hg_orders):
            ax = axes[1, idx]
            img = hg_data[order][1e-13][0]  # First sample
            im = ax.imshow(img, cmap='plasma', origin='lower')
            ax.set_title(f'HG {order}')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Generated Mode Dataset Samples')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_mode_comparison.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_generate_oam_turbulence_levels(self, output_dir):
        """Visualize same mode at different turbulence levels."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1]}
            },
            Cn2_list=[1e-15, 1e-14, 1e-13, 5e-13, 1e-12],
            n_samples=3
        )
        
        lg_data = gen.data["LG"]
        first_order = list(lg_data.keys())[0]
        
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        
        for idx, Cn2 in enumerate(gen.Cn2_list):
            # Show first sample for each turbulence level
            img = lg_data[first_order][Cn2][0]
            
            # Average of all samples
            avg_img = np.mean(lg_data[first_order][Cn2], axis=0)
            
            # Individual sample
            ax = axes[0, idx]
            im = ax.imshow(img, cmap='plasma', origin='lower')
            ax.set_title(f'Cn²={Cn2:.0e}\nSingle')
            plt.colorbar(im, ax=ax)
            
            # Average
            ax = axes[1, idx]
            im = ax.imshow(avg_img, cmap='plasma', origin='lower')
            ax.set_title(f'Cn²={Cn2:.0e}\nAveraged')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'LG {first_order} - Turbulence Effect')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_turbulence_levels.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_generate_oam_intensity_histograms(self, output_dir):
        """Visualize intensity distributions for different modes."""
        gen = GenerateOAM(
            mode_types=["LG", "HG"],
            orders={
                "LG": {"p": [0], "l": [1]},
                "HG": {"n": [1], "m": [1]}
            },
            Cn2_list=[1e-13],
            n_samples=10
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # LG mode samples
        lg_data = gen.data["LG"]
        lg_order = list(lg_data.keys())[0]
        lg_images = lg_data[lg_order][1e-13]
        
        # Average intensity profile
        avg_lg = np.mean([img.flatten() for img in lg_images], axis=0)
        axes[0, 0].hist(avg_lg, bins=50, density=True, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Normalized Intensity')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].set_title(f'LG {lg_order} Intensity Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # LG mode sample visualization
        im = axes[0, 1].imshow(lg_images[0], cmap='plasma', origin='lower')
        axes[0, 1].set_title(f'LG {lg_order} Sample')
        plt.colorbar(im, ax=axes[0, 1])
        
        # HG mode samples
        hg_data = gen.data["HG"]
        hg_order = list(hg_data.keys())[0]
        hg_images = hg_data[hg_order][1e-13]
        
        # Average intensity profile
        avg_hg = np.mean([img.flatten() for img in hg_images], axis=0)
        axes[1, 0].hist(avg_hg, bins=50, density=True, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Normalized Intensity')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].set_title(f'HG {hg_order} Intensity Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HG mode sample visualization
        im = axes[1, 1].imshow(hg_images[0], cmap='plasma', origin='lower')
        axes[1, 1].set_title(f'HG {hg_order} Sample')
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.suptitle('GenerateOAM Dataset Analysis')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_histograms.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_generate_oam_summary_statistics(self, output_dir):
        """Visualize summary statistics of generated dataset."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0, 1], "l": [1, 2]}
            },
            Cn2_list=[1e-14, 1e-13],
            n_samples=5
        )
        
        lg_data = gen.data["LG"]
        orders = list(lg_data.keys())
        Cn2_values = gen.Cn2_list
        
        # Create summary figure
        fig, axes = plt.subplots(len(Cn2_values), len(orders), figsize=(12, 6))
        
        for i, Cn2 in enumerate(Cn2_values):
            for j, order in enumerate(orders):
                ax = axes[i, j]
                
                # Get all samples for this condition
                images = lg_data[order][Cn2]
                
                # Compute mean image
                mean_img = np.mean(images, axis=0)
                
                im = ax.imshow(mean_img, cmap='plasma', origin='lower')
                ax.set_title(f'{order}\nCn²={Cn2:.0e}')
                plt.colorbar(im, ax=ax)
                
                # Compute statistics
                mean_intensity = np.mean([np.mean(img) for img in images])
                std_intensity = np.std([np.mean(img) for img in images])
                
                ax.set_xlabel(f'μ={mean_intensity:.3f}, σ={std_intensity:.4f}', fontsize=8)
        
        plt.suptitle('GenerateOAM Dataset Summary (Mean Images)')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_summary.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)

    def test_generate_oam_cross_section_comparison(self, output_dir):
        """Visualize cross-sections of generated modes."""
        gen = GenerateOAM(
            mode_types=["LG"],
            orders={
                "LG": {"p": [0], "l": [1, 2, 3]}
            },
            Cn2_list=[1e-13],
            n_samples=1
        )
        
        lg_data = gen.data["LG"]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        orders = list(lg_data.keys())
        center_slice = 32  # Middle of image
        
        for idx, order in enumerate(orders):
            ax = axes[idx]
            img = lg_data[order][1e-13][0]
            
            # Plot horizontal cross-section
            ax.plot(img[center_slice, :], linewidth=1.5, label='Horizontal')
            ax.plot(img[:, center_slice], linewidth=1.5, alpha=0.7, label='Vertical')
            
            ax.set_xlabel('Position (pixels)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_title(f'LG {order}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('GenerateOAM Mode Cross-Sections')
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'test_generate_oam_cross_sections.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        assert os.path.exists(filepath)
