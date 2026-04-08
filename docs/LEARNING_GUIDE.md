# 📚 Optiland Learning Guide

Welcome to the Optiland Learning Guide! This guide walks you through key concepts, tutorials, and workflows for using Optiland effectively—from beginner basics to advanced optical modeling and ML integration.


1. **Introduction to Optiland**
    - [Tutorial 1a - Optiland for Beginners](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1a_Optiland_for_Beginners.ipynb)
         - Lens entry
         - Material selection
         - Aperture, field and wavelength selection
         - Drawing a lens in 2D and 3D
    - [Tutorial 1b - Determining Lens Properties](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1b_Lens_Properties.ipynb)
        - Focal length
        - Magnification
        - F-Number, Entrance/Exit pupil sizes & positions
        - Focal, Principal, and Nodal points, etc.
    - [Tutorial 1c - Saving and Loading Lenses](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1c_Save_and_Load_Files.ipynb)
        - Saving and loading lens files in a json format
    - [Tutorial 1d - Using the Material Database](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1d_Material_Database.ipynb)
        - Defining materials for glass, chemicals, organics, gases, or using ideal or parameterized materials.
    - [Tutorial 1e - Non-rotationally Symmetric Systems](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1e_Non_Rotationally_Symmetric_Systems.ipynb)
        - Coordinate systems in Optiland and how to design non-rotationally symmetric systems
    - [Tutorial 1f - Differentiable Raytracing Hello World](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_1f_Differentiable_Ray_Tracing_Hello_World.ipynb)
        - Your first steps with DRT using the PyTorch backend.
2. **Real Raytracing & Analysis**
    - [Tutorial 2a - Tracing & Analyzing Rays](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_2a_Tracing_&_Analyzing_Rays.ipynb)
        - How to trace rays through a system
        - Analyzing ray paths & properties
    - [Tutorial 2b - Tilting & De-centering Components](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_2b_Tilting_&_Decentering_Components.ipynb)
        - Tracing rays through misaligned components
    - [Tutorial 2c - Monte Carlo Raytracing Methods](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_2c_Monte_Carlo_Raytracing.ipynb)
        - How variations in lens properties impact lens performance
    - [Tutorial 2d - Aspheric Components](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_2d_Raytracing_Aspheres.ipynb)
        - Modeling even aspheres
3. **Aberrations**
    - [Tutorial 3a - Common aberration analyses/plots](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_3a_Common_Aberration_Analyses.ipynb)
        - Spot diagrams
        - Ray fans
        - Y-Ybar plots
        - Distortion / Grid distortion plots
        - Field curvature plots
        - RMS spot size & wavefront error vs. field
        - Pupil aberration
    - [Tutorial 3b - 1st & 3rd-Order Aberrations](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_3b_First_&_Third_Order_Aberrations.ipynb)
        - Calculation of seidel, 1st & 3rd-order aberrations
    - [Tutorial 3c - Chromatic Aberrations](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_3c_Chromatic_Aberrations.ipynb)
        - Achromatic doublet to reduce chromatic aberrations
4. **Optical Path Difference (OPD), Point Spread Functions (PSF) & Modulation Transfer Function (MTF)**
    - [Tutorial 4a - Optical Path Difference](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_4a_Optical_Path_Difference_Calculation.ipynb)
        - OPD fans and plots
    - [Tutorial 4b - PSF & MTF Calculation](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_4b_PSF_&_MTF_Calculation.ipynb)
        - Geometric MTF
        - FFT-based PSF/MTF
        - Huygens PSF
    - [Tutorial 4c - Zernike Decomposition](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_4c_Zernike_Decomposition.ipynb)
        - Decomposing wavefront using Zernike polynomials
        - Coefficient types: Zernike standard, Zernike fringe, Zernike Noll
5. **Optimization**
    - [Tutorial 5a - Simple Optimization](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_5a_Simple_Optimization.ipynb)
        - Operand and variable definition
        - Local optimization
    - [Tutorial 5b - Advanced Optimization](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_5b_Advanced_Optimization.ipynb)
        - Global optimization
    - [Tutorial 5c - Optimization Case Study](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_5c_Optimization_Case_Study.ipynb)
        - Complete process of designing a Cooke triplet
    - [Tutorial 5d - User-defined Optimization Metrics](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_5d_User_Defined_Optimization.ipynb)
        - Customized optimization
6. **Coatings & Polarization**
    - [Tutorial 6a - Introduction to Coatings](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6a_Introduction_to_Coatings.ipynb)
        - Simple coatings in Optiland
        - Impact of coatings on system performance
    - [Tutorial 6b - Introduction to Polarization](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6b_Introduction_to_Polarization.ipynb)
        - Basics of polarization in Optiland
        - Analyzing polarization performance using exact matrix formalisms
        - Modeling polarizer and retarder coatings
    - [Tutorial 6c - Multilayer Stack](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6c_Multilayer_Stack.ipynb)
        - Transfer matrix method for thin films
    - [Tutorial 6d - Thin Film Optimization](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6d_Thin_Film_Optimization.ipynb)
        - Optimizing thin film layer thicknesses
    - [Tutorial 6e - Color Analysis for Thin Films](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6e_Color_Analysis_For_Thin_Film.ipynb)
        - Colorimetric performance of coatings
    - [Tutorial 6f - AR Coating System](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6f_AR_Coating_System.ipynb)
        - Designing an anti-reflective coating
    - [Tutorial 6h - Needle Synthesis](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6h_Needle_Synthesis.ipynb)
        - Automated layer insertion and iterative optimization
    - [Tutorial 6i - Thin Film Tolerance Analysis](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_6i_Thin_Film_Tolerance_Analysis.ipynb)
        - Sensitivity and Monte Carlo simulations for coatings
7. **Advanced Optical Design**
    - [Tutorial 7a - Lithographic Projection System](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_7a_Lithographic_Projection_System.ipynb)
        - Optimizing and Analyzing a Complex Lithography System
    - [Tutorial 7b - Surface Roughness & Scattering](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_7b_Surface_Roughness_&_Scattering.ipynb)
        - Lambertian and Gaussian scatter models
    - [Tutorial 7c - Freeform Surfaces](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_7c_Freeform_Surfaces.ipynb)
        - Designing non-standard optical systems with freeform surfaces
    - [Tutorial 7d - Three Mirror Anastigmat](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_7d_Three_Mirror_Anastigmat.ipynb)
        - Off-axis reflective telescope with freeform surfaces
8. **Tolerancing**
    - [Tutorial 8a - Introduction to Tolerancing](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_8a_Tolerancing_Sensitivity_Analysis.ipynb)
        - Sensitivity studies
    - [Tutorial 8b - Advanced Tolerancing](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_8b_Monte_Carlo_Analysis.ipynb)
        - Monte Carlo-based Tolerancing
9. **Lens Catalogue Integration**
    - [Tutorial 9a - Edmund Optics Catalogue](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_9a_Edmund_Optics_Catalogue.ipynb)
        - Reading Zemax, CODE V, and OSLO files
        - Reading and analyzing an aspheric lens
    - [Tutorial 9b - Thorlabs Catalogue](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_9b_Thorlabs_Catalogue.ipynb)
        - Reading and analyzing an achromatic doublet pair lens
10. **Extending Optiland**
    - [Tutorial 10a - Custom Surface Types](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_10a_Custom_Surface_Types.ipynb)
        - Adding new surface types
    - [Tutorial 10b - Custom Coating Types](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_10b_Custom_Coating_Types.ipynb)
        - Adding new coating types
    - [Tutorial 10c - Custom Optimization Algorithms](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_10c_Custom_Optimization_Algorithm.ipynb)
        - Creating a "random walk optimizer" to optimize an aspheric singlet
11. **Extended Sources**
    - [Tutorial 11a - Extended Source Modeling](https://github.com/optiland/optiland/blob/master/docs/examples/Tutorial_11a_Extended_Source_Modeling.ipynb)
        - Modeling extended sources
12. **Machine Learning in Optical Design** - note that these notebooks are hosted in the [LensAI repository](https://github.com/HarrisonKramer/LensAI)
    - [Tutorial 12a - Random Forest Regressor to Predict Optimal Lens Properties](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_1/Singlet_RF_Model_RMS_Spot_Size.ipynb)
        - Demonstrates how to build and train a random forest regressor to predict the radius of curvature of a plano-convex lens in order to minimize the RMS spot size.
    - [Tutorial 12b - Ray Path Failure Classification Model](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_2/Ray_Path_Failure_Classification_Model.ipynb)
        - Uses logistic regression to predict ray path failures in a Cooke triplet design.
    - [Tutorial 12c - Surrogate Ray Tracing Model Using Neural Networks](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_3/Double_Gauss_Surrogate_Model.ipynb)
        - Builds a neural network surrogate ray tracing model to increase effective "ray tracing" speed by 10,000x.
    - [Tutorial 12d - Super-Resolution Generative Adversarial Network to Enhance Wavefront Map Data](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_5/SR_GAN_for_wavefront_data.ipynb)
        - Utilizes a super-resolution GAN (SRGAN) to upscale low-resolution wavefront data into high-resolution data.
    - [Tutorial 12e - Optimization of Aspheric Lenses via Reinforcement Learning](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_4/RL_aspheric_singlet.ipynb)
        - Reinforcement learning is applied to the optimization of aspheric singlet lenses to generate new lens designs.
