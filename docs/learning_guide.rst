This guide is Optiland's primary learning path. Tutorials are grouped thematically and follow a
progressive arc where concepts from earlier sections inform later ones. Each notebook is
self-contained and runnable; you do not need to execute prior notebooks to run any given one.
New to Optiland? Start with :ref:`start_here` to find the path that fits your goals.


1. Introduction to Optiland
---------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_1a_Optiland_for_Beginners
   examples/Tutorial_1b_Lens_Properties
   examples/Tutorial_1c_Save_and_Load_Files
   examples/Tutorial_1d_Material_Database
   examples/Tutorial_1e_Non_Rotationally_Symmetric_Systems
   examples/Tutorial_1f_Differentiable_Ray_Tracing_Hello_World
   examples/Tutorial_1g_Material_Catalog_and_Registry
   examples/Tutorial_1h_Prescription_Generator


2. Real Raytracing & Analysis
------------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_2a_Tracing_&_Analyzing_Rays
   examples/Tutorial_2b_Tilting_&_Decentering_Components
   examples/Tutorial_2c_Monte_Carlo_Raytracing
   examples/Tutorial_2d_Raytracing_Aspheres


3. Aberrations
--------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_3a_Common_Aberration_Analyses
   examples/Tutorial_3b_First_&_Third_Order_Aberrations
   examples/Tutorial_3c_Chromatic_Aberrations


4. OPD, PSF, & MTF
------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_4a_Optical_Path_Difference_Calculation
   examples/Tutorial_4b_PSF_&_MTF_Calculation
   examples/Tutorial_4c_Zernike_Decomposition


5. Optimization
---------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_5a_Simple_Optimization
   examples/Tutorial_5b_Advanced_Optimization
   examples/Tutorial_5c_Optimization_Case_Study
   examples/Tutorial_5d_User_Defined_Optimization
   examples/Tutorial_5e_Differentiable_Optimization


6. Coatings & Polarization
--------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_6a_Introduction_to_Coatings
   examples/Tutorial_6b_Introduction_to_Polarization
   examples/Tutorial_6c_Multilayer_Stack
   examples/Tutorial_6d_Thin_Film_Optimization
   examples/Tutorial_6e_Color_Analysis_For_Thin_Film
   examples/Tutorial_6f_AR_Coating_System
   examples/Tutorial_6h_Needle_Synthesis
   examples/Tutorial_6i_Thin_Film_Tolerance_Analysis

7. Advanced Optical Design
--------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_7a_Lithographic_Projection_System
   examples/Tutorial_7b_Surface_Roughness_&_Scattering
   examples/Tutorial_7c_Freeform_Surfaces
   examples/Tutorial_7d_Three_Mirror_Anastigmat
   examples/Tutorial_7e_Glass_Expert
   examples/Tutorial_7f_Multi_Configuration_Zoom_Lenses


8. Tolerancing
--------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_8a_Tolerancing_Sensitivity_Analysis
   examples/Tutorial_8b_Monte_Carlo_Analysis


9. Lens Catalogue Integration
-----------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_9a_Edmund_Optics_Catalogue
   examples/Tutorial_9b_Thorlabs_Catalogue

10. Extending Optiland
----------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_10a_Custom_Surface_Types
   examples/Tutorial_10b_Custom_Coating_Types
   examples/Tutorial_10c_Custom_Optimization_Algorithm

11. Extended Source Modeling
---------------------------

.. toctree::
   :maxdepth: 1

   examples/Tutorial_11a_Extended_Source_Modeling

12. Machine Learning in Optical Design
--------------------------------------

.. note::
   The following tutorials are hosted in the `LensAI repository <https://github.com/HarrisonKramer/LensAI>`_.
   For in-repo differentiable modeling examples, see :ref:`Tutorial_1f
   <examples/Tutorial_1f_Differentiable_Ray_Tracing_Hello_World>` and
   :ref:`Tutorial_5e <examples/Tutorial_5e_Differentiable_Optimization>`.

These examples demonstrate how Optiland can be used in conjunction with machine learning to solve optical design problems.
Concepts in machine and deep learning are also covered.

.. toctree::
   :maxdepth: 1

   examples/Singlet_RF_Model_RMS_Spot_Size
   examples/Ray_Path_Failure_Classification_Model
   examples/Double_Gauss_Surrogate_Model
   examples/SR_GAN_for_wavefront_data
   examples/RL_aspheric_singlet
   examples/Misalignment_Prediction_Cooke_Triplet
