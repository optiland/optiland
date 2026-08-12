.. _how_do_i:

How Do I …?
===========

A task-shaped index. The :ref:`example gallery <example_gallery>` is organized by
Optiland feature; this page is organized by **what you are trying to do**. Find your
question, follow the link, copy the pattern.

New to Optiland? Read :doc:`conventions` first — it explains the surface-sequential
model, the "after" rule for ``thickness`` and ``material``, sign conventions, and units.
Most "why doesn't this work?" questions are answered there.

.. tip::

   Stuck on an error rather than a task? Call
   ``optiland.diagnostics.check_system(lens)``. It reports missing wavelengths,
   apertures, stops, and fields, each with a runnable fix. See
   :doc:`gallery/miscellaneous/diagnostics_demo`.

----

Build a system
--------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - build my first lens from scratch?
     - :doc:`Tutorial 1a — Optiland for Beginners <examples/Tutorial_1a_Optiland_for_Beginners>`
   * - see a whole design carried from requirements to prescription?
     - :doc:`Tutorial 1e — Design a Doublet End to End <examples/Tutorial_1e_Design_a_Doublet_End_to_End>`
   * - build a simple singlet?
     - :doc:`gallery/basic_lenses/singlet`
   * - build a cemented doublet?
     - :doc:`gallery/basic_lenses/doublet`
   * - build a Cooke triplet?
     - :doc:`gallery/basic_lenses/Cooke_Triplet`
   * - build a Petzval, Tessar, Heliar, or telephoto lens?
     - :doc:`gallery/basic_lenses/petzval`, :doc:`gallery/basic_lenses/tessar`,
       :doc:`gallery/basic_lenses/heliar`, :doc:`gallery/basic_lenses/telephoto`
   * - build a double Gauss objective?
     - :doc:`gallery/specialized_lenses/double_Gauss`
   * - add an aspheric surface?
     - :doc:`gallery/basic_lenses/aspheric_singlet`,
       :doc:`gallery/specialized_lenses/odd_asphere`
   * - define a mirror?
     - :doc:`gallery/reflective/parabola`
   * - build a two-mirror telescope?
     - :doc:`gallery/reflective/Cassegrain`, :doc:`gallery/reflective/hubble`
   * - build a three-mirror anastigmat?
     - :doc:`Tutorial 4f <examples/Tutorial_4f_Three_Mirror_Anastigmat>`,
       :doc:`gallery/reflective/three_mirror_anastigmat`
   * - use an off-axis parabola?
     - :doc:`gallery/reflective/off_axis_parabola`
   * - build a cylindrical (anamorphic) lens?
     - :doc:`gallery/basic_lenses/cylindrical_lens`
   * - lay out a system paraxially before choosing glass?
     - :doc:`gallery/basic_lenses/first_order_layout`
   * - turn an ideal thin lens into a real thick lens?
     - :doc:`gallery/miscellaneous/paraxial_to_thick_lens`
   * - build a zoom or multi-configuration system?
     - :doc:`Tutorial 4c <examples/Tutorial_4c_Zoom_Lenses_and_Multi_Configuration>`,
       :doc:`gallery/specialized_lenses/zoom_lens`
   * - use off-the-shelf catalog lenses?
     - :doc:`Tutorial 4d <examples/Tutorial_4d_Lens_Catalogue_Integration>`,
       :doc:`gallery/real_world_projects/Cooke_Triplet_with_Stock_Lenses`
   * - combine two existing lenses into one system?
     - :doc:`gallery/miscellaneous/combining_lenses`
   * - tilt or decenter a surface?
     - :doc:`Tutorial 4a <examples/Tutorial_4a_Tilts_Decenters_and_Asymmetric_Systems>`
   * - add a fold mirror?
     - :doc:`gallery/specialized_lenses/f_theta_with_fold_mirror`
   * - build a freeform surface (Chebyshev, polynomial, Forbes, NURBS)?
     - :doc:`gallery/freeform/chebyshev`, :doc:`gallery/freeform/polynomial`,
       :doc:`gallery/freeform/forbes_surface`,
       :doc:`gallery/freeform/nurbs_parabolic_mirror`
   * - model a diffraction grating?
     - :doc:`gallery/phase/linear_grating`, :doc:`gallery/phase/radial_grating`
   * - design a spectrometer?
     - :doc:`gallery/diffractive/Czerny_Turner_Spectrometer`,
       :doc:`gallery/diffractive/Littrow_Spectrometer`,
       :doc:`gallery/diffractive/Dyson_Spectrometer`
   * - model the human eye?
     - :doc:`gallery/miscellaneous/eye`
   * - pick a material, or use a specific glass catalog?
     - :doc:`Tutorial 1c <examples/Tutorial_1c_Material_Database_and_Catalogs>`
   * - design an infrared or lithographic system?
     - :doc:`gallery/specialized_lenses/infrared`,
       :doc:`Tutorial 4e <examples/Tutorial_4e_Lithographic_Projection_System>`

----

Constrain a system
------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - set the aperture stop?
     - :ref:`conventions — stop, aperture, and pupil <conventions_stop_aperture_pupil>`
   * - choose between EPD, f/#, and object-space NA?
     - :doc:`gallery/miscellaneous/aperture_demo`
   * - let the aperture float with the stop diameter?
     - :doc:`gallery/miscellaneous/float_by_stop_size`
   * - set my own semi-diameters instead of the computed ones?
     - :doc:`gallery/miscellaneous/custom_aperture_sizes`
   * - add a physical (clear) aperture to a surface?
     - :doc:`gallery/miscellaneous/zemax_circular_aperture_demo`
   * - apodize the pupil?
     - :doc:`gallery/miscellaneous/apodization`
   * - specify fields as angles vs. object height?
     - :ref:`conventions — field specification <conventions_fields>`
   * - specify fields as paraxial or real image height?
     - :doc:`gallery/miscellaneous/paraxial_image_height_field`,
       :doc:`gallery/miscellaneous/real_image_height_field`
   * - make a system telecentric?
     - :doc:`gallery/specialized_lenses/telecentric_lens`
   * - fix rays that miss the stop in a wide-angle system?
     - :doc:`gallery/miscellaneous/ray_aiming`
   * - tie one surface parameter to another (pickups and solves)?
     - :doc:`gallery/optimization/pickups`

----

Analyze a system
----------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - get EFL, BFL, f/#, and the pupil positions?
     - :doc:`Tutorial 1b <examples/Tutorial_1b_Lens_Properties_and_Prescription>`
   * - print a surface-by-surface prescription?
     - :doc:`gallery/analysis/prescription`
   * - plot a spot diagram?
     - :doc:`gallery/analysis/spot`
   * - plot RMS spot size vs. field?
     - :doc:`gallery/analysis/rms_spot_size_vs_field`
   * - see how the spot changes through focus?
     - :doc:`gallery/analysis/through_focus_spot_diagram`
   * - plot transverse ray-aberration (ray fan) curves?
     - :doc:`gallery/analysis/ray_fan`, :doc:`gallery/analysis/ray_fan_best_fit`
   * - plot field curvature and distortion?
     - :doc:`gallery/analysis/field_curvature`, :doc:`gallery/analysis/distortion`
   * - plot a grid-distortion map?
     - :doc:`gallery/analysis/grid_distortion`
   * - compute encircled energy?
     - :doc:`gallery/analysis/encircled_energy`
   * - plot the wavefront error (OPD fan or OPD map)?
     - :doc:`gallery/wavefront/opd_fan`, :doc:`gallery/wavefront/opd_map`
   * - plot RMS wavefront error vs. field?
     - :doc:`gallery/analysis/rms_wavefront_error_vs_field`
   * - decompose the wavefront into Zernike coefficients?
     - :doc:`gallery/wavefront/zernike_decomposition`
   * - compute the PSF?
     - :doc:`gallery/wavefront/fft_psf_2d`, :doc:`gallery/wavefront/huygens_psf_2d`
   * - compute the MTF at a given spatial frequency?
     - :doc:`gallery/wavefront/mtf_fft`, :doc:`gallery/wavefront/mtf_geometric`,
       :doc:`gallery/wavefront/mtf_huygens`
   * - plot MTF vs. field, or through focus?
     - :doc:`gallery/wavefront/mtf_vs_field`,
       :doc:`gallery/wavefront/through_focus_mtf`
   * - compute Seidel and chromatic aberration coefficients?
     - :doc:`Tutorial 2c <examples/Tutorial_2c_Aberration_Analyses>`
   * - trace rays myself and inspect the intersection coordinates?
     - :doc:`Tutorial 2a <examples/Tutorial_2a_Tracing_and_Analyzing_Rays>`
   * - run a Monte Carlo (random-ray) trace?
     - :doc:`Tutorial 2b <examples/Tutorial_2b_Monte_Carlo_Raytracing>`
   * - draw a y-ybar diagram?
     - :doc:`gallery/analysis/y_ybar`
   * - simulate the image of a scene through my lens?
     - :doc:`gallery/analysis/image_simulation`
   * - compute irradiance on a surface?
     - :doc:`gallery/analysis/irradiance`
   * - inspect pupil aberration?
     - :doc:`gallery/analysis/pupil_aberration`
   * - inspect surface sag?
     - :doc:`gallery/analysis/sag_surface_analysis`
   * - find out why my system will not trace?
     - :doc:`gallery/miscellaneous/diagnostics_demo`

----

Optimize a system
-----------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - vary a radius and minimize spot size?
     - :doc:`Tutorial 3a <examples/Tutorial_3a_Simple_Optimization>`,
       :doc:`gallery/optimization/rms_spot_size`
   * - optimize against wavefront error instead of spot size?
     - :doc:`gallery/optimization/wavefront_error`
   * - hold total track, edge thickness, or another constraint?
     - :doc:`gallery/optimization/constrained`,
       :doc:`gallery/optimization/bounded_operands`
   * - escape a local minimum (global optimization)?
     - :doc:`gallery/optimization/global`,
       :doc:`gallery/optimization/basin_hopping`,
       :doc:`gallery/optimization/shgo`,
       :doc:`gallery/optimization/particle_swarm_optimization`
   * - optimize aspheric coefficients?
     - :doc:`gallery/optimization/asphere`
   * - optimize a freeform surface?
     - :doc:`gallery/optimization/freeform`
   * - let the optimizer choose the glasses?
     - :doc:`Tutorial 3e <examples/Tutorial_3e_Glass_Expert_Categorical_Optimization>`,
       :doc:`gallery/optimization/glass_expert_example`
   * - add my own operand (merit function term)?
     - :doc:`Tutorial 3c <examples/Tutorial_3c_User_Defined_Optimization>`
   * - scale variables so the optimizer behaves?
     - :doc:`gallery/optimization/custom_scaler`,
       :doc:`gallery/optimization/reciprocal_radii_optimization`
   * - undo an optimization that made things worse?
     - :doc:`gallery/optimization/undo`
   * - work through a full optimization case study?
     - :doc:`Tutorial 3d <examples/Tutorial_3d_Optimization_Case_Study_Cooke_Triplet>`
   * - design a beam expander?
     - :doc:`gallery/optimization/beam_expander`
   * - use orthogonal descent?
     - :doc:`gallery/optimization/orthogonal_descent`

----

Tolerance a system
------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - find which parameters my performance is most sensitive to?
     - :doc:`Tutorial 6a <examples/Tutorial_6a_Tolerancing_Sensitivity_Analysis>`,
       :doc:`gallery/tolerancing/sensitivity`
   * - estimate manufacturing yield with Monte Carlo?
     - :doc:`Tutorial 6b <examples/Tutorial_6b_Monte_Carlo_Tolerancing_Analysis>`,
       :doc:`gallery/tolerancing/monte_carlo`
   * - add a compensator (e.g. refocus) to the analysis?
     - :doc:`gallery/tolerancing/compensators`
   * - include surface roughness and scattering?
     - :doc:`Tutorial 6c <examples/Tutorial_6c_Roughness_Scattering_and_Extended_Sources>`

----

Interoperate with other tools
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - import a Zemax ``.zmx`` file?
     - :doc:`Tutorial 1d <examples/Tutorial_1d_Saving_and_Loading>`
   * - import a CODE V ``.seq`` file?
     - :doc:`gallery/miscellaneous/codev_import_demo`
   * - import an OSLO ``.len`` file?
     - :doc:`gallery/miscellaneous/oslo_import_demo`
   * - save and reload a system as JSON?
     - :doc:`Tutorial 1d <examples/Tutorial_1d_Saving_and_Loading>`
   * - export a prescription for a report?
     - :doc:`gallery/analysis/prescription`

----

Coatings and polarization
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - apply an anti-reflection or multilayer coating?
     - :doc:`Tutorial 5a <examples/Tutorial_5a_Coatings_and_Multilayer_Stacks>`
   * - trace polarized light?
     - :doc:`Tutorial 5b <examples/Tutorial_5b_Introduction_to_Polarization>`
   * - inspect the Jones pupil?
     - :doc:`gallery/analysis/jones_pupil`
   * - optimize a thin-film stack?
     - :doc:`Tutorial 5c <examples/Tutorial_5c_Thin_Film_Optimization_and_Needle_Synthesis>`,
       :doc:`Tutorial 5d <examples/Tutorial_5d_Advanced_Thin_Film_Applications>`

----

Differentiate and accelerate
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - switch to the PyTorch backend (and to the GPU)?
     - :doc:`Tutorial 7a <examples/Tutorial_7a_Differentiable_Ray_Tracing_Hello_World>`,
       :doc:`gallery/differentiable_ray_tracing/basic_pytorch_backend`
   * - get gradients of a performance metric w.r.t. lens parameters?
     - :doc:`gallery/differentiable_ray_tracing/gradient_calculation`
   * - optimize a lens with autograd?
     - :doc:`Tutorial 7b <examples/Tutorial_7b_Differentiable_Lens_Optimization>`,
       :doc:`gallery/differentiable_ray_tracing/simple_optimization`
   * - put a lens inside a PyTorch training loop?
     - :doc:`gallery/optimization/torch_module_rms_spot`,
       :doc:`gallery/optimization/torch_module_custom_objective`
   * - visualize the solution space of a design?
     - :doc:`gallery/differentiable_ray_tracing/solution_space_visualization`
   * - use machine learning to predict or classify lens performance?
     - :doc:`Tutorial 9a <examples/Tutorial_9a_Predicting_Lens_Performance_with_Random_Forest>`,
       :doc:`Tutorial 9b <examples/Tutorial_9b_Classifying_Ray_Path_Failures_with_Machine_Learning>`

----

Extend Optiland
---------------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - write a custom surface type?
     - :doc:`Tutorial 8a <examples/Tutorial_8a_Custom_Surface_Types>`
   * - write a custom coating?
     - :doc:`Tutorial 8b <examples/Tutorial_8b_Custom_Coating_Types>`
   * - write a custom optimization algorithm?
     - :doc:`Tutorial 8c <examples/Tutorial_8c_Custom_Optimization_Algorithm>`
   * - write a custom operand?
     - :doc:`Tutorial 3c <examples/Tutorial_3c_User_Defined_Optimization>`
   * - ship my extension as an installable plugin?
     - :doc:`developers_guide/plugin_packages`
   * - understand Optiland's internals before contributing?
     - :doc:`developers_guide/architecture`

----

Visualize
---------

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - How do I …
     - Where to look
   * - change the plot theme (including dark mode)?
     - :doc:`gallery/miscellaneous/themes`
   * - draw the system in 3D?
     - :doc:`gallery/miscellaneous/lens_draw_projection`
   * - model an extended (non-point) source?
     - :doc:`gallery/extended_sources/beam_shaping_singlet`
   * - use the graphical interface instead of scripting?
     - :doc:`gui_quickstart`

----

Not listed here?
----------------

If your question is not answered above, it is worth telling us — a missing entry is a
gap in the documentation, not a gap in your understanding. Please
`open an issue <https://github.com/optiland/optiland/issues>`_ or start a
`discussion <https://github.com/optiland/optiland/discussions>`_.
