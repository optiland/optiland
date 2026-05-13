Materials
=========

This section provides a comprehensive overview of the materials available in Optiland.
A material is a set of properties that define the optical behavior of a medium. Specifically in Optiland, a material defines the refractive
index and extinction coefficient of a medium at any wavelength. A material may be described simply by refractive index and abbe number, or it may be a more complex
model based on Sellmeier, or other, coefficients. Optiland provides a wide range of materials, which are outlined here.

Optiland includes a database of common materials based on refrativeindex.info. Any material in this database
can be accessed by name via the Material class.

Abbe Material Model
-------------------

The :class:`~optiland.materials.abbe.AbbeMaterial` class utilizes a robust, data-driven statistical dispersion model to resolve the ambiguity of the two-parameter ($n_d, V_d$) definition. While standard approximations like the "Normal Line" rule often fail for anomalous glasses, Optiland's model is derived from a principal component analysis (PCA) of over 1,000 commercial optical glasses.

The model construction involves:
1.  **Dimensionality Reduction:** Using PCA to quantify the effective degrees of freedom in standard optical glasses.
2.  **Basis Selection:** Applying Sparse Regression (LassoLarsIC) to the Buchdahl dispersion formula to identify the minimum set of coefficients required for accurate spectral reconstruction.

This approach allows for accurate refractive index prediction across the visible and near-infrared spectrum, even for glasses that deviate from the standard "normal line".

For a detailed walkthrough of the model derivation and validation, please refer to the :doc:`Abbe Material Model Building <../references/AbbeMaterial_Model_Building>` notebook.

6th-Order Buchdahl Model Glass
------------------------------

The :class:`~optiland.materials.model.ModelMaterial` class implements a highly accurate 6th-order Buchdahl model glass that is parameterized by the refractive index at the d-line ($n_d$), the Abbe number ($V_d$), and the deviation of the relative partial dispersion ($dPgF$). It supports both visible and infrared (IR) routing architectures to provide precise chromatic modeling:

* **Visible Routing:** If no IR partial dispersion anchors are provided, the model predicts higher-order Buchdahl coefficients ($\nu_3..\nu_6$) using a 20-dimensional regression matrix. The lower-order coefficients ($\nu_1..\nu_2$) are then solved exactly using the F-C and g-F partial-dispersion equations as hard boundary anchors.
* **IR Routing:** An optional IR route is enabled when both reference partial dispersion ($P_{ref}$) and reference wavelength ($\lambda_{ref}$) are supplied. In this mode, the model predicts higher-order coefficients ($\nu_4..\nu_6$) via a 35-dimensional regression matrix. The lower-order coefficients ($\nu_1..\nu_3$) are solved exactly using the F-C, g-F, and the reference-wavelength anchors.

This dual-routing flexibility ensures numerical stability and physical validity for optical design across the visible and infrared spectra.

For an in-depth walkthrough of the model's math, routing logic, and validation details, see the following notebooks:
* :doc:`Visible Routing Derivation <../Buchdahl_model_material_details/Visible_Routing>`
* :doc:`IR Routing Derivation <../Buchdahl_model_material_details/IR_Routing>`
* :doc:`Runtime Model Validation <../Buchdahl_model_material_details/Runtime Validation>`

.. autosummary::
   :toctree: materials/
   :caption: Material Modules

   materials.abbe
   materials.ideal
   materials.material_file
   materials.material
   materials.model
