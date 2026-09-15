OSLO import compatibility
=========================

Use ``load_oslo_file("design.len")`` to read a sequential OSLO prescription.
Unsupported optical commands emit warnings; inspect them before relying on the
result. To reject unsupported commands and known approximations, use::

    from optiland.fileio import load_oslo_file

    optic = load_oslo_file(
        "design.len", strict=True, material_overrides={"GLASS": verified_material}
    )

Strict imports require an explicitly supplied definition for every named glass.
Use ``material_overrides`` to supply verified native materials. Permissive
imports warn before using the database or an approximation.

Malformed data raises ``ValueError``. Parser errors and unsupported-command
warnings identify the file, line and surface. ``OsloDataParser(...).parse()``
returns an ``OsloDataModel`` containing the parsed prescription before conversion
to an ``Optic``. This model also exposes structured ``diagnostics``.
Strict import is a compatibility check, not a certificate of agreement with OSLO.
No CCL, include file, optimization program or external executable is executed.

Supported mappings
------------------

Command names are case insensitive. Quoted strings may contain semicolons and
``//``; those delimiters split statements and comments only outside strings.
UTF-8 (including BOM) and legacy Windows-1252 text are accepted.

.. list-table:: Sequential prescription support
   :header-rows: 1
   :widths: 30 70

   * - Commands
     - Mapping and limits
   * - ``LEN NEW``, ``NXT``, ``GTO``, ``END``
     - Object, optical and image surfaces; updating a previous surface preserves
       its other data. The first prescription is imported.
   * - ``UNI``, ``EBR``, ``FNO``, ``NAO``, ``NAP``, ``PUK``, ``TELE``
     - Lens units converted to millimeters. Working f-number/image NA/slope
       determine the entrance pupil via a paraxial trace. EBR specifies the axial
       beam radius at surface 1; finite-object imports account for the displaced
       entrance pupil. These specifications require valid paraxial geometry.
       TELE sets object-space telecentricity and is preserved in native JSON.
       Real telecentric launch supports finite object-height fields in air with
       entry along +z, using an equivalent object NA. Other launch combinations
       warn in permissive mode and fail in strict mode.
       Native paraxial chief-ray analysis still aims at the stop; use real rays
       to evaluate the imported telecentric launch.
       NAO requires a finite object and an aperture greater than zero and less
       than the object medium's refractive index; hemisphere/extended launches
       are outside this mapping.
   * - ``ANG``, ``OBH``, ``GIH``; ``RST NEW`` / ``F``
     - Maximum field or explicit fractional X/Y positions, weights and symmetric
       pupil vignetting. Fractional object positions are converted through tangent
       space for angular fields. The angular reference must be below 90 degrees;
       wide-angle ray aiming (WARM) is not mapped. GIH refers to the Gaussian
       focal plane.
       Without a table, generate on-axis, 0.7 and full-field points.
   * - ``WV``, ``WVn``, ``WW``, ``WWn``
     - Replacement and indexed wavelength/weight assignments. Default d/F/C
       wavelengths are 0.58756/0.48613/0.65627 micrometers; WV1 is primary.
       Indexed edits preserve untouched wavelengths; a bulk WV replaces the set.
       The primary wavelength must have positive weight; other weights may be zero.
       Direct-index glass retains the wavelengths active when it was defined.
   * - ``RD``, ``RDF``, ``CV``, ``CVF``, ``TH``, ``THF``, ``CC``
     - Spheres/conics, planar RD=0, signed thickness and infinity sentinels.
       Object distances with magnitude at least 1e8 lens units are infinite,
       independently of the conversion to millimeters.
       Image-surface TH is a focus shift added to the preceding nominal gap
       after solves and pickups. The physical detector position and final gap
       retain that shift; native image thickness is zero. Focus shifts require
       an interior surface; combining them with an image GC reference is not
       mapped.
       Fixed markers describe editing constraints and do not change the snapshot.
   * - ``AD`` through ``AG``; ``ASP ADO/ASR/ARA/ASX`` and ``ASn``
     - AD starts at r^4. ASR uses even radial powers, ARA all positive radial
       powers, ASX triangular-indexed XY monomials. Nonzero radial AS0 is rejected.
       Dimensional coefficients scale with their actual polynomial powers.
       Nonzero ASR AS1, ARA AS1/AS2 and ASX AS0..AS5 retain real-ray geometry
       but warn because the native paraxial engine omits their changes to vertex,
       normal or power. Strict mode rejects those cases; derived paraxial pupils,
       fields and solves must be treated as approximations.
   * - ``CVX``, ``RDX``
     - Toroidal surfaces with supported rotational profiles. Unsupported
       combinations are rejected rather than converted to a different formula.
   * - ``AIR``, ``AIF``, ``RFL``, ``RFH``, ``GLA``, ``GLF``
     - Air, reflection, named catalog glass, constant index and sampled index
       data. Distinct saved indices use ``DataMaterial``: exact at samples,
       linear between samples, with extrapolation rejected. Wavelengths that
       collapse together in the active backend precision cannot be interpolated;
       use higher precision for those samples. Two-parameter model
       glass and historical fallback glasses use approximate Abbe dispersion.
       Strict import requires an explicit material binding. Explicit
       ``material_overrides`` bind names to verified native material definitions;
       No global material registry is changed.
   * - ``AP``, ``APF``, ``AP CHK/UNC``, ``APCK``, ``AST``
     - CHK enables circular clipping; unchecked drawing bounds are omitted. APCK OFF
       disables clipping. Default import uses checked/special apertures for
       clipping, matching OSLO spot-diagram behavior. The default stop is surface 1.
   * - ``DCX/Y/Z``, ``TLA/B/C``, ``DT``, ``TOX/Y/Z``, ``GC``, ``RCO``, ``BEN``
     - OSLO intrinsic Euler rotations, signed X/Y tilts, translation order,
       pivots, preceding global references and coordinate returns. BEN supports
       single-axis local RFL/RFH mirror bends; mixed-axis/global bends and bends
       relying on unmapped TIR-controlled reflection are rejected.
   * - ``PK CV/CVM/TH/THM/LN/LNM/AP/GLA/TD/TDM``
     - Static preceding-surface pickups, relative references and chains.
       Curvature and length constants are additive. Forward/self references are
       rejected. The result is an imported prescription, not live OSLO constraints.
       Solved values feed downstream pickups. TD/TDM retain local pivot data;
       pickups involving global references, coordinate returns or bends are rejected.
   * - ``PY``, ``PYC``, ``PU``, ``PUC``, ``EC``
     - Targeted marginal/chief height, outgoing slope and edge-contact solves.
       Targets are rechecked after rebuilding dependent pickups, pupils and fields;
       later solves must also preserve earlier accepted targets. Unsupported or
       unsatisfied solves restore saved values with a warning in permissive mode;
       strict mode rejects. General simultaneous constraint solving is not provided:
       a coupled case can be rejected even if a joint solution exists in OSLO.
       EC verifies physical contact at the first surface's local meridional edge
       after positioning both surfaces. Relative transforms that prevent this
       contact restore the saved prescription rather than accepting TH alone.
       Telecentric PYC/PUC solves are not mapped because the native paraxial chief
       ray used by those solves does not implement the telecentric launch.
   * - ``GSP``, ``GOR``
     - Ruled gratings through the existing phase model, with grooves parallel
       to local X. Lens-unit spacing is converted to millimeters. Blaze efficiency
       is not inferred.
   * - ``TCE``
     - Finite expansion coefficients retained in the intermediate model.
       Nominal reference-temperature geometry is unchanged; temperature-dependent
       dimensions and thermal studies are not implemented.
   * - ``CFG NEW`` with ``TH``, ``WVn``, ``WWn``; ``CFWT``, ``CFAC``
     - Independent one-based configuration snapshots. Thickness and indexed
       spectrum overrides apply to a copy of configuration 1 before pickups and
       coordinates are evaluated. Weights and active flags remain in the raw model;
       inactive configurations can still be selected explicitly. Alternate
       configurations with solves, and thickness overrides on pickup-controlled
       targets, are rejected pending verified constraint precedence.
   * - ``ATD``, ``CXD``, ``APD``, ``GCD``, ``RCD``, ``BED``, ``PFD``,
       ``TDD``, ``CSD``, ``TSD``
     - Clear the corresponding previously entered surface data/constraints.
   * - ``DES``, ``SNO<n>``, ``NOT``; drawing/group records
     - Names and notes retained in the intermediate model. Drawing records
       DRW/LDP/CBK/ELMDF1/2/BDI/BDD/VX/PF and sequential LMO ELE/EGR, LMN,
       LME do not affect ray tracing. Non-sequential LMO groups are diagnosed.

Specification and real-file validation
--------------------------------------

Mappings were checked against Lambda Research's
`OSLO Program Reference (10 March 2021) <https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf>`_
(printed pp. 18-25: configurations; 43-50: solves/apertures; 51-68: media/coordinates;
69-85: surfaces/gratings; 106-107: Jones elements; 120-123: system setup/wavelengths;
184-190: glass catalogs; 208-210: fields; 215: telecentricity;
508-514: commands),
the `Optics Reference <https://lambdares.com/hubfs/Support/support/OSLOOpticsReference_Sep21.pdf>`_
(pp. 105: special-aperture shapes and rotation; 142-145: coordinate transforms;
151: object conjugates/telecentric launch;
170-171: grating equation), and the
`official demo library <https://lambdares.com/support-posts/lens-demos>`_.
The `current release page <https://lambdares.com/support-posts/oslo-current-release>`_
links the reference editions used during research.

Download the official ``OSLOLensDemos.zip`` separately and run from the repository::

    python scripts/audit_oslo_examples.py OSLOLensDemos.zip oslo-audit.json
    python scripts/audit_oslo_examples.py OSLOLensDemos.zip oslo-audit-torch.json --backend torch

The audit records archive/file SHA-256 hashes, source URL, filenames, backend,
permissive warnings/errors, strict errors and an on-axis trace smoke check.
It performs no network access and does not vendor the proprietary example archive.
Smoke checks do not establish agreement with OSLO ray intercepts. Original minimal
test prescriptions provide independent numerical assertions for units, sag,
indices, poses, aperture clipping, grating directions and solve targets.


Configuration snapshots
-----------------------

Use ``configuration=2`` to select the second snapshot. Each selection starts
from the original parsed base; failed and repeated conversions do not change it.
Inactive configurations may be selected. Alternate solves and thickness
overrides of pickup-controlled values are rejected.
