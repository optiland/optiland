Multi-Sequence Tracing Framework
=================================

The sequences framework (``optiland.sequences``) lets a single :class:`~optiland.optic.optic.Optic`
be traced through user-defined **sub-sequences**: alternative traversal orders over the
*same* surface objects. This is how Optiland models ghost paths, reverse traces, and
sub-component views without duplicating the optical system.

Multi-sequence tracing is the middle ground between a plain ``Optic`` (a fixed surface
order) and full non-sequential ray tracing (no order at all): same surfaces, same
parameters, just a different order. Branching arms (beamsplitters) and true
non-sequential (ray-driven) tracing are separate, unrelated features.

Why not just reorder the surfaces?
-----------------------------------

A naive implementation would build a second list of the same ``Surface`` objects and
trace them in a new order. This breaks in two ways:

1. **Recorded ray data lives on the surface.** ``Surface._record_real`` writes
   ``surface.x/y/z/L/M/N/opd/intensity`` in place. A sequence that revisits a surface
   (e.g. a ghost path bouncing off it twice) would have the second hit silently
   overwrite the first, corrupting anything reading that surface's data.
2. **Incident material is derived from the physical chain.** ``Surface.material_pre``
   resolves via ``previous_surface.material_post``. Reordering or reversing the
   traversal without touching that chain yields physically wrong refraction.

``SurfaceView``
----------------

The unit of a sequence is not ``Surface`` but :class:`~optiland.sequences.surface_view.SurfaceView`,
a lightweight proxy that **shares** state with a base ``Surface`` and **owns** the rest:

.. list-table::
   :header-rows: 1

   * - Shared with the base ``Surface``
     - Owned by the view
   * - ``geometry``
     - Record buffers (``x, y, z, L, M, N, opd, intensity, u``)
   * - ``material_post`` (the physical media)
     - ``previous_view`` link
   * - ``aperture``, ``semi_aperture``
     - ``reverse`` flag
   * - Coating / BSDF objects (non-polarized)
     - ``interaction_model`` instance (rebound, ``is_reflective`` overridable)
   * - ``is_stop``, ``comment``, ``surface_type``, ``thickness``
     - Polarized coatings (``FresnelCoating``, ``ThinFilmCoating``): rebound copies

Because geometry and non-polarized materials/coatings are shared **by reference**,
editing a base surface, or optimizing a variable on it, is immediately visible in every
sequence built over it. Polarized coatings are the one exception: a
``FresnelCoating``/``ThinFilmCoating`` is constructed against a fixed ``(material_pre,
material_post)`` pair, which a reversed or reflected view resolves differently than its
base surface does. ``SurfaceView`` therefore rebuilds its own copy of the coating,
bound to the view's own resolved media (and, for a multilayer ``ThinFilmCoating``, with
the layer stack reversed for a ``reverse`` view) — everything else about the coating
(materials, layer thicknesses) still traces back to the base surface's definition.

Riding the existing pipeline
-----------------------------

``_TracingCoordinator.trace()`` is stateless and reaches everything through the
``surface`` argument it is given. The ray-side dispatch (``RealRays.trace_on_surface``,
``record_on_surface``) is likewise pure duck typing, calling ``surface._trace_real(self)``
and ``surface._record_real(self)``. A ``SurfaceView`` implementing that same interface
therefore traces through the **existing, unforked** pipeline. Concretely, every kernel on
``SurfaceView`` (``trace``, ``reset``, ``_trace_real``, ``_trace_paraxial``,
``_record_real``, ``_record_paraxial``) dispatches to ``type(base_surface).<kernel>``
bound to the view, rather than reimplementing ``Surface``'s physics inline. This matters
in practice: :class:`~optiland.surfaces.object_surface.ObjectSurface` and
:class:`~optiland.surfaces.image_surface.ImageSurface` override several of these kernels
(the object surface skips the coordinate-system localize/globalize step entirely, since
the object is often at ``z = -inf``, where that transform is undefined). Reimplementing
generic ``Surface`` physics inline would silently diverge for those two surface types.

This is a deliberate constraint, not an implementation detail: zero changes to
``Surface``, ``_TracingCoordinator``, or the rays classes are needed for this feature.
A design that forks the trace loop instead will rot on the next physics change to that
loop.

Material and direction resolution
------------------------------------

For each step, given the base surface and whether that step is traversed in reverse:

- **Forward:** ``material_pre`` = the base surface's own ``material_pre`` (via its
  normal chain); ``material_post`` = the base surface's own ``material_post``.
- **Reverse:** the two are swapped.
- **Reflective override:** ``material_post`` collapses to ``material_pre`` regardless of
  direction, since a reflection never crosses into the far medium.

A sequence is physically consistent iff, for every adjacent pair of steps, the previous
step's effective exit medium equals the next step's incident medium. This is checked at
construction time by :func:`~optiland.sequences.resolver.validate_sequence`, which raises
:class:`~optiland.sequences.resolver.SequenceValidationError` naming the offending step
index. Inconsistent sequences fail loudly at construction, not silently at trace time.

Step syntax and direction inference
--------------------------------------

A raw step is either a bare surface index (forward, nominal interaction) or an
``(index, "reflect")`` / ``(index, "refract")`` pair to force an interaction type at
that surface — a plain ``list`` of the same two elements (e.g. ``[3, "reflect"]``) is
accepted identically, since that is the shape a pair round-trips to through JSON.
Direction is not specified explicitly; it starts forward and flips after
every reflective step, since a reflection is what physically reverses the direction of
propagation. A surface that is itself a nominal mirror (``is_reflective`` already true on
its base interaction model, e.g. the primary of a Cassegrain) also flips the direction
even when the step gives no explicit override.
:func:`~optiland.sequences.steps.parse_steps` implements the raw-step inference;
:func:`~optiland.sequences.resolver.resolve_sequence` refines it against each base
surface's nominal reflectivity.

For example, a two-bounce ghost between surfaces 2 and 3::

    steps = [0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4]

reads as: forward through 0, 1, 2; partially reflect off 3 (direction now reverse);
partially reflect off 2 (direction now forward again); continue forward through 3, 4.

``SequencedSurfaceGroup`` and ``SequencedOptic``
---------------------------------------------------

:class:`~optiland.sequences.sequenced_surface_group.SequencedSurfaceGroup` presents the
read/trace subset of the :class:`~optiland.surfaces.surface_group.SurfaceGroup`
interface (stacked ``x``/``y``/``z``/... records, ``positions``, ``stop_index``,
``n()``, ``trace()``, ``reset()``) over a resolved list of views. Unlike
``SurfaceGroup``, a sequence is static once resolved: there is no ``add``/``remove``; to
change the traversal, resolve a new sequence.

:class:`~optiland.sequences.sequenced_optic.SequencedOptic` composes over a base
``Optic`` rather than subclassing it. ``aperture``, ``fields``, ``wavelengths``,
``polarization``, ``apodization``, and ``paraxial`` are delegated to the base optic by
reference; ``surfaces`` is exposed as a ``SequencedSurfaceGroup``.

Ray definition (conjugates, aperture stop, ray aiming) always comes from the base
optic's own nominal sequence: a sub-sequence defines *traversal* only.
``SequencedOptic.trace()``/``trace_generic()`` reuse the base optic's own
``RayGenerator`` to build the incoming rays, then trace them through the sequence's own
views instead of the base surfaces. This is deliberate: a ghost sub-sequence frequently
has no meaningful conjugates of its own, so deriving rays from the base sequence sidesteps
that problem entirely rather than solving it.

The only change to ``Optic`` itself is the additive
:meth:`~optiland.optic.optic.Optic.add_sequence` method, which resolves a
``SequencedOptic`` and stores it in ``optic.sequences[name]``::

    seq = optic.add_sequence(
        "ghost_2_3",
        steps=[0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4],
    )
    rays = seq.trace(Hx=0, Hy=0, wavelength=0.55)

Because ``SequencedOptic`` exposes ``trace``, ``trace_generic``, ``surfaces``, ``n()``,
``object_surface``, and ``image_surface`` with the same shapes as ``Optic``, most
analyses and optimization operands work against a sequence unmodified. ``image_surface``
means the sequence's own terminal step here, not necessarily the system's physical image
plane, since a ghost path does not always end there.

Serialization
----------------

``Optic.to_dict()``/``Optic.from_dict()`` round-trip every sequence registered via
``add_sequence``: each entry in ``optic.sequences`` is serialized as its ``raw_steps``
list under a top-level ``"sequences"`` key, and re-resolved against the (already
deserialized) base surfaces on load. Because a sequence is just a raw step list resolved
against the base surfaces, this is enough to reconstruct it exactly — there is no
separate view state to persist.

Known limitations (v1)
-------------------------

- **No per-sequence first-order analysis.** ``paraxial`` is always the base optic's; a
  sequence cannot yet define its own conjugates, entrance pupil, or first-order
  properties independent of the base system.
- **No visualization overlay.** There is no built-in way yet to draw the base system and
  a sequence together, color-coded by traversal.
- **No automated ghost enumeration.** Sequences are hand-specified; there is no helper
  that proposes candidate 2-bounce ghost paths from a surface list.
- **Analyses are not guaranteed to be meaningful for every sequence.** Some rays in a
  sequence may fail to survive real geometry (total internal reflection, missing the
  clear aperture after a bounce); this shows up as ``NaN`` in ray data, consistent with
  how the rest of Optiland already represents non-surviving rays.

None of these block ordinary use (ghost analysis, reverse traces, sub-component checks);
they are tracked as follow-up work.
