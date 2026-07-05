.. _dead_code_audit:

Dead-Code Audit
================

Per the "Dead-Code Audits" section of ``CONTRIBUTING.md``, this page records the result of
running ``vulture optiland/ --min-confidence 80`` against the package, cross-referenced against
``git log -1 --format=%ad -- <file>`` for each hit, so future audits don't have to re-derive
which findings were already investigated and ruled out.

Result: no removals
--------------------

No file or function was confirmed dead in this pass. ``vulture`` produced seven hits; every one
was either a false positive or, in one case, a live (if buggy) code path rather than genuinely
unreachable/unused code:

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Hit
     - Confidence
     - Disposition
   * - ``geometries/nurbs/nurbs_basis_functions.py``: unused variable ``nopython``
     - 100%
     - False positive — a Numba JIT decorator keyword argument, not a real unused local.
   * - ``optiland/jupyter/rl/train_agent.py``: unused ``globals_``/``locals_``
     - 100%
     - Out of package scope — ``optiland/jupyter/`` is git-ignored scratch space
       (``.gitignore``), not shipped code.
   * - ``optiland/jupyter/test_viz_bug.py``: unused import ``analysis``
     - 90%
     - Same as above — git-ignored scratch file.
   * - ``thin_film/optimization/optimizer.py``: unused ``fixed_wavelength_nm`` /
       ``fixed_angle_deg``
     - 100%
     - Assigned-but-unused local variables inside a larger method touched by the thin-film
       decomposition work; not dead *code* (no unreachable branch or orphaned function), just an
       unused intermediate. Left for that area's own follow-up rather than fixed here, to avoid
       mixing an unrelated behavioral touch into a docs/process pass.
   * - ``psf/base.py:205``: unreachable code after ``if``/``elif``/``else``
     - 100%
     - **Not simple dead code.** ``BasePSF.view()`` has an ``if projection == "2d": ... return``
       / ``elif projection == "3d": ... return`` / ``else: raise`` chain, after which sits
       ``if is_gui_embedding and hasattr(current_fig, "canvas"): current_fig.canvas.draw_idle()``
       — genuinely unreachable, since both live branches already return. This reads as a latent
       bug (the GUI-embedding canvas refresh silently never fires) rather than vestigial code
       with no purpose. Deleting the lines would just make the bug permanent instead of fixing
       it; fixing it is a behavioral change to a numerically-sensitive, GUI-adjacent path and
       belongs in its own PR with its own test, not bundled into a dead-code sweep. Flagged here
       for a follow-up fix, not removed.

Conclusion
----------

Per the audit methodology's own caution against assuming code is dead without verification, this
pass confirms there is currently no safe, atomic dead-code removal to make. The one interesting
finding (``psf/base.py``) is logged above as a bug-fix candidate rather than acted on here.
