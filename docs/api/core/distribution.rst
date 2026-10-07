distribution
============

.. automodule:: distribution

   
   .. rubric:: Functions

   .. autosummary::
   
      create_distribution
   
   .. rubric:: Classes

   .. autosummary::
   
      BaseDistribution
      CrossDistribution
      GaussianQuadrature
      HexagonalDistribution
      LineXDistribution
      LineYDistribution
      RandomDistribution
      RingDistribution
      SobolDistribution
      UniformDistribution

Before point generation, ``GaussianQuadrature.quadrature_weights`` is ``None``.
After each generation it exposes the current ``weights`` array, representing
normalized area on the distribution-coordinate unit disk: the weights sum to one
and approximate ``(1 / pi) * integral_unit_disk f(x, y) dA``. They include no
Jacobian for a mapping to a physical pupil, stop, or exit pupil and no intensity or
apodization factor. Other distributions return ``None`` rather than implying an
equal-area rule.
