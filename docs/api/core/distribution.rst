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

Sobol area integration
----------------------

``SobolDistribution`` maps a Sobol sequence to the unit disk and exposes equal
area weights through ``weights``. The weights are ``None`` before generation;
after generating ``N`` points, they form a backend-native array of shape
``(N,)`` with each entry equal to ``1/N``. Their sum is one to floating-point
precision, matching the normalization used by ``GaussianQuadrature.weights``.

For example, the area average of squared radius over the unit disk is 1/2:

.. code-block:: python

   from optiland.distribution import SobolDistribution

   dist = SobolDistribution(seed=42)
   dist.generate_points(1024)
   mean_radius_squared = ((dist.x**2 + dist.y**2) * dist.weights).sum()
   # Approximately 0.5 on either the NumPy or PyTorch backend.

The map ``r = sqrt(u1)``, ``theta = 2*pi*u2`` has constant area Jacobian
``dA = pi du1 du2``. Consequently, ``sum(weights * f(x, y))`` estimates the
normalized area integral, without any additional radial weight. To estimate an
unnormalized integral over a physical disk of radius ``R``, evaluate the
integrand at ``(R*x, R*y)`` and multiply the weighted sum by ``pi*R**2``.
These weights describe the sampling disk; they do not include illumination,
apodization, or the area Jacobian of an optical mapping to another pupil.

Scrambling defaults to ``True`` and can be disabled with
``SobolDistribution(scramble=False)``. The seed remains an optional positional
argument and is interpreted by the active backend. With a fixed seed and
scrambling setting, repeated generation restarts the same sequence within that
backend. NumPy and PyTorch may produce different scrambled samples.
Powers of two preserve Sobol balance properties; other positive integer counts
are also supported, including NumPy integers. Non-integer counts, booleans, and
nonpositive counts raise ``ValueError``. Successful generation replaces points
and weights together; a failed call preserves the previous result.
