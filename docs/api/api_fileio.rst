File IO
=======

This section covers Optiland functionality related to reading and writing various file formats, including Zemax, CODE V, and Optiland files.

The ``optiland.fileio`` package handles saving and loading of optical systems. 
It supports the native Optiland JSON format, as well as Zemax (.zmx), CODE V (.seq), 
and OSLO (.len) files.

.. autosummary::
   :toctree: generated
   :nosignatures:

   optiland.fileio.load_optiland_file
   optiland.fileio.save_optiland_file
   optiland.fileio.zemax
   optiland.fileio.codev
   optiland.fileio.oslo
