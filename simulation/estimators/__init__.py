"""Online terrain estimators, each tied to a role in the manuscript.

* ``grit`` -- GRIT (Grid Residual Inference of Terrain), the
  deployed estimator: joint, ground-datum-free inference of the Bekker
  exponent ``n`` and friction angle ``phi``. Table 2's joint row.
  Implemented in ``grit_terrain_estimator.py``.
* ``scalar_parent`` -- the frozen scalar parent, which infers only ``n``
  and reads ``phi`` off the manifold. Table 2's comparison row.
* ``bekker_ukf`` -- the analytical-tire UKF used to reproduce Dallas et al.
  under their own protocol (Sec. 5), backed by ``ukf_reference_models``.
* ``terrain_parameterization`` -- the clay-dirt-sand soil manifold every
  backend maps a hypothesis through.

Naming: the manuscript says GRIT where the code and the artifacts say
``grit``. That name is fixed, not stylistic. The identifier is
recorded in the published evidence manifests; the module's own SHA-256 is
pinned as ``candidate_source_sha256`` in the frozen joint replay manifest;
and its path is listed in ``benchmarking/paper_provenance.py``, which the
publisher matches against every recorded run. Renaming the module, or
editing its contents, invalidates the Table 2 evidence chain.
"""
