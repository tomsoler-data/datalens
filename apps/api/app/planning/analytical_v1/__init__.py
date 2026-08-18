"""
Production analytical planning stack for DataLens.

This package contains production-promoted versions of the
analytical planning components validated through the eval
harness.

Important architectural rule:

    app.planning.analytical_v1

must never import from:

    the evaluation package

Evaluation code may depend on production code in future
versions, but production code must not depend on evaluation
artifacts.
"""

ANALYTICAL_V1_PACKAGE_VERSION = "analytical_v1"
