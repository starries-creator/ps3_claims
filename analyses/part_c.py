"""Legacy import path for the shared ``analyses.claims_evaluation`` module."""

import sys

from analyses import claims_evaluation

sys.modules[__name__] = claims_evaluation
