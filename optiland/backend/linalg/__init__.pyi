from typing import Any

from numpy.typing import ArrayLike

from optiland._types import BEArrayT
from optiland.backend import ndarray

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LinAlgError(Exception): ...

# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------

def svd(
    a: BEArrayT, full_matrices: bool = True
) -> tuple[BEArrayT, BEArrayT, BEArrayT]: ...
def eig(a: BEArrayT) -> tuple[BEArrayT, BEArrayT]: ...

# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def norm(x: BEArrayT, axis: int | None = None, **kwargs: Any) -> BEArrayT: ...
def solve(a: ArrayLike, b: ArrayLike) -> ndarray: ...
def lstsq(
    a: BEArrayT, b: BEArrayT, rcond: float | None = None
) -> tuple[BEArrayT, BEArrayT, BEArrayT, BEArrayT]: ...

# ---------------------------------------------------------------------------
# Matrix functions
# ---------------------------------------------------------------------------

def pinv(a: BEArrayT, rcond: float | None = None) -> BEArrayT: ...
def det(a: BEArrayT) -> BEArrayT: ...
def inv(a: BEArrayT) -> BEArrayT: ...
