import os
import re
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Annotated, Any

import pyproj
from mashumaro.types import SerializationStrategy


def validate_path_exists(p: Path) -> Path:
    """Ensures that the path actually exists on the filesystem."""
    if p is not None and not p.exists():
        raise ValueError(f"Path does not exist: '{p}'")
    return p


def resolve_path(path: Path) -> Path:
    return Path(str(os.path.expandvars(path.expanduser())))


def path_type(
    file_okay: bool = True,
    dir_okay: bool = True,
    resolve: bool = True,
    exists: bool = True,
) -> Callable[[Path], Path]:
    """Ensures a path points to a file, directory, or either."""

    def validator(p: Path) -> Path:
        if p is not None and p.exists():
            if not file_okay and p.is_file():
                raise ValueError(f"Path must be a directory, but it is a file: '{p}'")
            if not dir_okay and p.is_dir():
                raise ValueError(f"Path must be a file, but it is a directory: '{p}'")

        if resolve:
            return resolve_path(p)
        else:
            return p

    return validator


def validate_positive(v: Any) -> Any:
    """Ensures value is strictly greater than zero."""
    if v is not None and v <= 0:
        raise ValueError(f"Must be greater than 0, got {v}")
    return v


def validate_non_negative(v: Any) -> Any:
    """Ensures value is greater than or equal to zero."""
    if v is not None and v < 0:
        raise ValueError(f"Must be non-negative (>= 0), got {v}")
    return v


def gt(limit: Any) -> Callable[[Any], Any]:
    """Greater than threshold constraint ( > limit )."""

    def validator(v: Any) -> Any:
        if v is not None and not v > limit:
            raise ValueError(f"Must be strictly greater than {limit}, got {v}")
        return v

    return validator


def ge(limit: Any) -> Callable[[Any], Any]:
    """Greater than or equal to threshold constraint ( >= limit )."""

    def validator(v: Any) -> Any:
        if v is not None and not v >= limit:
            raise ValueError(f"Must be greater than or equal to {limit}, got {v}")
        return v

    return validator


def lt(limit: Any) -> Callable[[Any], Any]:
    """Less than threshold constraint ( < limit )."""

    def validator(v: Any) -> Any:
        if v is not None and not v < limit:
            raise ValueError(f"Must be strictly less than {limit}, got {v}")
        return v

    return validator


def le(limit: Any) -> Callable[[Any], Any]:
    """Less than or equal to threshold constraint ( <= limit )."""

    def validator(v: Any) -> Any:
        if v is not None and not v <= limit:
            raise ValueError(f"Must be less than or equal to {limit}, got {v}")
        return v

    return validator


# ==========================================
# 3. STRING & SEQUENCE VALIDATORS
# ==========================================


def min_len(length: int) -> Callable[[Collection[Any]], Collection[Any]]:
    """Ensures strings, lists, or dicts have at least a minimum length."""

    def validator(v: Collection[Any]) -> Collection[Any]:
        if len(v) < length:
            raise ValueError(
                f"Minimum allowed length/size is {length}, got length {len(v)}"
            )
        return v

    return validator


def max_len(length: int) -> Callable[[Collection[Any]], Collection[Any]]:
    """Ensures strings, lists, or dicts don't exceed a maximum length."""

    def validator(v: Collection[Any]) -> Collection[Any]:
        if v is not None and len(v) > length:
            raise ValueError(
                f"Maximum allowed length/size is {length}, got length {len(v)}"
            )
        return v

    return validator


def regex(pattern: str | re.Pattern[str]) -> Callable[[str], str]:
    """Validates that a string matches a compiled regular expression."""
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def validator(v: str) -> str:
        if v is not None and not compiled.match(v):
            raise ValueError(
                f"String does not match the required pattern: r'{compiled.pattern}'"
            )
        return v

    return validator


def latitude(value: float) -> None:
    if value is not None and not (-90 <= value <= 90):
        raise ValueError(
            f"Latitude values must be in the range [-90, 90], found {value}."
        )


def longitude(value: float) -> None:
    if value is not None and not (-180 <= value <= 180):
        raise ValueError(
            f"Longitude values must be in the range [-180, 180], found {value}."
        )


def in_choices(choices: Collection[Any]) -> Callable[[Any], Any]:
    """Ensures a value is a member of an allowed set of options."""
    allowed = set(choices)

    def validator(v: Any) -> Any:
        if v is not None and v not in allowed:
            raise ValueError(f"Value must be one of {allowed}, got {v!r}")
        return v

    return validator


PositiveInt = Annotated[int, validate_positive]
NonNegativeInt = Annotated[int, validate_non_negative]
NonNegativeFloat = Annotated[float, validate_non_negative]
PositiveFloat = Annotated[float, validate_positive]
Latitude = Annotated[float, latitude]
Longitude = Annotated[float, longitude]

UnitIntervalFloat = Annotated[float, validate_non_negative, le(1.0)]

ExistingPath = Annotated[Path, path_type(exists=True)]
ExistingFile = Annotated[Path, path_type(file_okay=True, dir_okay=False, exists=True)]
ExistingDir = Annotated[Path, path_type(file_okay=False, dir_okay=True, exists=True)]

NonEmptyStr = Annotated[str, min_len(1)]
NonEmptyList = Annotated[list[Any], min_len(1)]


class CRSStrategy(SerializationStrategy, use_annotations=True):
    def __init__(self):
        pass

    def serialize(self, value: pyproj.CRS) -> str:
        return str(value)

    def deserialize(self, value: int | str) -> pyproj.CRS:
        return pyproj.CRS(value)
