def parse_version(version: str) -> tuple[int, int, int]:
    """Parse "1.2.3" into (1, 2, 3). Missing patch defaults to 0.

    Raises ValueError for anything that isn't 2-3 dot-separated integers.
    """
    parts = version.split(".")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"invalid version: {version!r}")
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)  # type: ignore[return-value]
