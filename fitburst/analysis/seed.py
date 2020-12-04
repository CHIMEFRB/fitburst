#!/usr/bin/env python
"""Sample analysis functions."""
import uuid
from typing import Union


def get_uuid(flavor: str = None) -> Union[str, bytes]:
    """Generate a Universally Unique Identifier.

    Parameters
    ----------
    flavor : str, optional
        flavor of the uuid, can be str, hex or bytes, by default None

    Returns
    -------
    Union[str, bytes]
        uuid

    Raises
    ------
    ValueError
        Raised when uuid flavor is not defined
    """
    if flavor == "str":
        return str(uuid.uuid1())
    elif flavor == "hex":
        return uuid.uuid1().hex
    elif flavor == "bytes":
        return uuid.uuid1().bytes
    else:
        raise ValueError("undefined flavor")
