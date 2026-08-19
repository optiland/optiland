"""Zemax File Source Handler

Resolves a Zemax file source (local path or URL) to a local file path,
downloading to a temporary file if needed.

Kramer Harrison, 2024
"""

from __future__ import annotations

import os
import re
import tempfile

import requests


class ZemaxFileSourceHandler:
    """Handles source input resolution for Zemax files (local vs URL).

    Attributes:
        source: The original source string (file path or URL).
    """

    def __init__(self, source: str):
        self.source = source
        self._is_tempfile = False
        self._local_file: str | None = None

    def _is_url(self) -> bool:
        """Return True if source looks like an HTTP/HTTPS URL."""
        return re.match(r"^https?://", self.source) is not None

    def get_local_file(self) -> str:
        """Resolve the source to a local file path.

        Downloads the file to a temporary location if the source is a URL.

        Returns:
            The local file path.

        Raises:
            ValueError: If the URL download fails.
        """
        if self._is_url():
            response = requests.get(self.source, timeout=10)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(response.content)
                    self._local_file = tmp.name
                self._is_tempfile = True
            else:
                raise ValueError("Failed to download Zemax file.")
        else:
            self._local_file = self.source
        return self._local_file

    def cleanup(self) -> None:
        """Remove the temporary file if one was created."""
        if self._is_tempfile and self._local_file and os.path.exists(self._local_file):
            os.remove(self._local_file)
