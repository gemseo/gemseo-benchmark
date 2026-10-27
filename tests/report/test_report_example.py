# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Test the example generating the sample benchmarking report."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import weasyprint  # noqa: F401
except (ImportError, OSError):
    # WeasyPrint raises an OSError when its native libraries cannot be loaded.
    HAS_WEASYPRINT = False
else:
    HAS_WEASYPRINT = True

EXAMPLE_PATH = Path(__file__).parents[2] / "docs" / "examples" / "all" / "report.py"


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_WEASYPRINT,
    reason="WeasyPrint or its system libraries are not installed.",
)
def test_report_example(tmp_path):
    """Check that the example generating the benchmarking report runs end-to-end.

    mkdocs-gallery does not execute this example
    since its file name is not prefixed by `plot_`.
    It is run in its own process
    since it changes the global configuration of GEMSEO.
    """
    # The report is built by the properdocs executable of the environment of the tests.
    path = os.pathsep.join((str(Path(sys.executable).parent), os.environ["PATH"]))
    subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        cwd=tmp_path,
        env={**os.environ, "PATH": path},
    )
    assert (tmp_path / "report" / "_build" / "benchmarking_report.pdf").is_file()
