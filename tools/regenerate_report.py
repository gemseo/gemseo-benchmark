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

"""Regenerate the sample benchmarking report committed in docs/report.

Run it from the root of the repository with
`uv run --group doc python tools/regenerate_report.py`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXAMPLE_PATH = ROOT / "docs" / "examples" / "all" / "report.py"
REPORT_DIRECTORY = ROOT / "docs" / "report"

# The example writes its outputs in the current working directory,
# so it runs in a fresh directory to start from scratch
# and to keep the source tree clean.
with tempfile.TemporaryDirectory() as directory:
    subprocess.run([sys.executable, str(EXAMPLE_PATH)], check=True, cwd=directory)
    build_directory = Path(directory) / "report" / "_build"
    shutil.rmtree(REPORT_DIRECTORY / "html", ignore_errors=True)
    shutil.copytree(build_directory / "html", REPORT_DIRECTORY / "html")
    shutil.copy2(
        build_directory / "benchmarking_report.pdf",
        REPORT_DIRECTORY / "benchmarking_report.pdf",
    )
