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
from pathlib import Path

ROOT = Path(__file__).parents[1]
EXAMPLE_DIRECTORY = ROOT / "docs" / "examples" / "all"
BUILD_DIRECTORY = EXAMPLE_DIRECTORY / "report" / "_build"
REPORT_DIRECTORY = ROOT / "docs" / "report"

subprocess.run([sys.executable, "report.py"], check=True, cwd=EXAMPLE_DIRECTORY)
shutil.rmtree(REPORT_DIRECTORY / "html", ignore_errors=True)
shutil.copytree(BUILD_DIRECTORY / "html", REPORT_DIRECTORY / "html")
shutil.copy2(
    BUILD_DIRECTORY / "benchmarking_report.pdf",
    REPORT_DIRECTORY / "benchmarking_report.pdf",
)
