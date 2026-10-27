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
    html_directory = build_directory / "html"
    pdf_path = build_directory / "benchmarking_report.pdf"
    # docs/report is changed only if the whole report was built,
    # so that a partial build does not replace the committed report.
    missing_paths = [path for path in (html_directory, pdf_path) if not path.exists()]
    if missing_paths:
        msg = (
            "The report was not built completely; "
            f"{' and '.join(str(path) for path in missing_paths)} "
            f"{'does' if len(missing_paths) == 1 else 'do'} not exist, "
            "so docs/report is left unchanged."
        )
        raise FileNotFoundError(msg)

    shutil.rmtree(REPORT_DIRECTORY / "html", ignore_errors=True)
    shutil.copytree(html_directory, REPORT_DIRECTORY / "html")
    shutil.copy2(pdf_path, REPORT_DIRECTORY / "benchmarking_report.pdf")
