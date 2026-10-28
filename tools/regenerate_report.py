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

    # The new report is first copied next to docs/report,
    # on the same file system,
    # so that a failed copy leaves docs/report unchanged
    # and the old files are then replaced by renaming,
    # which is fast and does not leave a half-written file.
    # If renaming the new HTML directory or the new PDF file fails,
    # the old HTML directory is put back before the error is raised again,
    # so docs/report keeps either the whole old report or the whole new one.
    # If putting the old HTML directory back fails too,
    # docs/report can be inconsistent,
    # but the staging directory is kept with the files not moved back yet
    # and the error says where it is,
    # so that no file is lost and the old report can be restored by hand.
    staging_directory = Path(tempfile.mkdtemp(dir=REPORT_DIRECTORY.parent))
    keep_staging_directory = False
    try:
        new_html_directory = staging_directory / "html"
        new_pdf_path = staging_directory / "benchmarking_report.pdf"
        shutil.copytree(html_directory, new_html_directory)
        shutil.copy2(pdf_path, new_pdf_path)
        old_html_directory = staging_directory / "old_html"
        report_html_directory = REPORT_DIRECTORY / "html"
        if report_html_directory.exists():
            report_html_directory.replace(old_html_directory)

        try:
            new_html_directory.replace(report_html_directory)
            new_pdf_path.replace(REPORT_DIRECTORY / "benchmarking_report.pdf")
        except OSError:
            try:
                # If the new HTML directory is already in docs/report,
                # it is put aside in the staging directory
                # so that the old one can take its place again.
                if not new_html_directory.exists():
                    report_html_directory.replace(new_html_directory)

                if old_html_directory.exists():
                    old_html_directory.replace(report_html_directory)
            except OSError as rollback_error:
                keep_staging_directory = True
                if old_html_directory.exists():
                    restore_note = (
                        "It holds the old HTML directory in old_html "
                        "and the new report in html and benchmarking_report.pdf, "
                        "except the ones already moved to docs/report. "
                        "To restore the old HTML report by hand, "
                        f"remove {report_html_directory} if it exists "
                        "and move old_html there."
                    )
                else:
                    restore_note = (
                        "It holds the new report in html and benchmarking_report.pdf, "
                        "except the ones already moved to docs/report. "
                        "There was no old HTML report, "
                        "so to restore the old state by hand, "
                        f"remove {report_html_directory} if it exists."
                    )

                rollback_error.add_note(
                    "The old report could not be restored, "
                    "so docs/report may be inconsistent. "
                    f"The staging directory {staging_directory} is kept. "
                    f"{restore_note}"
                )
                raise

            raise
    finally:
        # The staging directory is removed with what it still holds,
        # that is the old HTML directory after a success
        # and the new report after a failure whose rollback succeeded.
        if not keep_staging_directory:
            shutil.rmtree(staging_directory, ignore_errors=True)
