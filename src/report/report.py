"""Generation of a benchmarking report"""
from os import chdir
from pathlib import Path
from shutil import copy
from subprocess import call
from typing import Dict, Iterable, Union

from problems.problems_group import ProblemsGroup


class Report(object):
    """A benchmarking report."""
    TEMPLATES_DIR = Path(__file__).parent / "templates"
    INDEX_TEMPLATE_PATH = TEMPLATES_DIR / "index_template.rst"
    GROUP_TEMPLATE_PATH = TEMPLATES_DIR / "group_template.rst"

    BASIC_SOURCES_DIR = Path(__file__).parent / "basic_sources"
    CONF_PATH = BASIC_SOURCES_DIR / "conf.py"
    MAKE_PATH = BASIC_SOURCES_DIR / "make.bat"
    MAKEFILE_PATH = BASIC_SOURCES_DIR / "Makefile"

    GROUPS_DIR = "groups"
    IMAGES_DIR = "images"

    def __init__(
            self,
            root_directory,  # type: Union[str, Path]
            algorithms,  # type: Dict[str, Dict]
            problems_groups,  # type: Iterable[ProblemsGroup]
            histories_paths,  # type: Dict[str, str]
    ):  # type: (...) -> None
        """
        Args:
            root_directory: The path to the root directory of the report.
            algorithms: The compared algorithms.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories.

        """
        self._root_directory = Path(root_directory)
        self._algorithms = algorithms
        self._problems_groups = problems_groups
        self._histories_paths = histories_paths

    def generate_report_sources(self):  # type: (...) -> None
        """Generate the source files of the benchmarking report."""
        self._create_root_directory()
        self._create_groups_files()
        self._create_index()
        self._build_report()

    def _create_root_directory(self):  # type: (...) -> None
        """Create the source directory and basic files."""
        root_directory = self._root_directory
        root_directory.mkdir(exist_ok=True)
        (root_directory / "_static").mkdir(exist_ok=True)
        for directory in [Report.GROUPS_DIR, Report.IMAGES_DIR]:
            (root_directory / directory).mkdir(exist_ok=True)
        for source_file in (Report.CONF_PATH, Report.MAKE_PATH, Report.MAKEFILE_PATH):
            copy(source_file, root_directory / source_file.name)

    def _create_groups_files(self):  # type: (...) -> None
        """Create the files corresponding to the problems groups."""
        group_template_path = Path(Report.GROUP_TEMPLATE_PATH)
        for a_group in self._problems_groups:

            # Create the directory dedicated to the group
            group_directory = (self._root_directory / Report.IMAGES_DIR /
                               a_group.name.replace(" ", "_"))
            group_directory.mkdir(exist_ok=True)

            # Generate the data profile of the group
            data_profile_path = group_directory / "data_profile.jpg"
            a_group.generate_data_profile(
                self._algorithms, show=False, destination_path=data_profile_path
            )
            data_profile=".. image:: /{}"\
                .format(data_profile_path.relative_to(self._root_directory).as_posix())

            # Create the file dedicated to the group
            group_template = Report._read_template(group_template_path)
            group_contents = group_template.format(
                name=a_group.name,
                description=a_group.description,
                data_profile=data_profile,
            )
            group_path = (self._root_directory / Report.GROUPS_DIR /
                          "{}.rst".format(a_group.name))
            Report._write_file(group_path, group_contents)

    def _create_index(self):  # type: (...) -> None
        """Create the index file of the reST report."""
        # Create the table of contents tree
        toctree_contents = "\n".join([
            "   {}/{}".format(Report.GROUPS_DIR, a_group.name)
            for a_group in self._problems_groups
        ])

        # Read the index template
        index_template_path = Path(Report.INDEX_TEMPLATE_PATH)
        index_template = Report._read_template(index_template_path)

        # Fill the index template
        index_contents = index_template.format(toctree_contents=toctree_contents)

        # Create the index
        index_path = self._root_directory / "index.rst"
        Report._write_file(index_path, index_contents)

    @staticmethod
    def _read_template(
            template_path,  # type: Path
    ):  # type: (...) -> str
        """Read the contents of a file template.

        Args:
            template_path: The path to the file template.

        Returns:
            The contents of the file template.

        """
        with template_path.open() as file:
            contents = file.read()
        return contents

    @staticmethod
    def _write_file(
            file_path,  # type: Path
            contents,  # type: str
    ):  # type: (...) -> None
        """Write contents into a file.

        Args:
            file_path: The path to the file to be written.
            contents: The contents to be written in the file.

        """
        with file_path.open("w") as file:
            file.write(contents)

    def _build_report(self):  # type: (...) -> None
        """Build the benchmarking report."""
        root_directory = self._root_directory
        chdir(root_directory)
        call("make html", shell=True)
        call("make latexpdf", shell=True)
