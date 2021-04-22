"""Generation of a benchmarking report"""
from os import chdir
from pathlib import Path
from shutil import copy
from subprocess import call
from typing import Dict, Iterable, Union

from gemseo.algos.opt.opt_factory import OptimizersFactory
from jinja2 import Environment, FileSystemLoader

from problems.problems_group import ProblemsGroup


class Report(object):
    """A benchmarking report."""
    TEMPLATES_DIR = Path(__file__).parent / "templates"
    INDEX_FILENAME = "index.rst"
    ALGOS_FILENAME = "algorithms.rst"
    GROUP_FILENAME = "group.rst"

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
            histories_paths,  # type: Dict[str, Dict[str, str]]
    ):  # type: (...) -> None
        """
        Args:
            root_directory: The path to the root directory of the report.
            algorithms: The compared algorithms.
            problems_groups: The groups of reference problems.
            histories_paths: The paths to the reference histories for each algorithm.

        """
        self._root_directory = Path(root_directory)
        self._algorithms = algorithms
        self._problems_groups = problems_groups
        self._histories_paths = histories_paths
        # TODO: check key / name consistency
        if not set(algorithms) <= set(histories_paths):
            raise ValueError("Nonexistent histories: {}"
                             .format(", ".join(set(algorithms) - set(histories_paths))))

    def generate_report_sources(self):  # type: (...) -> None
        """Generate the source files of the benchmarking report."""
        self._create_root_directory()
        self._create_algos_file()
        self._create_groups_files()
        self._create_index()
        self._build_report()

    def _create_root_directory(self):  # type: (...) -> None
        """Create the source directory and basic files."""
        root_directory = self._root_directory
        root_directory.mkdir(exist_ok=True)
        # Create the subdirectories
        (root_directory / "_static").mkdir(exist_ok=True)
        for directory in [Report.GROUPS_DIR, Report.IMAGES_DIR]:
            (root_directory / directory).mkdir(exist_ok=True)
        # Create the basic source files
        for source_file in (Report.CONF_PATH, Report.MAKE_PATH, Report.MAKEFILE_PATH):
            copy(source_file, root_directory / source_file.name)

    def _create_algos_file(self):  # type: (...)-> None
        """Create the file describing the algorithms"""
        # Get the descriptions of the algorithms
        algos_descriptions = dict()
        for a_name in self._algorithms:
            library = OptimizersFactory().create(a_name)
            algos_descriptions[a_name] = library.lib_dict[a_name][library.DESCRIPTION]

        # Create the file
        file_path = self._root_directory / Report.ALGOS_FILENAME
        Report._fill_template(
            file_path,
            Report.ALGOS_FILENAME,
            algorithms=algos_descriptions
        )

    def _create_groups_files(self):  # type: (...) -> None
        """Create the files corresponding to the problems groups."""
        for a_group in self._problems_groups:
            # Create the directory dedicated to the group
            group_directory = (self._root_directory / Report.IMAGES_DIR /
                               a_group.name.replace(" ", "_"))
            group_directory.mkdir(exist_ok=True)

            # Generate the data profile
            data_profile_path = group_directory / "data_profile.jpg"
            a_group.generate_data_profile(
                self._algorithms, self._histories_paths, show=False,
                destination_path=data_profile_path,
            )
            data_profile = ".. image:: /{}" \
                .format(data_profile_path.relative_to(self._root_directory).as_posix())

            # Create the file
            group_path = (self._root_directory / Report.GROUPS_DIR /
                          "{}.rst".format(a_group.name))
            Report._fill_template(
                group_path,
                Report.GROUP_FILENAME,
                name=a_group.name,
                description=a_group.description,
                data_profile=data_profile,
            )

    def _create_index(self):  # type: (...) -> None
        """Create the index file of the reST report."""
        # Create the table of contents tree
        toctree_contents = [Report.ALGOS_FILENAME]
        toctree_contents.extend([
            "{}/{}".format(Report.GROUPS_DIR, a_group.name)
            for a_group in self._problems_groups
        ])

        # Create the file
        index_path = self._root_directory / Report.INDEX_FILENAME
        Report._fill_template(
            index_path, Report.INDEX_FILENAME, documents=toctree_contents
        )

    @staticmethod
    def _fill_template(
            file_path,  # type: Path
            template_name,  # type: str
            **kwargs
    ):  # type: (...) -> None
        """Fill a file template.

        Args:
            file_path: The path to the file to be written.
            template_name: The name of the file template.

        Returns:
            The filled file template.

        """
        file_loader = FileSystemLoader(Report.TEMPLATES_DIR)
        environment = Environment(loader=file_loader)
        template = environment.get_template(template_name)
        file_contents = template.render(**kwargs)
        with file_path.open("w") as file:
            file.write(file_contents)

    def _build_report(self):  # type: (...) -> None
        """Build the benchmarking report."""
        root_directory = self._root_directory
        chdir(root_directory)
        call("make html", shell=True)
        call("make latexpdf", shell=True)
