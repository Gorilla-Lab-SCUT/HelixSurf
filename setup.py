# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import subprocess
import sys
from distutils.sysconfig import get_config_var, get_python_inc

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

__version__ = None
exec(open("helixsurf/version.py", "r").read())


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir=".", **kwa):
        Extension.__init__(self, name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    def build_extensions(self):
        # Ensure that CMake is present and working
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        for ext in self.extensions:
            # extdir = os.path.abspath(os.path.dirname(
            #     self.get_ext_fullpath(ext.name)))
            cfg = "Debug" if self.debug else "Release"

            cmake_args = [
                f"-DCMAKE_BUILD_TYPE={cfg}",
                f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
                f"-DPYTHON_INCLUDE_DIR={get_python_inc()}",
                f"-DPYTHON_LIBRARY={get_config_var('LIBDIR')}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={os.path.abspath(self.build_temp)}",
            ]

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # Config
            subprocess.check_call(
                ["cmake", "-S", ext.cmake_lists_dir] + cmake_args, cwd=self.build_temp
            )

            # Build
            build_args = ["--config", cfg]
            build_args += ["--", "-j16"]
            subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

            # move from build temp to final position
            for ext in self.extensions:
                self.move_output(ext)

    def move_output(self, ext):
        build_temp = os.path.abspath(self.build_temp)
        dest_path = os.path.abspath(self.get_ext_fullpath(ext.name))
        # source_path = os.path.join(build_temp, self.get_ext_filename(ext.name))
        source_path = os.path.join(build_temp, f"{ext.name.split('.')[-1]}.so")
        dest_directory = os.path.dirname(dest_path)
        os.makedirs(dest_directory, exist_ok=True)
        self.copy_file(source_path, dest_path)


setup(
    name="helixsurf",
    version=__version__,
    author="Zhihao Liang",
    author_email="eezhihaoliang@mail.scut.edu.cn",
    description="Scene Reconstruction of Gorilla-Lab",
    long_description="Scene Reconstruction of Gorilla-Lab",
    packages=find_packages(),
    ext_modules=[CMakeExtension(name="helixsurf.libvolume")],
    cmdclass={"build_ext": cmake_build_ext},
)
