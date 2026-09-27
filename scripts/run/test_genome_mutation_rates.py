#!/usr/bin/env python3
"""Compile and run engine integration tests without a simulator or MPI run."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    includes = ["src/engine", "src/experiments", "src/experiments/evaluators",
                "src/environments", "src/logging", "modules/libfastsim/src"]
    flags = ["-std=c++23", "-Wno-deprecated-declarations", "-Wno-format",
             "-ffunction-sections", "-fdata-sections"]
    if sys.platform == "darwin":
        prefix = subprocess.check_output(["brew", "--prefix"], text=True).strip()
        includes += [f"{prefix}/include", f"{prefix}/include/eigen3"]
        links = [f"-L{prefix}/lib", "-Wl,-dead_strip"]
    else:
        includes += ["/usr/include/eigen3"]
        flags += shlex.split(subprocess.check_output(
            ["pkg-config", "--cflags", "ompi-cxx", "eigen3", "yaml-cpp"], text=True))
        links = ["-Wl,--gc-sections"]
    sources = ["tests/mutation_rate_genome_test.cc", "src/engine/RegisterMachine.cc",
               "src/engine/TPG.cc", "src/engine/instruction.cc", "src/engine/misc.cc"]
    with tempfile.TemporaryDirectory(prefix="tpg-genome-test-") as directory:
        executable = str(Path(directory) / "test")
        command = compiler + flags + [f"-I{path}" for path in includes] + sources
        command += links + ["-lboost_iostreams", "-lyaml-cpp", "-o", executable]
        subprocess.run(command, cwd=root, check=True)
        subprocess.run([executable], cwd=root, check=True)


if __name__ == "__main__":
    main()
