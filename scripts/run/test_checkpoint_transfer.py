#!/usr/bin/env python3
"""Compile and exercise the training resume path on all three real checkpoints."""
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("transfer", Path(__file__).with_name("transfer_pendulum_to_acrobot.py"))
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


def main():
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    includes = ["src/engine", "src/experiments", "src/experiments/evaluators",
                "src/environments", "src/environments/classic_control", "src/logging", "modules/libfastsim/src"]
    flags = ["-std=c++23", "-DHPCC", "-Wno-deprecated-declarations", "-Wno-format",
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
    sources = ["tests/checkpoint_transfer_test.cc", "src/engine/RegisterMachine.cc",
               "src/engine/TPG.cc", "src/engine/instruction.cc", "src/engine/misc.cc",
               "src/engine/team.cc", "src/engine/hebbian.cc"]
    with tempfile.TemporaryDirectory(prefix="tpg-transfer-test-") as directory:
        if sys.platform == "darwin":
            # Linux-style GL include paths for the SDK's existing headers.
            sdk = Path(subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip())
            gl = Path(directory) / "include/GL"
            gl.mkdir(parents=True)
            for header in ("gl.h", "glu.h", "glext.h"):
                (gl / header).symlink_to(sdk / "System/Library/Frameworks/OpenGL.framework/Headers" / header)
            (gl / "glut.h").symlink_to(sdk / "System/Library/Frameworks/GLUT.framework/Headers/glut.h")
            includes.append(str(gl.parent))
        executable = Path(directory) / "test"
        command = compiler + flags + [f"-I{path}" for path in includes] + sources
        command += links + ["-lboost_iostreams", "-lyaml-cpp", "-o", str(executable)]
        subprocess.run(command, cwd=ROOT, check=True)
        for variant, experiment in transfer.SOURCE_CONFIGS.items():
            working = Path(directory) / variant
            working.mkdir()
            config = working / "acrobot.yaml"
            source = ROOT / "experiments" / experiment / "checkpoints/cp.1000.1.0.rslt"
            config.write_text(transfer.match_checkpoint_registers(
                transfer.experiment_configs(variant)[1], source))
            standalone = ROOT / "configs" / f"acrobot_{variant}.yaml"
            for checked_config in (standalone, config):
                subprocess.run([str(executable), str(checked_config), str(source), variant],
                               cwd=working, check=True)


if __name__ == "__main__":
    main()
