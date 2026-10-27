"""Check that a Git source archive builds with the checkout's package version."""

from __future__ import annotations

import argparse
import email.parser
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def run(args, cwd):
    result = subprocess.run(
        list(map(str, args)), cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def wheel_version(directory):
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one built wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Expected one wheel metadata file")
        metadata = email.parser.Parser().parsestr(archive.read(names[0]).decode())
    if metadata["Name"] != "llm-pathway-curator":
        raise ValueError("Unexpected wheel project")
    return metadata["Version"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--expected-version")
    parser.add_argument("--no-isolation", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if run(["git", "diff", "--name-only", "HEAD"], repo):
        raise ValueError("Use a clean, committed checkout for the archive comparison")
    with tempfile.TemporaryDirectory(prefix="llmpath-archive-") as temporary:
        root = Path(temporary)
        archive_path = root / "source.zip"
        run(["git", "archive", "--format=zip", "--output", archive_path, "HEAD"], repo)
        source = root / "source"
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                if not (source / name).resolve().is_relative_to(source.resolve()):
                    raise ValueError("Invalid source archive path")
            archive.extractall(source)
        versions = {}
        for label, path in (("checkout", repo), ("archive", source)):
            outdir = root / (label + "-dist")
            command = [sys.executable, "-m", "build", "--wheel", "--outdir", outdir]
            if args.no_isolation:
                command.append("--no-isolation")
            run(command + [path], repo)
            versions[label] = wheel_version(outdir)
        if versions["checkout"] != versions["archive"]:
            raise ValueError(f"Source archive version differs: {versions}")
        if versions["archive"] in ("0.0.0", "0+unknown"):
            raise ValueError("Archive used an unknown fallback version")
        if args.expected_version and versions["archive"] != args.expected_version:
            raise ValueError(f"Expected {args.expected_version}; got {versions}")
    print(json.dumps({"status": "VERSION_MATCH", **versions}, indent=2))


if __name__ == "__main__":
    main()
