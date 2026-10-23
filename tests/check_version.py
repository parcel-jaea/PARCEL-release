#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""版の照合 (ビルド不要)。

  python3 tests/check_version.py            # パッケージのルートで。不一致があれば終了コード 1

版の正は Include/parcel_version.hpp の PARCEL_VERSION_MAJOR / MINOR / PATCH。C / Fortran の API と Python の
ネイティブ層はそこから値を取るので照合は要らない (実行時の照合は回帰試験の version の項目: examples/sample_c.c、
examples/sample_f.f90、tests/regression/sample_py.py と、検証サンプルの [VERSION] の行)。
ここでは、ヘッダから取れずに版を書き写しているファイルを照合する:
  python/pyproject.toml の version、python/parcel/__init__.py の __version__   MAJOR.MINOR.PATCH と同じ
  spack のレシピ 2 本の version("...")                                         MAJOR.MINOR.PATCH と同じ
  CITATION.cff の version                                                        MAJOR.MINOR.PATCH と同じ
  spack の環境の定義 (spack/stack/envs/*/spack.yaml) と README の parcel-sycl@ / py-parcel-sycl@  MAJOR.MINOR.PATCH と同じ
回帰試験の生成器 (tests/regression/gen_regression.py) も read_header_version() で版を読み、
試験に PARCEL_EXPECT_VERSION として渡す。
"""
import glob
import os
import re
import sys

sys.dont_write_bytecode = True       # パッケージの中に __pycache__ を作らない
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_header_version(root=ROOT):
    """Include/parcel_version.hpp の (MAJOR, MINOR, PATCH)"""
    text = open(os.path.join(root, "Include", "parcel_version.hpp"), encoding="utf-8").read()
    out = []
    for key in ("MAJOR", "MINOR", "PATCH"):
        m = re.findall(r"^#define PARCEL_VERSION_%s (\d+)\s*$" % key, text, re.M)
        if len(m) != 1:
            raise ValueError("Include/parcel_version.hpp: PARCEL_VERSION_%s の定義が %d 個" % (key, len(m)))
        out.append(int(m[0]))
    return tuple(out)


def version_string(root=ROOT):
    return "%d.%d.%d" % read_header_version(root)


def _one(path, pattern, root):
    text = open(os.path.join(root, path), encoding="utf-8").read()
    m = re.findall(pattern, text, re.M)
    if len(m) != 1:
        return None, "%s: 版の記述が %d 個 (1 個のはず)" % (path, len(m))
    return m[0], None


def check(root=ROOT):
    """(結果の行のリスト, 不一致の数)"""
    major, minor, patch = read_header_version(root)
    full = "%d.%d.%d" % (major, minor, patch)
    targets = [
        ("python/pyproject.toml", r'^version = "([^"]+)"', [full]),
        ("python/parcel/__init__.py", r'^__version__ = "([^"]+)"', [full]),
        ("spack/spack_repo/parcel/packages/parcel_sycl/package.py", r'^\s*version\("([^"]+)"', [full]),
        ("spack/spack_repo/parcel/packages/py_parcel_sycl/package.py", r'^\s*version\("([^"]+)"', [full]),
        ("CITATION.cff", r'^version: "?([^"\s]+)"?\s*$', [full]),
    ]
    lines, bad = ["Include/parcel_version.hpp: %s" % full], 0
    for path, pat, ok_values in targets:
        got, err = _one(path, pat, root)
        if err:
            lines.append("NG  " + err)
            bad += 1
        elif got in ok_values:
            lines.append("OK  %s: %s" % (path, got))
        else:
            lines.append("NG  %s: %s (期待 %s)" % (path, got, " か ".join(ok_values)))
            bad += 1
    # 仕様の版の指定 (parcel-sycl@X・@=X / py-parcel-sycl@X・@=X): spack の環境の定義 (specs と develop) と README。すべて MAJOR.MINOR.PATCH
    spec_files = sorted(glob.glob(os.path.join(root, "spack", "stack", "envs", "*", "spack.yaml")))
    spec_files += [os.path.join(root, p) for p in ("README.md", "spack/README.md")]
    for p in spec_files:
        if not os.path.exists(p):
            continue
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        vers = re.findall(r"(?<![\w-])(?:py-)?parcel-sycl@=?([0-9][0-9.]*[0-9])", open(p, encoding="utf-8").read())
        wrong = sorted(set(v for v in vers if v != full))
        if wrong:
            lines.append("NG  %s: parcel-sycl@%s (期待 @%s)" % (rel, " / @".join(wrong), full))
            bad += 1
        elif vers:
            lines.append("OK  %s: @%s が %d か所" % (rel, full, len(vers)))
    return lines, bad


def main():
    lines, bad = check()
    print("\n".join(lines))
    print("版の照合: %s" % ("一致" if bad == 0 else "%d 件不一致" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
