#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
"""配布用の tarball とその sha256 を作る (パッケージのディレクトリから、同じ中身なら同じ tarball になる)。

  python3 tools/make_tarball.py [--outdir <DIR>]     # tarball と <名前>.tar.gz.sha256 を作る (既定 DIR はパッケージの親)
  python3 tools/make_tarball.py --pin [--outdir <DIR>]  # 作った後、spack のレシピ 2 本の version() に sha256 を書く
  python3 tools/make_tarball.py --check              # レシピに書いた sha256 が今のパッケージの tarball と一致するか
  python3 tools/make_tarball.py --list               # tarball に入れるファイルの一覧だけを出す

名前: parcel_sycl-<MAJOR>.<MINOR>.<PATCH> (例 parcel_sycl-1.2.0.tar.gz)。版は Include/parcel_version.hpp から読む。
tarball の中のトップのディレクトリも同じ名前。GitHub の Release (タグ v<MAJOR>.<MINOR>.<PATCH>) に置く
(spack のレシピの url)。

同じ中身から同じバイト列を作るため、次をそろえる (git や tar のコマンドの版によらない。Python の tarfile と gzip で作る):
  - ファイルは名前の順。ディレクトリの項目も入れる
  - 時刻はすべて SOURCE_DATE_EPOCH (既定 1767225600 = 2026-01-01 00:00:00 UTC)。gzip の頭の時刻は 0、元の名前は空
  - 所有者は 0 / 0 (名前なし)。モードは実行可能なら 0755、それ以外は 0644、ディレクトリは 0755
  - 形式は GNU tar、gzip の圧縮レベル 9
  gzip の圧縮結果は zlib の版で変わり得る (実際に変わることはまれ)。sha256 は配布する tarball そのものの値。

入れないもの: __pycache__ と *.pyc、.git、試験・ビルドの生成物 (bin/、buildlog/、regression_results/、perf_results/、
examples/build/、python/build/、python/parcel/libparcel_py.so・_build_config.py、tests/ の sub_*.sh)、*.tar.gz とその .sha256。

spack のレシピと sha256 の循環について:
  レシピ (spack/spack_repo/parcel/packages/{parcel_sycl,py_parcel_sycl}/package.py) は tarball の中に入るので、
  tarball の sha256 をそのままレシピに書くと tarball の中身が変わり、sha256 も変わる。そこで tarball に入れるときは
  レシピの version("X.Y.Z", sha256="...") の行を常に sha256 なしの version("X.Y.Z") に置き換える。こうすると --pin で
  手元のレシピに sha256 を書いた後に作り直しても、tarball (と sha256) は変わらない (--check で確かめられる)。
  tarball の中のレシピは sha256 を持たないので、tarball から展開したものは spack develop か、sha256 を書いた
  レシピ (公開するレシピのリポジトリなど) で使う。
"""
import argparse
import gzip
import hashlib
import io
import os
import re
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True       # パッケージの中に __pycache__ を作らない
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from check_version import read_header_version  # noqa: E402

DEFAULT_EPOCH = 1767225600          # 2026-01-01 00:00:00 UTC
RECIPES = ["spack/spack_repo/parcel/packages/parcel_sycl/package.py",
           "spack/spack_repo/parcel/packages/py_parcel_sycl/package.py"]
# レシピの版の行。sha256 の有無どちらにも当たる (各レシピにちょうど 1 行)
VERSION_RE = re.compile(r'^(?P<indent>[ \t]*)version\("(?P<ver>[^"]+)"(?:, sha256="(?P<sha>[0-9a-f]{64})")?\)[ \t]*$', re.M)

EXCLUDE_DIRS = {"__pycache__", ".git", "bin", "buildlog", "regression_results", "perf_results"}
EXCLUDE_PATHS = {"examples/build", "python/build", "python/parcel/libparcel_py.so", "python/parcel/_build_config.py"}


def release_name():
    major, minor, patch = read_header_version(ROOT)
    return "parcel_sycl-%d.%d.%d" % (major, minor, patch)


def excluded(rel):
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]) or parts[-1] in EXCLUDE_DIRS:
        return True
    if any(rel == p or rel.startswith(p + "/") for p in EXCLUDE_PATHS):
        return True
    base = parts[-1]
    if base.endswith(".pyc") or base.endswith(".tar.gz") or base.endswith(".tar.gz.sha256"):
        return True
    if parts[0] == "tests" and re.match(r"sub_.*\.sh$", base):
        return True
    return False


def file_list():
    """tarball に入れるファイル (パッケージのルートからの相対パス、名前の順)"""
    out = []
    for d, dirs, files in os.walk(ROOT):
        rel_d = os.path.relpath(d, ROOT)
        rel_d = "" if rel_d == "." else rel_d.replace(os.sep, "/") + "/"
        dirs[:] = sorted(x for x in dirs if not excluded(rel_d + x))
        for f in files:
            rel = rel_d + f
            if os.path.islink(os.path.join(ROOT, rel)):
                sys.exit("シンボリックリンクは入れられない: %s" % rel)
            if not excluded(rel):
                out.append(rel)
    return sorted(out)


def unpin(text, path):
    """レシピの version 行を sha256 なしの形にする"""
    if len(VERSION_RE.findall(text)) != 1:
        sys.exit("%s: version(...) の行が 1 行ではない" % path)
    return VERSION_RE.sub(lambda m: '%sversion("%s")' % (m.group("indent"), m.group("ver")), text)


def content_of(rel):
    data = open(os.path.join(ROOT, rel), "rb").read()
    if rel in RECIPES:
        data = unpin(data.decode("utf-8"), rel).encode("utf-8")
    return data


def build(out_path, epoch):
    name = release_name()
    files = file_list()
    dirs = sorted({"/".join(f.split("/")[:i]) for f in files for i in range(1, f.count("/") + 1)})

    def info(arcname, isdir, size=0, exe=False):
        ti = tarfile.TarInfo(arcname)
        ti.mtime = epoch
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        if isdir:
            ti.type, ti.mode = tarfile.DIRTYPE, 0o755
        else:
            ti.size, ti.mode = size, (0o755 if exe else 0o644)
        return ti

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tf:
        entries = [(name, None)] + [(name + "/" + d, None) for d in dirs] + [(name + "/" + f, f) for f in files]
        for arc, rel in sorted(entries, key=lambda e: e[0]):
            if rel is None:
                tf.addfile(info(arc, True))
            else:
                data = content_of(rel)
                exe = os.access(os.path.join(ROOT, rel), os.X_OK)
                tf.addfile(info(arc, False, len(data), exe), io.BytesIO(data))
    with open(out_path, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=9, mtime=0) as gz:
            gz.write(raw.getvalue())
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    return sha, len(files)


def pinned_shas():
    out = {}
    for rel in RECIPES:
        m = VERSION_RE.findall(open(os.path.join(ROOT, rel), encoding="utf-8").read())
        if len(m) != 1:
            sys.exit("%s: version(...) の行が 1 行ではない" % rel)
        out[rel] = (m[0][1], m[0][2])          # (版, sha256 または "")
    return out


def pin(sha):
    for rel in RECIPES:
        p = os.path.join(ROOT, rel)
        text = open(p, encoding="utf-8").read()
        if len(VERSION_RE.findall(text)) != 1:
            sys.exit("%s: version(...) の行が 1 行ではない" % rel)
        new = VERSION_RE.sub(lambda m: '%sversion("%s", sha256="%s")' % (m.group("indent"), m.group("ver"), sha), text)
        with open(p, "w", encoding="utf-8") as f:
            f.write(new)
        print("  %s: sha256 を書いた" % rel)


def main():
    ap = argparse.ArgumentParser(description="配布用の tarball と sha256 を作る")
    ap.add_argument("--outdir", default=os.path.dirname(ROOT), help="出力先 (既定はパッケージの親のディレクトリ)")
    ap.add_argument("--pin", action="store_true", help="spack のレシピ 2 本の version() に sha256 を書く")
    ap.add_argument("--check", action="store_true", help="レシピの sha256 と今のパッケージの tarball を照合する (何も書かない)")
    ap.add_argument("--list", action="store_true", help="入れるファイルの一覧を出す")
    a = ap.parse_args()
    if a.list:
        print("\n".join(file_list()))
        return 0
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_EPOCH))
    name = release_name()
    if a.check:
        with tempfile.TemporaryDirectory() as td:
            sha, n = build(os.path.join(td, name + ".tar.gz"), epoch)
        bad = 0
        for rel, (ver, got) in pinned_shas().items():
            st = "OK" if got == sha else ("未記入" if not got else "NG")
            bad += st != "OK"
            print("%-4s %s: version(\"%s\") sha256=%s" % (st, rel, ver, got or "-"))
        print("今のパッケージの %s.tar.gz (%d ファイル): sha256 %s" % (name, n, sha))
        return 0 if bad == 0 else 1
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, name + ".tar.gz")
    sha, n = build(out, epoch)
    with open(out + ".sha256", "w") as f:
        f.write("%s  %s\n" % (sha, os.path.basename(out)))
    print("%s (%d ファイル, %d バイト)" % (out, n, os.path.getsize(out)))
    print("sha256 %s  (%s.sha256)" % (sha, os.path.basename(out)))
    if a.pin:
        pin(sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
