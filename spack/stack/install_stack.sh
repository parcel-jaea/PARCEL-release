#!/bin/bash
# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 日本原子力研究開発機構 システム計算科学センター
# PARCEL をゼロから入れる: spack の取得から LLVM / AdaptiveCpp / OpenBLAS / Python, PARCEL 本体,
# C / Fortran の API, Python パッケージまでを 1 本で行う。
#
#   使い方: bash spack/stack/install_stack.sh <環境> <入れ先> [並列数]
#     [並列数] を省くと envs/<環境>/site_env.sh の PARCEL_STACK_JOBS、それも無ければ 16
#     <環境>   envs/<環境>/spack.yaml を使う (s86 など)
#     <入れ先> spack 本体・設定・キャッシュ・インストール先をすべてここに置く (既存の spack や ~/.spack は使わない)
#     例: bash spack/stack/install_stack.sh s86 $HOME/parcel_stack 16
#
#   MPI・CUDA・ROCm・コンパイラは各環境の既存品を envs/<環境>/spack.yaml で外部パッケージとして登録している。
#   途中で止まったら同じコマンドで再実行すればよい (入ったものは使い回す)。
#   終わったら <入れ先>/setup.sh を読み込めば spack と環境が使える。
set -eo pipefail
SITE=${1:?環境名 (envs/ の下のディレクトリ名)}
PREFIX=${2:?入れ先のディレクトリ}

HERE=$(cd "$(dirname "$0")" && pwd)
PARCEL_SRC=${PARCEL_SRC:-$(cd "$HERE/../.." && pwd)}   # Include/, capi/, python/ があるディレクトリ
PARCEL_REPO=$HERE/../spack_repo/parcel
ENV_YAML=$HERE/envs/$SITE/spack.yaml

# 確認済みの版に固定する
SPACK_COMMIT=4d910ee0e86587069b6ec21123f8d70d63bbc992   # spack 1.1.0.dev0 (2025-08-27)
PACKAGES_COMMIT=bab6c661ed90e88c2936d6d039715278f0702170 # spack-packages (2025-08-18): hipsycl 24.10.0, llvm 18.1.8

[ -f "$ENV_YAML" ] || { echo "no such environment: $ENV_YAML" >&2; exit 1; }
for d in Include capi python; do
  [ -d "$PARCEL_SRC/$d" ] || { echo "PARCEL_SRC=$PARCEL_SRC に $d/ が無い" >&2; exit 1; }
done
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)

# 1. spack 本体
if [ ! -d "$PREFIX/spack/.git" ]; then
  git clone -q -c feature.manyFiles=true https://github.com/spack/spack.git "$PREFIX/spack"
fi
git -C "$PREFIX/spack" checkout -q "$SPACK_COMMIT"

# 2. 環境ごとの追加設定 (サイトの MPI・ROCm などの module, コンパイラ)。インストール中と利用時の両方で読む
: > "$PREFIX/site_env.sh"
[ -f "$HERE/envs/$SITE/site_env.sh" ] && cp "$HERE/envs/$SITE/site_env.sh" "$PREFIX/site_env.sh"
set +e; . "$PREFIX/site_env.sh"; set -e
JOBS=${3:-${PARCEL_STACK_JOBS:-16}}   # 同時コンパイル数 (メモリの少ない計算ノードでは site_env.sh で下げる)
echo "jobs: $JOBS"

# 3. 既存の設定 (~/.spack, /etc/spack) を読まない。設定・キャッシュは入れ先の中に置く
cat > "$PREFIX/setup.sh" <<EOF
# PARCEL の spack 環境を使う: . $PREFIX/setup.sh
. $PREFIX/site_env.sh
export SPACK_ROOT=$PREFIX/spack
export SPACK_USER_CONFIG_PATH=$PREFIX/user_config
export SPACK_USER_CACHE_PATH=$PREFIX/cache
export SPACK_DISABLE_LOCAL_CONFIG=1
. \$SPACK_ROOT/share/spack/setup-env.sh
spack env activate parcel
EOF
# 環境を有効にした後の設定 (spack で作ったコンパイラを PATH に置くなど)
if [ -f "$HERE/envs/$SITE/after_activate.sh" ]; then
  cp "$HERE/envs/$SITE/after_activate.sh" "$PREFIX/after_activate.sh"
  echo ". $PREFIX/after_activate.sh" >> "$PREFIX/setup.sh"
fi
export SPACK_ROOT=$PREFIX/spack SPACK_USER_CONFIG_PATH=$PREFIX/user_config SPACK_USER_CACHE_PATH=$PREFIX/cache
export SPACK_DISABLE_LOCAL_CONFIG=1
. "$SPACK_ROOT/share/spack/setup-env.sh"

# 4. パッケージのレシピ集 (builtin) を固定
spack repo update --scope site --commit "$PACKAGES_COMMIT" builtin
echo "builtin: $(git -C "$(spack location --repo builtin)" rev-parse HEAD)"

# 4b. 先に入れるコンパイラ (envs/<環境>/compilers.yaml があるとき。OS の GCC が古い環境で新しい GCC を作る)。
#   この版の spack は, 1 回の解決の中で「コンパイラを作ってそれで別のパッケージを作る」ことができないので,
#   別の環境 compilers で先にインストールし, 本体の環境ではインストール済みのものを使い回させる
if [ -f "$HERE/envs/$SITE/compilers.yaml" ]; then
  if spack env list | tr -s ' \t' '\n\n' | grep -qx 'compilers'; then
    cp "$HERE/envs/$SITE/compilers.yaml" "$(spack location -e compilers)/spack.yaml"
  else
    spack env create compilers "$HERE/envs/$SITE/compilers.yaml"
  fi
  spack -e compilers concretize -f | tail -3
  [ "${CONCRETIZE_ONLY:-0}" = 1 ] || spack -e compilers install -j "$JOBS" --fail-fast
fi

# 5. 環境を作る (定義の @PARCEL_SRC@ / @PARCEL_REPO@ を置き換える)
mkdir -p "$PREFIX/env_def"
sed -e "s#@PARCEL_SRC@#$PARCEL_SRC#g" -e "s#@PARCEL_REPO@#$(cd "$PARCEL_REPO" && pwd)#g" "$ENV_YAML" > "$PREFIX/env_def/spack.yaml"
if spack env list | tr -s ' \t' '\n\n' | grep -qx 'parcel'; then
  cp "$PREFIX/env_def/spack.yaml" "$(spack location -e parcel)/spack.yaml"   # 定義を更新した場合
else
  spack env create parcel "$PREFIX/env_def/spack.yaml"
fi

# 6. 解決してインストール
spack -e parcel concretize -f | tee "$PREFIX/concretize.txt" | tail -3
[ "${CONCRETIZE_ONLY:-0}" = 1 ] && { echo "CONCRETIZE_ONLY: $PREFIX/concretize.txt"; exit 0; }
spack -e parcel install -j "$JOBS" --fail-fast
# 環境ごとのインストール後の処理 (envs/<環境>/post_install.sh があるとき)
if [ -f "$HERE/envs/$SITE/post_install.sh" ]; then
  spack env activate parcel
  set +e; . "$HERE/envs/$SITE/post_install.sh"; set -e
fi
spack -e parcel find --format '{name} {variants} /{hash:7}' parcel-sycl py-parcel-sycl \
  | sed 's/ build_system=[^ ]*//; s/ kernel=auto//'
echo "done. 使うときは: . $PREFIX/setup.sh"
