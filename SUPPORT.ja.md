[English](SUPPORT.md) | 日本語

# サポート

## 問い合わせ先

不具合の報告・質問は、公開用リポジトリの GitHub の Issues へ。

- 問い合わせ先: <https://github.com/parcel-jaea/PARCEL-release/issues>
- 配布元: <https://ccse2.jaea.go.jp/software/parcel/>
- 公開用リポジトリ: <https://github.com/parcel-jaea/PARCEL-release> (配布 tarball は Release に置く)

## 報告に書いてほしいこと

不具合の報告・質問には、次を添えてください。

- PARCEL_SYCL の版 (C++ は `parcel::version_string()`、C / Fortran は `parcel_version_string()`、
  Python は `parcel.__version__` と `parcel.library_version()`。例: `1.2.0`) と、配布 tarball の名前・sha256 (分かれば)
- 入れ方 (ソースから直接ビルド / spack の `parcel-sycl` / `py-parcel-sycl` / pip) と、ビルドのコマンド
  (`--acpp-targets`、`-D` で付けたマクロ。spack なら `spack spec` の出力と `share/parcel/build_config.txt`)
- 計算機・OS、CPU / GPU の型番
- SYCL の処理系 (AdaptiveCpp の版と、それを作った LLVM の版)、ホストのコンパイラ、CUDA / ROCm の版、MPI の実装と版
- 使ったソルバ・前処理・精度の設定、行列の大きさ (行数・非零数) とランク数・スレッド数
- 再現の手順 (できれば `examples/` か `tests/verify/` の検証サンプルで再現する最小の例) と、
  rank 0 の標準出力 (` converged_reason = ...` の行を含む) とエラーメッセージ

試験の流し方は `tests/regression/README.ja.md` (回帰試験) を参照。
