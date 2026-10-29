[English](README.md) | 日本語

# 性能試験

PARCEL_SYCL v1.2 の性能試験。回帰試験 (`../regression/`) が「正しさ」を見るのに対し、こちらは「速さ」を見る。
各ソルバを収束まで解いた時間を、装置数 4 / 8 / 16 で測る。
実行ファイルは回帰試験のビルド (`tests/regression/sub_build_<plat>.sh`) が作る `bin/<plat>/verify_<精度>` を使う
(コンパイル時の NX は実行時オプション `PARCEL_NX` の既定値にしか使われないので、同じ実行ファイルで測れる)。

## 使い方

パッケージのルートで行う。環境ごとの設定は `tests/site_config.py` の `PERF` (キュー・資源・並列配置・walltime) にある。

```sh
python3 tests/regression/gen_regression.py --plat <plat> --lang cpp   # ビルドスクリプトを作る (回帰試験と共通)
bash tests/regression/sub_build_<plat>.sh                             # bin/<plat>/verify_* を作る

python3 tests/perf/gen_perf.py [--nx 512] [<plat> ...]     # tests/perf/sub_perf_<ソルバ>_d<装置数>_<plat>.sh
python3 tests/perf/gen_perf_prec.py [<plat> ...]           # tests/perf/sub_perf_d<装置数>_<plat>_prec_<精度>.sh と sub_build_prec_<plat>.sh
bash tests/perf/sub_build_prec_<plat>.sh                     # 混合精度の実行ファイル bin/<plat>/perf_prec/verify_<精度> を作る
for f in tests/perf/sub_perf_*_<plat>.sh tests/perf/sub_perf_*_<plat>_prec_*.sh; do qsub $f; done
python3 tests/perf/parse_perf.py . [--csv perf.csv]        # 集計 (各 3 回の最速)
python3 tests/perf/parse_perf_prec.py . [--md prec.md]     # 混合精度と倍精度の比較
```

投入できる本数に上限がある場合は `bash tests/regression/feed_regression.sh <plat> <上限> perf` (混合精度は `prec`) か
`reg_drive.sh` の種別 `perf` / `prec` を使う。`--nx` を 512 以外にすると、スクリプト名と出力先に `_nx<NX>` が付く
(投入は `feed_regression.sh <plat> <上限> perf _nx<NX>`、集計は `parse_perf.py . --dir perf_results_nx<NX>`)。

## 測定条件

| 項目 | 設定 |
|---|---|
| 問題 | 3 次元 7 点の格子 NX=512 (n = 134,217,728)、b = 1、x0 = 0、倍精度 |
| 行列 | 非対称向けのソルバ (pgmres、pcagmres) は移流拡散 (`PARCEL_MATRIX=convdiff PARCEL_CONVDIFF_C=0.05`)、pcg・pbicgstab と固有値ソルバはポアソン |
| 収束 | 相対残差 1e-8 まで、最大 10,000 反復 |
| 設定 | ソルバのパラメータはライブラリの既定 (`PARCEL_LIB_DEFAULTS=1`)。明示するのは下の表の値だけ |
| 回数 | 各構成 3 回。集計は総時間が最短の回 |
| 装置数 | 4 / 8 / 16 (CPU はソケット、GPU は 1 台) |

| ジョブ | 構成 (ラベル) | 前処理 | 行列 |
|---|---|---|---|
| `pcg` | `pcg_p{0,4}_*` | なし / ヤコビ | ポアソン |
| `pbicgstab` | `pbicgstab_p{0,4}_*` | なし / ヤコビ | ポアソン |
| `pgmres` | `cd0.05_pgmres_m60_p{0,4}_*` (再始動 m=60) | なし / ヤコビ | 移流拡散 |
| `pcagmres_explicit` | `cd0.05_pcagmres_explicit_s8t8_p{0,4}_*` (s = t = 8、Q を陽に作る) | なし / ヤコビ | 移流拡散 |
| `pcagmres_implicit` | `cd0.05_pcagmres_implicit_s8t8_p{0,4}_*` (s = t = 8、Q を作らない CholeskyQR 版) | なし / ヤコビ | 移流拡散 |
| `power` | `power_p0_NONE` (最大 20,000 反復) | なし | ポアソン |
| `inverse_power` | `inverse_power_p0_NONE` (外側最大 200、内部ソルバは pcg・ヤコビ前処理 (`PARCEL_LINSOLVE_SOLVER=pcg PARCEL_LINSOLVE_PRECON=4`)、内部の閾値 1e-10・最大 5,000) | なし | ポアソン |
| `ks_arnoldi` | `ks_arnoldi_p0_NONE` (m = 40、k_keep = 10、nev = 1、再始動最大 200) | なし | ポアソン |

- pbicgstab は移流拡散の前処理なし NX=512 で破綻する (丸め誤差の増幅による手法の性質で、PETSc の BiCGStab でも同じ) ので、
  ポアソンで測る。
- 固有値ソルバを前処理なしに限るのは、前処理を付けると A M⁻¹ の固有値という別の問題を解くことになるため。
- `ks_arnoldi` の直交化はライブラリの既定 (CGS2) で測る。ただし PS-A (`ps_a`) は MGS2 (`PARCEL_KS_QR=mgs`。検証サンプルが
  `set_qr_algo(QR_MGS)` を呼ぶ) で測る (`ps_a` のビルドは既定も MGS2 (`-DPARCEL_KS_ORTHO_DEFAULT=mgs`) だが、実行時オプションでも明示する)。
  1 スレッドあたりの局所長が小さく大きなキャッシュを持つ CPU では MGS2 が速いことがあるため。環境ごとに足す実行時オプションは
  `site_config.py` の `PERF` の `job_args` (`{ジョブ名: 実行時オプション}`) に書く (1 回目の出力で
  `[LIB_DEFAULTS]` の行の `qr = MGS (env)` を確かめる)。
- 各構成の 1 回目の出力で、実行時オプションが効いているかを `[LIB_DEFAULTS]` の行で確かめる (外れたら `PARAM_NG` で止める)。
  性能測定 (`PARCEL_VERIFY=0`) は行列名を出力しないので、NX=512 では残差の履歴の先頭 4 桁で移流拡散かポアソンかを
  ゆるく照合する (外れても測定は止めず、`param_check.txt` に `soft-NG` を残す)。

## 混合精度の比較 (`gen_perf_prec.py`)

同じ条件・並列配置で、実行ファイルだけを `bin/<plat>/perf_prec/verify_<精度>` に替えて、ヤコビ前処理の pcg (ポアソン) と
pgmres m=60 (移流拡散) を測る。既定の精度は `precon_dataF` (前処理行列・前処理ベクトルのデータを単精度にする)。
`--prec precon_dataF,precon_dataF16` のように回帰試験の精度構成の名前を並べられる。

実行ファイルは `gen_perf_prec.py` が書く `tests/perf/sub_build_prec_<plat>.sh` で作る (回帰試験のビルド
`sub_build_<plat>.sh` と同じ環境・コンパイル行・精度フラグに、`gen_perf_prec.py` の `PERF_EXTRA_FLAGS` を足したもの)。
**`precon_dataF` はヤコビ前処理のその場計算なし (`-DPARCEL_JACOBI_ONTHEFLY=0`) で作る。** 既定 (その場計算あり、
`PARCEL_JACOBI_ONTHEFLY=1`。マニュアルの「メモリと速さの選択」、`doc/ja/user/kernel-opt.html#perf-memory-options`) では、前処理の行列を単精度にしても前処理のたびに元の行列 A の
倍精度の値を読むので、データを単精度にする利点が出にくい。その場計算なしでは前処理の行列 (D⁻¹)(L+U) を単精度で保存して
読む。そのぶんメモリが増える (A の非零 1 個あたり 4 バイト。NX=512 の 7 点の格子で装置数 4 なら 1 台あたり約 0.9 GB)。
比べる相手の倍精度 (`gen_perf.py` の結果、`bin/<plat>/verify_double`) と回帰試験の `precon_dataF`
(`bin/<plat>/verify_precon_dataF`) は既定 (その場計算あり) のまま。ビルドの前処理後のマクロは
`bin/<plat>/perf_prec/macro_<精度>.txt` に残る (`#define PARCEL_JACOBI_ONTHEFLY 0`)。
ラベルは `<倍精度のラベル>_<精度>` で、出力先は性能試験と同じ。`parse_perf_prec.py` が同じ環境・同じ装置数の
倍精度 (`gen_perf.py` の結果) と比べ、総時間の比 (反復数の違いを含む) と 1 反復の比を出す。対象の環境は `PERF_PREC_PLATS`。

## 並列度

| 環境の例 | 1 装置 | np (装置数 4 / 8 / 16) | ノード数 |
|---|---|---|---|
| `s86_cpu` | 1 ソケット = 4 ランク x 5 スレッド | 16 / 32 / 64 | 2 / 4 / 8 |
| `s86_gpu` | V100 1 台 = 1 ランク x 12 スレッド | 4 / 8 / 16 | 1 / 2 / 4 |
| `miy_c` | 1 ソケット = 4 ランク x 14 スレッド (1 NUMA に 1 ランク) | 16 / 32 / 64 | 2 / 4 / 8 |
| `miy_g` | GH200 1 台 = 1 ランク x 72 スレッド | 4 / 8 / 16 | 4 / 8 / 16 |
| `ps_a` | 1 ソケット = 9 ランク x 14 スレッド (1 NUMA に 3 ランク) | 36 / 72 / 144 | 2 / 4 / 8 |
| `ps_b` | MI300A 1 台 = 1 ランク x 6 スレッド | 4 / 8 / 16 | 1 / 2 / 4 |

- CPU は `mpirun --map-by ppr:<rpd>:socket:PE=<スレッド数> --bind-to core` でソケット単位に配置する。
- GPU は 1 ノードに載せるランク数を「np とノードあたりの台数の小さい方」にする
  (超えると GPU を掴めないランクが CPU で動き、集団通信が全体を律速する)。ランクごとに 1 台を割り当てる (`gpu_bind_*.sh`)。
- `gpu_bind_*.sh` はノード内の順番 (ローカルランク) で GPU を選ぶので、ジョブの装置がノードを他のジョブと分け合うと
  割り当てられていない GPU を使う。1 装置 = 1 チャンクの環境 (例 `ps_b`) は `place` で、1 ノードに収まるジョブを
  `pack`、収まらないジョブを `exclhost` (ノードを専有) にする。
- Intel MPI は `--map-by` を解釈しないので、`-ppn` で 1 ノードのランク数を決め、`I_MPI_PIN_DOMAIN` に NUMA ごとのマスクを
  並べて固定する (`gen_regression.py` の `PIN_SETUP`、例では `ps_a`)。
- `OMP_PROC_BIND` は設定しない (libomp が初期スレッドを 1 コアに固定し、それを継承したワーカースレッドの
  並列が効かなくなるため)。
- walltime は環境ごとの既定と、ソルバ・装置数ごとの上書き (`walltime_by`) で決め、`queues` の中で収まる最初のキューを使う。
  同梱の値は開発時の測定からの見積もりなので、環境に合わせて直す。Wisteria の 2 環境はこの条件で測っていないので `PERF` に入れていない。

## 実行時オプションはコマンドライン引数で渡す

```sh
mpirun -n 4 ./bin/<plat>/verify_double PARCEL_NX=512 PARCEL_SOLVER=pcg PARCEL_PRECON=4 PARCEL_LIB_DEFAULTS=1
```

検証サンプルは実行時オプションを環境変数でも受け付けるが、MPI 実装によっては環境変数が他のノードのランクに
届かず、そのランクだけが既定値で動いて破綻する。コマンドライン引数は必ず全ランクに届く。
環境変数は引数より優先されるので、ジョブの環境に `PARCEL_*` が残っているとジョブの先頭で知らせる。

## 結果

`perf_results/<plat>/<構成>_np<N>/` に出力される。

| ファイル | 内容 |
|---|---|
| `status.txt` | `OK` / `TIMEOUT` / `RUN_FAIL_<rc>` / `PARAM_NG` / `PIN_IGNORED` |
| `result<R>.txt` | rank 0 が回ごとに書く出力 (`PARCEL_OUTPUT`)。集計はこれを優先して読む |
| `run<R>.txt`, `time<R>.txt` | 各回の標準出力と経過時間 [秒] |
| `dev.txt` | 装置数 |
| `param_check.txt` | 1 回目の出力の照合 (`[LIB_DEFAULTS]` の行と行列) |
| `bindings.txt` | ランクがどのソケット・ノードに載ったか |

`parse_perf.py` は 3 回のうち総時間が最短の回について、総時間・反復数・1 反復あたりの時間 [ms] と、`init` / `loop` / `dot` /
`spmv` / `axpy` / `precon` / `qr` の内訳を拾う (`--csv`)。`status.txt` が OK でも収束していない構成は「未収束」と出す。
3 回の最長が最短の 1.5 倍を超える構成には `※` が付く。
`dot` は `MPI_Allreduce`、`spmv` は袖通信を含むので通信の性能を映し、`axpy` は計算だけなので、
通信の問題と計算の問題を切り分けられる。既定では内積の時間計測の前後でデバイスを同期しないので、`dot` に直前の
カーネル (MGS の axpy など) の時間が入る。内訳を正確に取るときは実行時に `PARCEL_TIMER_SYNC=1` を付ける
(ソルバ全体の時間は少し延びる)。

## 注意

- **同時に走るジョブが同じファイルに書かないようにしてある。** GPU の割り当てスクリプト (`gpu_bind_*.sh`) は
  ジョブごとに別名にしている。
- **実行ファイルの古さはヘッダも見る。** PARCEL はヘッダオンリーなので、`Include/` を直したら作り直す
  (ジョブは `Include/*.hpp` の方が新しいと警告を出す)。
- **済んだ構成は飛ばす。** walltime 切れなどで再投入すると残りだけを測る。測り直すときは `PARCEL_PERF_FORCE=1`。

## ファイル

| ファイル | 内容 |
|---|---|
| `gen_perf.py` | 性能試験のジョブスクリプトの生成器 (測定条件・構成の定義) |
| `gen_perf_prec.py` | 混合精度と倍精度の比較のジョブスクリプトとビルドスクリプト (`sub_build_prec_<plat>.sh`) の生成器 |
| `parse_perf.py` | 性能試験の集計 |
| `parse_perf_prec.py` | 混合精度と倍精度の比較の集計 |
