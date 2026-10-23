[English](README.md) | 日本語

# 回帰試験

PARCEL_SYCL v1.2 の回帰試験。検証サンプル `tests/verify/parcel_krylov_sample_verify.cpp` を精度構成ごとに
ビルドし、ソルバ・行列・前処理・MPI 並列数を実行時オプションで切り替えて解き、参照解と比べる。
ジョブスクリプトは生成器 `gen_regression.py` が作る (条件を変えるときは生成器を直して作り直す)。

## 使い方

すべてパッケージのルートで行う (生成したスクリプトは `qsub` / `pjsub` を実行したディレクトリを
パッケージのルートとして使う)。MPI あり / なしと言語 (検証サンプル (C++) / C / Fortran / Python) は、生成・ビルド・投入・集計の
どの段階でも独立に選べる (下記「選んで流す」)。スケジューラの無い計算機は `run_local.sh` が流す (下記)。

1. **環境の設定。** `tests/site_config.py` の `PROJECT` (課金先)、`ENV_SETUP` (モジュールと環境変数)、
   `REGRESSION` (キュー・資源・コンパイルフラグ・起動コマンド) を自分の環境に合わせる。
   同梱の 17 環境 (`s86_cpu` / `s86_gpu` / `wis_ody` / `wis_aqu` / `miy_c` / `miy_g` / `ps_a` / `ps_b`、
   スケジューラの無い `mac`、ゼロから入れた spack 環境で s86 を使う `s86spack_cpu` / `s86spack_gpu`、同じく他の計算機で
   spack に入れた PARCEL を試す `miy_cspack` / `miy_gspack` / `ps_aspack` / `ps_bspack` / `wis_aquspack` / `wis_odyspack`) は例なので、
   近いものを写して書き換えるとよい。
2. **生成。**
   ```sh
   python3 tests/regression/gen_regression.py [環境 ...]     # 省略すると REGRESSION の全環境と、その MPI なしの構成
   python3 tests/regression/gen_regression.py --plat s86_cpu --mpi both --lang cpp,c,fortran,python   # 選んで作る
   ```
   `tests/regression/` に次ができる (何も選ばないときは前回の生成物をすべて消してから作り、選んだときは選んだ組の
   生成物だけを作り直す)。
   - `sub_build_<plat>.sh`: 13 精度の実行ファイル `bin/<plat>/verify_<精度>` と `bin/<plat>/ddcheck` を作る
   - `sub_regression_<plat>_<精度>_<分割>_np<N>.sh`: 試験ジョブ (1 環境あたり約 200 本)
   - `sub_regression_<plat>_ddcheck_np{1,4}.sh`: double-double の劣化検出
   - MPI なしの構成 `<plat>_nompi` の同じもの (下記「MPI なしの構成」)
   - `sub_build_lang_<plat>.sh` と `sub_regression_<plat>_lang_{c,f,py}_np<N>.sh`: C / Fortran / Python の API の試験
     (下記。`REGRESSION_LANG` / `REGRESSION_LANG_PY` の環境だけ)
3. **ビルド。** コンパイルは MPI も GPU も使わないので、ログインノードで実行してよい
   (A64FX のように計算ノードでビルドする環境はジョブとして投入する)。
   ```sh
   PARCEL_BUILD_PAR=6 bash tests/regression/sub_build_<plat>.sh   # 同時コンパイル数 (既定は精度の数)
   ```
   出来ている実行ファイルは飛ばすので、一部が失敗しても再実行すれば残りだけ作る。作り直すときは
   `bin/<plat>/` を消す。`PARCEL_BUILD_EXTRA` に追加のフラグを与えられる (例: `PARCEL_BUILD_EXTRA=-ffp-contract=off`。
   使ったフラグは `bin/<plat>/build_extra.txt` に記録し、違うフラグで作ったものが混ざる場合は止まる)。
   同じログインノードで複数の環境をビルドするときは `build_chain.sh` で 1 つずつ行う。
   コンパイルフラグは `site_config.py` の `target` と `extra` で決まる。V100 (sm_70) の例 `s86_gpu` には
   `-Xcuda-ptxas -maxrregcount=255` を付けている (V100 で推奨。ptxas のレジスタ割り当てで Jacobi 前処理などの
   読み出しが分かれるのを防ぐ。spill は出ず、結果は変わらない)。他の GPU の例には付けていない。
4. **投入。**
   ```sh
   for f in tests/regression/sub_regression_<plat>_*.sh; do qsub $f; done
   ```
   投入できる本数に上限がある場合は `feed_regression.sh` (1 回ぶん投入) と `reg_drive.sh` (繰り返し投入) を使う。
   ```sh
   nohup bash tests/regression/reg_drive.sh <plat> <同時に投入しておく本数> &
   ```
   試験ジョブは OK で済んだケースを飛ばすので、walltime 切れなどで再投入すると残りだけが走る
   (最初からやり直すときは `PARCEL_REG_FORCE=1`)。
5. **集計。**
   ```sh
   bash tests/regression/summarize.sh [<plat> ...]
   ```
   環境・精度ごとの status の件数、言語別 (C++ / C / Fortran / Python) の合計、OK 以外のケースの一覧を出す。

## 選んで流す (MPI あり / なし・言語)

MPI あり (`<plat>`) / なし (`<plat>_nompi`) と、言語 `cpp` (検証サンプル `verify_*` の全精度と ddcheck) / `c` / `fortran` / `python` を
任意に組み合わせて、生成・ビルド・投入・集計できる。ビルドも言語ごと・MPI の有無ごとに別なので、たとえば
`verify_*` を作らずに C の試験だけを流せる。結果は `regression_results/<plat>[_nompi]/<精度 | lang_c | lang_f | lang_py>/` に分かれる。

| 段階 | 選び方 |
|---|---|
| 生成 | `gen_regression.py --plat <環境,...> --mpi yes\|no\|both --lang cpp,c,fortran,python` (既定は全環境・両方・全言語)。選んだ組の生成物だけを作り直し、他のスクリプトは残す |
| ビルド | `cpp` は `sub_build_<plat>.sh`、`c` / `fortran` / `python` は `sub_build_lang_<plat>.sh [c] [fortran] [python]` (引数で言語を選ぶ。省略するとその環境の全言語)。MPI なしは `<plat>_nompi` のスクリプト |
| 投入 | `feed_regression.sh <plat> <上限> --mpi … --lang …` (選んだ組を順に流し、上限は全体で数える)。種別 `cpp` / `lang_c` / `lang_f` / `lang_py` で 1 組ずつでもよい。`reg_drive.sh` は 5 番目以降の引数をそのまま渡す |
| 集計 | `summarize.sh --mpi … --lang … [<plat> ...]` |

```sh
python3 tests/regression/gen_regression.py --plat s86_cpu --mpi no --lang c        # s86_cpu_nompi の C だけ作る
bash tests/regression/sub_build_lang_s86_cpu_nompi.sh c
bash tests/regression/feed_regression.sh s86_cpu 20 --mpi no --lang c
bash tests/regression/summarize.sh --mpi no --lang c s86_cpu
nohup bash tests/regression/reg_drive.sh s86_cpu 20 "" sel --mpi both --lang cpp,fortran &
```

種別ごとに投入済みの記録 (`feed_<種別>_<plat>.jobs` / `.state`) を分けているので、同じジョブを別の種別 (既定の `regression` は
`cpp` と言語別の両方を含む) で二重に投入しないこと。

## スケジューラの無い計算機 (`sched="local"`、例 `mac`)

`site_config.py` の `REGRESSION` で `sched="local"` にした環境は、ジョブスクリプトに資源の指定を書かず、`run_local.sh` が
bash でそのまま流す。ケース・判定・結果の置き場所は他の環境と同じ。同時に流すジョブは、使うコア数 (並列数 × スレッド数) の
合計が `--cores` (既定は論理 CPU 数) 以下になるように並べる。

```sh
python3 tests/regression/gen_regression.py --plat mac          # mac と mac_nompi
bash tests/regression/run_local.sh --plat mac --mpi both --lang cpp,c,fortran,python --build   # ビルド・実行・集計
```

- `--build` は `sub_build_<plat>.sh` と `sub_build_lang_<plat>.sh` を環境ごとに順に流す (同時に 2 つのビルドを走らせない)。
  同時コンパイル数は `PARCEL_BUILD_PAR` (例 `mac` の既定 4)。`--no-run` でビルドだけ、`--dry-run` で流すジョブの一覧だけ。
- ログは `tests/regression/local_logs/<plat>/<ジョブ名>.log`。済んだケースは飛ばすので、止めても同じコマンドで続きから流せる。
- 打ち切り時間 (`timeout`) を越えたときは、そのジョブの子だけを止める (同じ実行ファイルを使う並行のジョブを名前で探して止めない)。
- 例 `mac` (macOS + Homebrew の adaptivecpp・open-mpi・openblas・gcc。Python は numpy・scipy): 1 ランク 5 スレッド
  (`s86_cpu` と同じ。CPU のブロックヤコビの分け方をそろえて結果を照合できるようにする。np=4 ではコア数を超えるが、
  `run_local.sh` はコア数を超えるジョブを単独で流す)。macOS の Open MPI は
  プロセスをコアに固定できないので `--bind-to none` (`map_by=False`)。GNU の `timeout` が無いので `ENV_SETUP["mac"]` が同じ呼び方の
  関数を perl で用意する。`run_local.sh` などは macOS の `/bin/bash` (3.2) で動く。Homebrew の場所は `PARCEL_BREW` で変えられる。

## 試験の内容

| 項目 | 内容 |
|---|---|
| 問題の大きさ | NX=8 (n=512)。全ケースで密行列の参照解を作れる |
| ソルバ | pcg / pbicgstab / pgmres / pcagmres / power / inverse_power / ks_arnoldi |
| 行列 | 3 次元 7 点ポアソン、3 次元移流拡散 (非対称、c=0.05)、移流拡散 c=2.0 (固有値が複素共役対。ks_arnoldi のみ) |
| 前処理 | なし / 点ヤコビ / ブロックヤコビ / ヤコビ |
| 並列数 | MPI 1 / 2 / 4 (np ごとに別ジョブ。資源指定を np に合わせる) |
| 精度構成 | 13 種 (下表) |
| リスタート周期 | ライブラリの既定 (pgmres 30、pcagmres (s, t) = (8, 4)、ks_arnoldi のリスタート上限 200)。例外は calcF16 × pgmres の m = 15 (下記) |
| その他 | A M⁻¹ の固有値モード (`inverse_power` と `ks_arnoldi` の shift-invert)、Q を陽に作らない `pcagmres` (`CAGMRES_Q_IMPLICIT`、CholeskyQR とモーメント法。calcF16 / dataF16 では対象外) |

移流拡散の行列は double でだけ試す (他の精度はポアソンのみ)。`pcg` と `inverse_power` (内部で対称正定値を
前提とする) はポアソンだけ。`ks_arnoldi` × ブロックヤコビは重いので double でだけ試す。

反復の上限は線形ソルバ 10000、power 5000、inverse_power 200。power × ブロックヤコビ (A M⁻¹ の固有値) だけは 20000 にする。
ブロックの区切り位置 (CPU ではランクあたりスレッド数で決まる) で固有値の間隔が大きく変わり、5000 回に届く分割があるため。

リスタート周期は検証サンプルに `PARCEL_GMRES_RESTART=default` / `PARCEL_CAGMRES_S=default` / `PARCEL_KS_RESTART=default`
を渡し、`set_restart` などを呼ばずにライブラリの既定で試す (検証サンプルのコンパイル時の値 (pgmres 8、pcagmres (8, 1)、
ks_arnoldi 50) は使わない。性能試験はこちらの値を明示する)。周期の境目でしか収束を判定しないので、反復数は短い周期より
多く出ることがある。精度ごとに既定から変えるものは `gen_regression.py` の `STM_OVERRIDES` に書く。今は calcF16 × pgmres の
m = 15 だけで、m = 30 では残差が判定値 1e-3 のそばで停滞して反復数が大きく増える。

### 精度構成と閾値

| 精度 | コンパイル時マクロ | 収束判定 (外側・内部) | 誤差の合格閾値 |
|---|---|---|---|
| `double` | なし | 1e-9 | 1e-6 (power 1e-4) |
| `calcDD` / `dataDD` | 演算 (とデータ) を double-double | 1e-9 | 1e-6 (power 1e-4) |
| `calcF` / `dataF` | 演算 (とデータ) を単精度 | 1e-6 | 1e-4 (前処理付き power の一部 5e-4) |
| `calcF16` | 演算を半精度 | 1e-3 | 2e-2 |
| `dataF16` | 演算とデータを半精度 | 1e-2 | 5e-2 |
| `precon_calcF` / `precon_dataF` / `precon_allF` | 前処理の演算 / データ / 両方を単精度 | 1e-9 (ks_arnoldi は calc / all で 1e-6) | 1e-6 (power 1e-4) |
| `precon_calcF16` / `precon_dataF16` / `precon_allF16` | 前処理を半精度 | 1e-9 (calc / all は pbicgstab 1e-5、ks_arnoldi 1e-3) | 1e-6 (calc / all は pbicgstab・ks_arnoldi・power、data は ks_arnoldi・power を 1e-3) |

閾値はその精度で到達できる相対残差に合わせてある。上の表の値に加えて、低精度で残差が床に着いて破綻・停滞する
組合せだけ、`gen_regression.py` の `LOWPREC_OVERRIDES` で判定値を実測の床の 10 倍前後に緩めている (誤差の閾値も合わせる)。

| 精度 | ソルバ | 前処理 | 環境 | 収束判定 | 誤差の閾値 |
|---|---|---|---|---|---|
| `precon_calcF16` / `precon_allF16` | pcg / pgmres / pcagmres (implicit を含む) | 点ヤコビ | 全環境 | 1e-5 | 1e-4 |
| `precon_calcF16` / `precon_allF16` | inverse_power | 点ヤコビ | 全環境 | 内部 2e-5 | 1e-6 |
| `precon_calcF16` / `precon_allF16` | power | ブロックヤコビ / ヤコビ | 全環境 | 1e-5 | 2e-2 / 3e-3 |
| `precon_calcF16` / `precon_allF16` | A M⁻¹ モード (inverse_power・shift-invert) | 点ヤコビ / ブロックヤコビ / ヤコビ | 全環境 | 内部 5e-2 | — |
| `precon_dataF16` | A M⁻¹ モード (inverse_power) | ブロックヤコビ | 全環境 | 内部 3e-2 | — |
| `calcF16` | inverse_power・A M⁻¹ モード | 全種 | 全環境 | 内部 5e-2 | — |
| `calcF16` | pcagmres | 全種 | 全環境 | 1e-2 | — |
| `calcF16` | pbicgstab | 点ヤコビ / ブロックヤコビ | 全環境 | 3e-2 | 1e-1 |
| `calcF16` | pcg | 点ヤコビ | 全環境 | 1e-2 | 2e-2 |
| `calcF16` / `dataF16` | pcg | ブロックヤコビ | 全環境 | 1e-1 | 2e-1 |

「—」は上の表の値のまま。最後の 3 行は GPU で破綻・停滞する組合せで、CPU (半精度をエミュレートしていて床が低い) でも
ブロックヤコビの分割 (ランクあたりのスレッド数) によっては破綻するので、CPU と GPU で同じ判定値を当てる。
最後の行は実用的な精度に届かず、動作の確認に留まる。

- `dataF16` はベクトルも半精度で持つので、残差が 1e-2 付近で丸めと同程度になり、漸化式の残差が真の残差から離れる。
- 半精度の前処理は相対残差 ~1e-6 を下回ると (特に pbicgstab で) NaN で破綻しうる。
- 前処理付きの固有値の参照解 A M⁻¹ 自体が前処理の丸めを受けるので、半精度の前処理では誤差の閾値を
  半精度の相対精度 (~1e-3) に合わせる。
- 半精度のデータ・演算精度 (`calcF16` / `dataF16`) の `pcagmres` は Q を陽に作る版 (explicit) を対象とし、
  CholeskyQR 系 (implicit・モーメント法) は対象外とする (s, t によらず NaN か未収束になる)。
- 内部ソルバ (inverse_power の内部ソルバ、shift-invert の内部 pgmres) の閾値は外側と同じにする
  (A M⁻¹ モードと上の個別の判定値を除く)。
  対称行列のレイリー商は固有ベクトルの誤差の 2 乗の精度を持つので、内部を外側より厳しくする必要はない。
  逆に内部を緩めすぎると、精度の低い答えを「収束した」と返す。

## 検証方法

| 対象 | 参照解 |
|---|---|
| 線形ソルバ (ポアソン) | 製造解 と 密行列 `dgesv` による直接解 |
| 線形ソルバ (移流拡散) | 密行列 `dgesv` による直接解 |
| 固有値ソルバ (前処理なし) | 解析固有値 |
| 固有値ソルバ (点ヤコビ) | 解析固有値を 1/d 倍 (対角が一定値 d のとき) |
| 固有値ソルバ (ブロックヤコビ・ヤコビ、A M⁻¹ モード) | 密行列 A M⁻¹ の `dgeev` (PARCEL の前処理を単位ベクトルに適用して組む) |
| `inverse_power`, shift-invert | 解析固有値 (前処理によらず λ(A)) |

## 結果

`regression_results/<plat>/<精度>/<ケース>_np<N>/` に出力される。`result.txt` が標準出力、
`parcel_result.txt` が rank 0 が書く同じ内容 (標準出力をジョブログへ流す MPI 向け)、
`run_info.txt` が実行したノードと時刻、`status.txt` が判定である。

| `status.txt` | 意味 |
|---|---|
| `OK` | 実行が成功し、収束し、参照解との誤差が閾値以下 |
| `NOT_CONVERGED` | 収束しなかった (固有値ソルバの `converged = NO`、線形ソルバの最終残差 > tol、最後に解いたソルバの収束状態が `DIVERGED_*`) |
| `INNER_NOT_CONVERGED` | 固有値ソルバの内部の線形ソルバが反復の上限に達した (`linsolve_failures` が 1 以上)。外側が収束していても OK にしない |
| `ERROR_TOO_LARGE` | 参照解との誤差が閾値を超えた |
| `NONFINITE` | 解・固有値・残差に NaN / Inf が出た |
| `VERIFY_SKIPPED` | 参照解を作れず精度を検証していない |
| `NO_VERIFY_OUTPUT` | 検証の出力が無い (出力を取り損ねている) |
| `RUN_FAIL_<rc>` | 実行時エラー (`RUN_FAIL_3` は版の不一致: 検証サンプルが最初に出す `[VERSION] PARCEL <版>` が、生成器が `Include/parcel_version.hpp` から与える `PARCEL_EXPECT_VERSION` と違うと試験をせずに終わる) |
| `TIMEOUT` | 打ち切り (ks_arnoldi・power は 300 秒、他は 120 秒) |
| `BUILD_MISSING` | 実行ファイルが無い (先にビルドする) |

線形ソルバの収束は、停止条件に使う漸化式の残差 (反復履歴の最後の `N : residual norm`) で判定する。
真の残差 `b-Ax` は低精度の構成で漸化式の残差と離れ、pcg では解の更新をまとめる間隔 (`PARCEL_PSTEP`、既定 1) を
2 以上にするとその間隔ごとにしか意味を持たないので、判定には使わない。解そのものの精度は参照解との誤差で見る。

### double-double の劣化検出 (`ddcheck`)

double-double の構成 (`calcDD` / `dataDD`) の合格閾値は double の水準なので、コンパイラの最適化などで
double-double が黙って double に落ちても合格してしまう。`ddcheck/ddcheck.cpp` は、ソルバの dd 構成と同じ
`parcel::blas` の経路 (dot・axpy・gemv・SpMV・残差計算。MPI の集約・袖通信を含む) に、厳密値が分かっていて
double では表せない入力を与え、相対誤差が 1e-28 以下かを見る。`[DDCHECK] RESULT PASS` のときだけ `OK`、
FAIL なら `DD_DEGRADED`。`sub_build_<plat>.sh` が calcDD と同じフラグで `bin/<plat>/ddcheck` を作る。
`gemv_coef_lo` (係数ベクトルが dd の gemv) は既定では合否に数えない (`DDCHECK_STRICT_GEMV=1` で数える)。

## MPI なしの構成 (`<plat>_nompi`)

`-DPARCEL_MPI=0` でビルドした実行ファイル (MPI を読まず、リンクもしない) を、`mpirun` / `mpiexec` を
使わずに 1 プロセスで直接起動する構成も試験する。`tests/site_config.py` の `REGRESSION_NOMPI` に挙げた環境
(既定は `REGRESSION` の全環境) ごとに、環境名 `<plat>_nompi` のスクリプトを作る。設定は元の環境から派生させ、
違うのは次の点だけである。

| 項目 | MPI なしの構成 |
|---|---|
| コンパイラ | `ENV_SETUP` の `CXX=` の行を `NOMPI_CXX` (既定 `acpp`) に置き換える。MPI のラッパは通さない |
| コンパイルフラグ | 元の環境のフラグに `-DPARCEL_MPI=0` を足す |
| 実行ファイル | `bin/<plat>_nompi/verify_<精度>` と `bin/<plat>_nompi/ddcheck` (`sub_build_<plat>_nompi.sh` が作る) |
| 並列数 | 1 だけ (試験ジョブは `_np1` のみ、ddcheck も np=1 のみ)。資源の指定は元の環境の np=1 と同じ |
| 起動 | 実行ファイルを直接起動する。GPU は割り当てスクリプトを通さず、MPI あり np=1 のランク 0 と同じ装置 (`CUDA_VISIBLE_DEVICES=0` / `HIP_VISIBLE_DEVICES=0`) を使う。ランクの NUMA 固定 (`pin`) は使わない |
| ケース・判定 | 元の環境の np=1 と同じ |
| 結果 | `regression_results/<plat>_nompi/` |

ビルド・投入・集計は環境名に `<plat>_nompi` を指定するだけで、MPI ありの環境と同じ手順になる
(`feed_regression.sh <plat>` は `<plat>_nompi` のスクリプトを含めない)。

```sh
bash tests/regression/sub_build_s86_cpu_nompi.sh
nohup bash tests/regression/reg_drive.sh s86_cpu_nompi <同時に投入しておく本数> &
bash tests/regression/summarize.sh s86_cpu_nompi
```

MPI なしと MPI あり np=1 は同じ計算をするので、結果は一致するはずである。`compare_nompi.py` で突き合わせる。

```sh
python3 tests/regression/compare_nompi.py s86_cpu [...] [--md compare_nompi.md]   # 両方の結果があるホストで
python3 tests/regression/compare_nompi.py --csv iters_*.csv                     # collect_iters.sh の CSV から
```

`regression_results/<plat>/.../<ケース>_np1` と `regression_results/<plat>_nompi/.../<ケース>_np1` (ddcheck を含む) を組にし、
状態・反復数 (反復履歴の最後の番号、外側の反復数・リスタート数)・数値 (参照解との誤差、真の残差、固有値、`[DDCHECK]` の行)・
出力全体 (時間の行と AdaptiveCpp の警告を除く) が一致するかを数え、違う組を並べる。状態・反復数・数値が 1 組でも
違えば終了コード 1。CSV から比べる場合は状態と反復数だけを見る。

## C / Fortran / Python の API の試験 (`lang_c` / `lang_f` / `lang_py`)

`examples/sample_c.c` / `examples/sample_f.f90` (C / Fortran の API から全ソルバを呼び、結果を独立に検証する例) と
`tests/regression/sample_py.py` (同じ問題・項目・判定を Python のパッケージ `parcel` から行う) をそのまま回帰試験に載せる。
対象は `tests/site_config.py` の `REGRESSION_LANG` (C / Fortran) と `REGRESSION_LANG_PY` (Python) に挙げた環境と、その MPI なしの構成。

| 項目 | 内容 |
|---|---|
| ビルド | `sub_build_lang_<plat>.sh [c] [fortran] [python]` が `bin/<plat>/sample_c`・`sample_f`・`libparcel_py.so` を作る (ログインノードで bash で実行してよい。引数を省くとその環境の全言語)。フラグは `verify_*` と同じ (`target`・`extra`・`PARCEL_BUILD_EXTRA`、MPI なしは `-DPARCEL_MPI=0`)。C / Fortran はラッパ `capi/wrapper_parcel_sycl.cpp` を SYCL コンパイラで、main を `LANG_MPI_CC` / `LANG_MPI_FC` (MPI なしは `LANG_NOMPI_CC` / `LANG_NOMPI_FC`) で作り、SYCL コンパイラでリンクする (`examples/build_examples.sh` と同じ形)。MPI なしの Fortran は `examples/sample_f_mpi_stub.f90` と組む。Python はネイティブ層 `python/src/parcel_py.cpp` を共有ライブラリにし (pip は使わない)、numpy と scipy の入った Python (`PARCEL_REG_PY`、既定 `python3`) を `bin/<plat>/lang_python.txt` に記録する |
| ジョブ | `sub_regression_<plat>_lang_{c,f,py}_np{1,2,4}.sh` (MPI なしは `_np1` だけ)。資源・起動・スレッド数は他の試験と同じ。NX はサンプルの既定の 16。Python はパッケージの `python/` と `bin/<plat>/libparcel_py.so` を読む (`PARCEL_TEST_FROM_SOURCE=1`、`PARCEL_PY_LIB`) |
| 判定 | 1 項目 = 1 ケース。`regression_results/<plat>/lang_<c\|f\|py>/<項目>_np<N>/status.txt` に PASS なら `OK`、FAIL なら `FAIL`。項目名は記号を `_` にしたもの (`pcagmres(impl,sym)` → `pcagmres_impl_sym`)。最後の `ALL PASS` の行と終了コード 0 を 1 ケース (`all`) と数えるので、1 回の実行で C / Fortran は 13 ケース (先頭の `version` は版の照合: `parcel_get_version()` と `parcel_version_string()` が同じ版で、生成器が `Include/parcel_version.hpp` から与える `PARCEL_EXPECT_VERSION` と一致するか)。Python は旧 API の項目の代わりに設定を省いた `parcel.solve` (`pcg_defaults`) を見て、先頭にネイティブ層の MPI の有無 (`has_mpi`) と、GPU の環境 (`py_device` を書いた環境) ではデバイス名 (`device`) の確認が付く。`version` は `parcel.library_version()` (ネイティブ層) と `parcel.__version__` と `PARCEL_EXPECT_VERSION` の一致 (14 ケース、GPU は 15)。出力に無い項目 (途中で落ちた) は `TIMEOUT` / `RUN_FAIL_<rc>` / `NO_ITEM_OUTPUT` |
| 出力 | 出力全体は `lang_<言語>/_run_np<N>/result.txt` (status.txt を置かないので件数に入らない)。各ケースの `result.txt` は該当の行だけ |

```sh
PARCEL_REG_PY=<numpy と scipy の入った python> bash tests/regression/sub_build_lang_s86_cpu.sh
bash tests/regression/feed_regression.sh s86_cpu 20 lang     # bin/s86_cpu/lang_built.txt を待って投入 (1 言語だけなら lang_c / lang_f / lang_py)
bash tests/regression/summarize.sh s86_cpu                   # lang_c / lang_f / lang_py の行と、言語別の合計
```

`feed_regression.sh <plat>` (既定の regression) にも `lang` のジョブは含まれる (ddcheck と同じ)。先に `sub_build_lang_<plat>.sh` を
済ませておくこと (無ければ各ケースが `BUILD_MISSING` になる)。開発時に C / Fortran / Python の試験を確認していない Wisteria の 2 環境は
`REGRESSION_LANG` / `REGRESSION_LANG_PY` に入れていない (spack に入れた PARCEL を使う `wis_aquspack` / `wis_odyspack` は入れている)。

## spack の環境で流す (`s86spack_cpu` / `s86spack_gpu`、`miy_cspack` など)

`spack/stack/install_stack.sh` でゼロから入れた spack の環境 (`spack/README.ja.md`) のコンパイラ・ライブラリで試験を流す例。
キュー・資源・フラグ・ケースは `s86_cpu` / `s86_gpu` と同じで、ジョブの環境設定 (`ENV_SETUP["s86spack"]`) だけが違う。

```sh
python3 tests/regression/gen_regression.py --plat s86spack_cpu,s86spack_gpu --mpi both
export PARCEL_STACK=$HOME/parcel_stack                     # install_stack.sh の入れ先 (既定は site_config.py の SPACK_STACK)
bash tests/regression/sub_build_s86spack_cpu.sh            # スクリプトが自分で spack の環境を読む
bash tests/regression/sub_build_lang_s86spack_cpu.sh       # Python は spack の環境の python3 (numpy・scipy 入り)
for f in tests/regression/sub_regression_s86spack_cpu_*.sh; do qsub -v PARCEL_STACK $f; done
```

- ビルドのスクリプトとジョブは `. ${PARCEL_STACK}/setup.sh` で spack の環境を有効にし、`spack load hipsycl openblas openmpi` で AdaptiveCpp・LLVM・
  OpenBLAS・Open MPI を読む。LAPACK は spack の OpenBLAS (`${openblas_root}`) をリンクする。
- 何を試すかは `parcel_from` で選ぶ。
  - `"source"` (`parcel_from` の既定): パッケージのソースを spack のツールチェーンで組む。検証サンプル (C++) と ddcheck は
    `Include/` を、C / Fortran は `capi/` のラッパを、Python は `python/` を使う。
  - `"installed"`: spack に入れた PARCEL を試す。C / Fortran は `parcel-sycl` の `libparcel_sycl_c.so` / `libparcel_sycl_f.a` と
    ヘッダ・モジュールでリンクし、Python は `py-parcel-sycl` を `import parcel` で読む (`spack_parcel` に書いた構成と MPI の有無で
    `spack load` する)。検証サンプル (C++) と ddcheck は `parcel-sycl` のヘッダ (`$PARCEL_PREFIX/include`) で組む
    (サンプルは `Include/` を相対パスで読むので、ビルドのスクリプトが試験のソースを `buildlog/<plat>/installed_src/` に写し、
    その隣の `Include` を `$PARCEL_PREFIX/include` に向ける。使った場所は `bin/<plat>/parcel_prefix.txt`)。
- 同梱の例 (`s86spack_*` と、他の計算機の `miy_cspack` / `miy_gspack` / `ps_aspack` / `ps_bspack` / `wis_aquspack` / `wis_odyspack`) は `"installed"` で、
  spack に入れた PARCEL そのものを試す。キュー・資源・フラグ・ケースは元の環境 (`s86_cpu`・`miy_c` など) と同じ。s86 以外の環境設定は `site_config.py` の `_stack_env`
  (`setup.sh` がサイトの module を読み、`spack load hipsycl openblas`)、入れ先は `SPACK_STACK_SITE` (ジョブの `PARCEL_STACK` が優先)。
  ```sh
  python3 tests/regression/gen_regression.py --plat miy_cspack --mpi both
  bash tests/regression/sub_build_miy_cspack.sh; bash tests/regression/sub_build_lang_miy_cspack.sh   # MPI なし (_nompi) も同様
  nohup bash tests/regression/reg_drive.sh miy_cspack 4 "" sel --mpi both --lang cpp,c,fortran,python &
  ```
- Wisteria-Odyssey (`wis_odyspack`、A64FX) は他と次の点が違う。環境設定は `ENV_SETUP["odyspack"]` (`_stack_env` ではない)。
  - ビルドは計算ノードで行う (ログインノードは x86)。`pjsub tests/regression/sub_build_wis_odyspack.sh` (と `_nompi`、`sub_build_lang_*`)。
  - PARCEL は富士通のコンパイラ・MPI・LAPACK で作った `parcel-sycl+fj`。検証サンプルは `wis_ody` と同じく `mpiFCC` (AdaptiveCpp の
    ホスト側は FCC) で組み、C / Fortran の試験は `mpifcc` / `mpifrt` (MPI なしは `fcc` / `frt`) に `-Kopenmp` を付けて組む
    (インストールした Fortran のモジュールは frt で作ってあり gfortran では読めない。frt で `-Kopenmp` を付けずに組んだプログラムから
    PARCEL を複数スレッドで呼ぶと線形ソルバの結果が壊れる)。frt のモジュールの出力先は `-M` (`lang_fmod`)。
  - 富士通の `mpiexec` はランクの標準出力を、シェルのリダイレクト先 (`> result.txt`) ではなくジョブのログに書く。
    起動コマンドに `-std mpiexec_out.txt` を付け、実行後に `result.txt` に足す (`rank_out`)。C / Fortran / Python の試験は標準出力で判定するので要る。
  - 富士通 MPI は `mpiexec` を通さないと MPI を初期化できないので、言語別ビルドでの MPI ありの `import parcel` の確認も
    `mpiexec -n 1` を通す (`py_check_launch`)。
- 性能試験も同じ名前の環境 (`tests/site_config.py` の `PERF["s86spack_cpu"]` など) で流せる (他の計算機の spack 環境の `PERF` は無い)。

## 反復数の環境間比較 (報告のみ)

合否は誤差と収束だけで判定し、反復数の違いでは落とさない。低精度の構成では縮約の順序 (スレッド数・ランク数・
装置で変わる) による丸めの違いが増幅され、反復数が環境で変わるので、その大きさを別に報告する。

同じ環境でも、ランクが何ノードに分かれるかで縮約の順序が変わり、低精度の構成では反復数が変わる。そのため試験ジョブは
ランクを最小のノード数に詰めて置く。1 ランク 1 チャンクの PS-B (`ps_b` / `ps_bspack`。1 ノード = 4 APU、`node_chunks=4`) は、
np が 4 以下なら `place=pack` で 1 ノード、4 を超えれば `place=exclhost` (ホストを専有) と起動コマンドの `ppr:4:node` で
ceil(np/4) ノードに 4 ランクずつ置く (`tests/site_config.py` の `place`・`node_chunks`)。`place` を指定しないと、
キューの既定 (B_S は free) でチャンクが空いた vnode に散らばり、np=4 でも 2〜3 ノードにまたがることがある。

```sh
bash tests/regression/collect_iters.sh <plat> [...] > iters_<host>.csv          # 各ホストで
python3 tests/regression/compare_iters.py iters_*.csv --md compare_iters.md     # 手元で
```

## ファイル

| ファイル | 内容 |
|---|---|
| `gen_regression.py` | ジョブスクリプトの生成器 (精度構成・ケース・閾値・判定の定義) |
| `ddcheck/ddcheck.cpp` | double-double の劣化検出 |
| `sample_py.py` | Python の API の試験 (`examples/sample_c.c` に準じる。`lang_py`) |
| `feed_regression.sh` | 同時に投入しておく本数を守って 1 回ぶん投入する (`regression` / `cpp` / `lang` / `lang_c` / `lang_f` / `lang_py` / `dd` / `pspec` / `perf` / `prec`、または `--mpi` / `--lang`) |
| `reg_drive.sh` | `feed_regression.sh` を繰り返し呼ぶ |
| `build_chain.sh` | 複数環境のビルドを 1 つずつ行う |
| `summarize.sh` | 結果の集計 (環境・精度ごと、言語別。`--mpi` / `--lang` で絞れる) |
| `run_local.sh` | スケジューラの無い計算機でビルド・実行・集計する (`sched="local"`) |
| `collect_iters.sh`, `compare_iters.py` | 反復数の環境間比較 |
| `compare_nompi.py` | MPI なしの構成と MPI あり np=1 の突き合わせ |
