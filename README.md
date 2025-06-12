
# サンプルコード
3次元ポアソン方程式を前処理付き共役勾配法により解く.

# コンパイル方法
SYCL及びMPIライブラリをリンクしてコンパイルする.

- コマンド例
mpicxx -o a.out parcel_krylov_sample.cpp

# 実行方法
mpirun -n 4 ./run

# パッケージファイル
- Include : SYCL版PARCELソースコード
- parcel_default.hpp : 計算設定ファイル
- parcel_krylov_sample.cpp : c++サンプルコード
- poisson_cpu : c++サンプルコード行列作成コード
- problem.f90 : Fortran/c言語サンプルコード行列作成コード
- sample_c.c : c言語サンプルコード
- sample_f.f90 : Fortranサンプルコード
- wrapper_parcel_sycl.cpp : SYCL版PARCELのFortran/c言語ラッパー
