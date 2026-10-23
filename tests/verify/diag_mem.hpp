// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_VERIFY_DIAG_MEM_HPP_
#define PARCEL_VERIFY_DIAG_MEM_HPP_

// ホスト側のピーク常駐メモリ (VmHWM) を自プロセスから読む。
//   外部コマンドに頼らないので MPI ランチャの都合に左右されない。
//   CPU バックエンドでは PARCEL の「デバイス」メモリもホストメモリなので、
//   これが実質的な総使用量になる。
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include "../../Include/util_sycl.hpp"   // PARCEL が確保したデバイスメモリの記録 (report_parcel_device_peak)
namespace diag {
  inline long host_peak_kb(){
    FILE *f = std::fopen("/proc/self/status", "r");
    if(!f) return -1;
    char line[256]; long v = -1;
    while(std::fgets(line, sizeof(line), f)){
      if(std::sscanf(line, "VmHWM: %ld kB", &v) == 1) break;
    }
    std::fclose(f);
    return v;
  }
  // 大きいページ (HugeTLB) に載っている量 [kB] (/proc/self/status の HugetlbPages)。
  //   A64FX (富士通のコンパイラ) はヒープを大きいページに置くので、VmHWM (RSS) に数えられない。
  //   ホストのメモリは VmHWM + これで見る。VmHWM と違ってピークではなく、読んだ時点の値。
  //   欄が無い・読めない環境では 0 (大きいページを使わない環境では 0 になるだけ)。
  inline long host_hugetlb_kb(){
    FILE *f = std::fopen("/proc/self/status", "r");
    if(!f) return 0;
    char line[256]; long v = 0;
    while(std::fgets(line, sizeof(line), f)){
      if(std::sscanf(line, "HugetlbPages: %ld kB", &v) == 1) break;
    }
    std::fclose(f);
    return v;
  }
  // ランクごとのファイルにも同じ行を書く。
  //   標準出力だけだと MPI ランチャの集約で一部のランクの行が取りこぼされることがある。
  //   出力先は実行時のカレントディレクトリ (構成ディレクトリ) で、
  //   ソルバの計測区間の外なので時間には影響しない。
  inline void write_rank_file(int rank, const std::string &line){
    char name[64];
    std::snprintf(name, sizeof(name), "mem_rank%05d.txt", rank);
    FILE *f = std::fopen(name, "a");
    if(!f) return;
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }

  inline void report_host_peak(const char *label, int rank){
    if(!std::getenv("PARCEL_MEM_REPORT")) return;
    long kb = host_peak_kb();
    std::string line = "[MEM] rank " + std::to_string(rank) + " " + label
                     + " peak RSS = " + std::to_string(kb / 1024) + " MiB"
                     + ", HugetlbPages (now) = " + std::to_string(host_hugetlb_kb() / 1024) + " MiB";
    std::cout << line << std::endl;
    write_rank_file(rank, line);
  }

  // GPU のデバイス使用量 [MiB]。SYCL に可搬な問い合わせ API が無いので nvidia-smi を読む。
  //   PARCEL_MEM_REPORT のときだけ、ソルバの計測が終わった後に 1 回だけ呼ぶ。
  //   自ランクに割り当てられたデバイス (CUDA_VISIBLE_DEVICES) を指定する。
  //   CPU 環境や nvidia-smi が無い環境では何も出さない。
  inline long device_mem_used_mib_now(){
    const char *vis = std::getenv("CUDA_VISIBLE_DEVICES");
    std::string cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits ";
    if(vis && vis[0]) cmd += std::string("--id=") + vis + " ";
    cmd += "2>/dev/null | head -1";
    FILE *p = popen(cmd.c_str(), "r");
    if(!p) return -1;
    char buf[64] = {0};
    if(!std::fgets(buf, sizeof(buf), p)){ pclose(p); return -1; }
    pclose(p);
    return std::atol(buf);
  }
  inline void report_device_mem(const char *label, int rank){
    if(!std::getenv("PARCEL_MEM_REPORT")) return;
    long mib = device_mem_used_mib_now();
    if(mib < 0) return;
    std::string line = "[MEM] rank " + std::to_string(rank) + " " + label
                     + " device memory used = " + std::to_string(mib) + " MiB";
    std::cout << line << std::endl;
    write_rank_file(rank, line);
  }
  // PARCEL が確保したデバイスメモリのピーク [MiB] (util_sycl の記録)。PARCEL_MEM_REPORT のときだけ。
  inline void report_parcel_device_peak(const char *label, int rank){
    if(!std::getenv("PARCEL_MEM_REPORT")) return;
    const long mib = (long)(util_sycl::device_mem_peak_bytes() >> 20);
    std::string line = "[MEM] rank " + std::to_string(rank) + " " + label
                     + " PARCEL device peak = " + std::to_string(mib) + " MiB";
    std::cout << line << std::endl;
    write_rank_file(rank, line);
  }
}

#endif
