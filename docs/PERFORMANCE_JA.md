# Sinkhorn / UOT-CluReg 性能設計・測定記録

測定日: 2026-10-01。対象は既存 Sinkhorn 実装に対する同値最適化であり、原論文や他の登録法に対する速度・精度優位の比較ではない。目的関数は [実装仕様](SINKHORN_IMPLEMENTATION_JA.md) を参照する。

## 1. ビルドと実行の選択

既定は可搬性を重視した Release。手元の CPU 専用には Native、追加のリンク時最適化には Native+IPO を選べる。`-ffast-math`、float32 化、近傍打切りは使っていない。

| CMake オプション | 既定 | 効果・制限 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | GNU 系の測定構成は `-O3 -DNDEBUG` |
| `CLUSTERREG_NATIVE` | `OFF` | GNU/Clang の `-march=native`。CUDA はホスト側のみ `-Xcompiler=-march=native`。他 CPU へ配る binary では OFF 推奨 |
| `CLUSTERREG_IPO` | `OFF` | CPU library / CLI / tests / Python の LTO。`CheckIPOSupported` で検査し、未対応なら configure エラー |
| `CLUSTERREG_OPENMP` | `ON` | OpenMP が見つかった場合に点対処理を並列化。実行時 `threads` は明示して測定する |
| `CLUSTERREG_BLAS` | `OFF` | 外部 BLAS を使用する既存オプション。今回の速度比較は OFF。多重スレッドによる過剰並列に注意 |
| `CLUSTERREG_CUDA` | `OFF` | 任意 CUDA バックエンド。今回の IPO と同時 ON は明示的に拒否 |
| `CLUSTERREG_CUDA_LINEINFO` | `OFF` | CUDA の `--generate-line-info`。Nsight 等の解析用であり速度向上を保証するフラグではない |

リポジトリルートで実行する。以下は既存 Pixi 環境を使い、環境や依存関係の移行は不要。

```bash
rtk pixi run test                       # 通常 CPU: C++ / Python / 理論照合
rtk pixi run test-native                # Native 単独: C++
rtk pixi run test-fast                  # Native+IPO: C++ / Python / 理論照合
rtk pixi run profile-sinkhorn-fast
rtk pixi run profile-sinkhorn-student-fast
rtk pixi run -e cuda build-cuda-fast    # Native host + CUDA lineinfo、IPO OFF
```

CLI の出力先は順に `build/pixi/`、`build/pixi-native/`、`build/pixi-fast/`、`build/pixi-cuda-fast/`。Python の `install-python-fast` と `install-python` は同じ環境内の拡張を置き換えるので、切替後の設定を取り違えないこと。通常インストールは Native/IPO を明示 OFF にする。CUDA の Python は既存 `install-python-cuda` を利用する（CUDA fast タスクは CLI ビルド用）。

CMake を直接使う場合の CPU 専用例（C++17、Eigen、Ninja が見つかる環境）:

```bash
rtk cmake -S . -B build/local-fast -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCLUSTERREG_NATIVE=ON -DCLUSTERREG_IPO=ON -DCLUSTERREG_CUDA=OFF
rtk cmake --build build/local-fast --parallel 2
rtk ctest --test-dir build/local-fast --output-on-failure
```

CUDA Pixi タスクは compute capability 7.5 を指定する。他の GPU では `CMAKE_CUDA_ARCHITECTURES` を実機に合わせる。NVCC と互換のホスト C++ コンパイラを指定する必要があり、タスクは `NVCC_CCBIN` と `CMAKE_CUDA_HOST_COMPILER` を CUDA 環境の `$CXX` に統一する。

実行時は CLI の `--backend cpu|cuda|auto`、Python の `backend="cpu"|"cuda"|"auto"` を選択する。`auto` の実際の選択は `backend_used` / `metrics.json` の `backend` で確認する。明示的 CUDA 要求は不可用ならエラーであり、黙って CPU に切り替えない。

## 2. 行列形式とカーネルの対応

この章では source を \(Y\in\mathbb R^{M\times d}\)、target を \(X\in\mathbb R^{N\times d}\)、白色化 Nyström 基底を \(Q\in\mathbb R^{M\times r}\)、変形を \(Z=Y+QA\)、\(A\in\mathbb R^{r\times d}\) とする。`s` は分散 `sigma2` であり標準偏差ではない。

点対距離の行は列優先座標の直接差で計算する:

\[
D_{i,:}^{\top}=\sum_{k=1}^d (X_{:,k}-Z_{ik})^{\odot2}.
\]

Gram 展開による大きな値の差し引きを避け、Eigen の連続ベクトルに SIMD を適用する。Gaussian / Student-t はループ外でテンプレート dispatch。Student-t の `log1p` は Kahan 補正式と小引数の scalar `log1p` を併用し、`log(1+x)` の桁落ちを避ける。

一様参照質量 \(b_i=1/M,a_j=1/N\)、\(L_{ij}=\log b_i+\log a_j-C_{ij}/\eta\) を用いて、対数双対を

\[
u_i\leftarrow\theta_y\{\log b_i-\operatorname{LSE}_j(L_{ij}+v_j)\},\qquad
v_j\leftarrow\theta_x\{\log a_j-\operatorname{LSE}_i(L_{ij}+u_i)\},\quad
\theta_\ell=\frac{\tau_\ell}{\tau_\ell+\eta}
\]

と更新する。\(\theta\) は overflow を避けた比で求め、double で 0/1 に丸まる極端なパラメータは明示エラーにする。CPU は max-shift LSE と vector exp、OpenMP チームを sweep 間で再利用する。極小 exp は scalar libm に戻し、計画生成の従来の `log Γ < -745` → 0 という境界も維持する。LSE にはこの計画用 cutoff を適用しない。

\[
\Gamma=\exp(L+u1_N^\top+1_Mv^\top),\qquad
\Omega=\Gamma\odot W,\qquad
W_{ij}=\begin{cases}1&\text{Gaussian},\\(\nu+d)/(\nu+D_{ij}/s)&\text{Student-t}.\end{cases}
\]

\(\Gamma,\Omega,D,L\) の完全な \(M\times N\) 行列を保存せず、行ごとに十分統計 \(p=\Omega1_N\)、\(P=\Omega X\)、\(\sum\Omega D\)、\(\sum\Omega\|X_j\|^2\)、Gamma の両周辺と総質量を集約する。source 行を並列化し、worker 別 target 統計と long-double scalar を決まった worker 順に合成する。異なるスレッド数のビット一致ではなく、数値許容内の一致を検証する。

M-step は逆行列を作らず正定値系を解く:

\[
(Q^\top\operatorname{diag}(p)Q+\lambda sI_r)A
=Q^\top(P-\operatorname{diag}(p)Y),\qquad Z'=Y+QA.
\]

スケール更新は \(s'=\operatorname{clip}(\sum_{ij}\Omega_{ij}\|Z'_i-X_j\|^2/(d\sum_{ij}\Gamma_{ij}),s_{\min},s_{\max})\)。Student-t でも分母は元の Gamma 質量であり、robust mass へ置換しない。Gaussian の固定計画候補 cost は

\[
\frac d2\Big(\sum\Gamma\Big)\log(2\pi s')+\frac{\mathrm{SSE}(Z')}{2s'}
\]

から評価して追加の点対走査を除去する。Student-t の真の候補 cost は旧 Gamma を使って再走査し、並列化する。目的値下降判定や停止条件を省略しない。

KKT 残差では \(h_i=\eta u_i+\tau_y(\log p_i^\Gamma-\log b_i)\)、\(k_j=\eta v_j+\tau_x(\log q_j^\Gamma-\log a_j)\) により

\[
\max_{ij}|h_i+k_j|=
\max\{|\min_i h_i+\min_j k_j|,|\max_i h_i+\max_j k_j|\}
\]

と縮約し、点対再走査を \(O(M+N)\) にする。clustering の dense M-step も `K * coefficients` の重複積を除去したが、更新順や目的関数は変えていない。

CUDA は thread 内 online LSE と warp/block 合成を採用。1 sweep の cost 評価は旧 4MN 回から 2MN 回、block 縮約の barrier は 10 回から 3 回へ削減した（ソース構造上の回数であり実測速度比ではない）。source log 周辺を統計収集へ融合。target log 周辺は最後の target dual 更新から得る従来の診断 identity を保持し、実 target mass は従来どおり点対 exp/cutoff から集約する。

### 計算量とメモリ

輸送は依然として全点対 exact 評価であり、1 sweep は \(O(MNd)\)。距離を保存せず再計算することと、算術量を線形化することは別である。CPU transport scratch は worker あたり `8*(5*max(M,N)+3*N)` bytes + scalar、Student-t fixed-cost scratch は約 `8*6*N` bytes。これらの処理は順次であり、追加メモリは \(O(t(M+N))\)、ほかに双対・出力 \(O(M+N+Md)\)、基底 \(O(Mr)\)、求解 \(O(r^2)\) を要する。`t` は worker 数。

fixed rank でなく既定の `rank_ratio=0.3` を使うと `r` も点数に比例するため、全体を線形メモリとは主張できない。大規模には rank を明示する。`ru_maxrss` はプロセス全体の高水位で、カーネル workspace の独立ピークではない。今回 GPU メモリの実測はしていない。

## 3. 同一入力の旧版対最適化版

CPU: Intel i7-9750H、6 physical / 12 logical、AVX2/FMA。Linux 6.8 / x86-64、Eigen 3.4、CPU GCC 15.3、OpenMP ON、BLAS OFF。3次元・rank64・threads4・seed42、欠損/外れ値各0.1、外側3・内側50、停止 tolerance 0、同じ正規化と重心整合。Gaussian と Student-t (ν=4) を別々に比較した。

変更前に保存した CLI と 3候補に同じ CSV を渡し、各ケースの warm-up を除外、3回を交互順で実行。時間は native `fit_seconds` の中央値（秒）で、入力読込・CSV 出力・プロセス起動は含まない。準備、外側更新、最終 E-step は含む。wall time と全生値も保存する。

| source点数 / noise | 旧 Release | 最適化 Release | Native | Native+IPO | IPO / 旧版の速度比 |
|---|---:|---:|---:|---:|---:|
| 256 / Gaussian | 0.226804 | 0.132820 | 0.056220 | 0.057019 | 3.98× |
| 256 / Student-t | 0.450116 | 0.236619 | 0.131245 | 0.130738 | 3.44× |
| 1024 / Gaussian | 4.018914 | 1.701579 | 1.011702 | 0.918403 | 4.38× |
| 1024 / Student-t | 5.966512 | 2.872987 | 1.688850 | 1.663604 | 3.59× |
| 2048 / Gaussian | 10.905309 | 6.117544 | 3.221618 | 3.481182 | 3.13× |
| 2048 / Student-t | 22.884346 | 11.499963 | 6.691977 | 6.529018 | 3.51× |

Portable Release 自体も 1024点 Student-t で 2.077倍。Native+IPO が常に最速ではなく、2048 Gaussian は Native 単独より約8.1%遅かった。LTO は選択肢として残し、対象入力で測って選ぶ。

共有ホストの他負荷は停止していない。測定中 load average 8.01/5.13/4.32、CPU governor `powersave`、1観測の CPU0 周波数約3.50GHz。CPU 固定 affinity や governor 変更は行っていない。3回の最小〜最大は、1024 Student-t 旧版 `5.9514–6.5024 s`、IPO `1.6530–1.6992 s`、2048 Gaussian IPO `3.2028–3.9659 s`。当機の観測であり速度保証・統計的信頼区間ではない。

証跡は `build/results/perf_opt/optimized_compare.json` / `.csv` と同 report の `artifacts_dir`。CLI、CMake cache、CSV の SHA256、全コマンド、72測定run、目的値・反復・誤差、24 warm-up probe を保存する。これらはローカル生成物であり git 管理対象ではない。

旧 CLI SHA256: `17f51f574f8d875f6876750a150204afd9b36ad3751030133b079a2661da0ee1`。旧版は当日の作業記録で Release/GCC15.3 を確認したが、保管先に CMake cache をコピーしていないため report の旧 `compiler/build_config` は null（原構成を report だけから完全復元はできない）。候補3構成は cache/compiler/hash を report に収録。異なる実装版との比較には、測定前に旧 CLI を別パスへ保管すること。

### 実反復と数値ゲート

全72 run で外側3、history に記録した内側150、`max_iterations`、有限性・正の質量/分散・目的値非増加を確認。さらに warm-up を K+1 外側の probe とし、その history[K] の E-step が測定 K 外側の **最終 E-step** と同じ入力状態・warm dual になることを first-K 軌跡と質量で検証した。最終 E も全候補50 sweep。probe の余分な M-step は計測外で、probe E の時間を測定 fit へ足していない。

座標・source mass・scale・質量・目的値は `atol=1e-9, rtol=1e-8`、KKT は `<=1e-8`。最大 history KKT は `9.24e-13`。独立監査による保存座標/alpha の旧版との差は最大 `8.88e-16`。無効な結果を「速い」と判定することはしない。中央値が旧版より10%超悪化した場合は5回交互順で再測定する方針で、今回は該当なし。性能値は correctness CI の速度閾値にはしない。

### ステージ時間の読み方

1024 Student-t IPO の中央値: 準備 `0.00237 s`、history E-step 合計 `1.20710 s`、solve 合計 `0.00175 s`、fixed-cost + loop overhead 推定 `0.02316 s`、最終 E + return overhead 推定 `0.43862 s`。各ステージ中央値の和は fit の中央値と一致するとは限らない。

native `iteration_seconds` から history E/solve を引いた値は fixed-cost とループ処理の合計推定であり専用タイマーではない。`fit−preparation−iteration` は最終 E と返却処理の合計推定。これらを互いに混同しない。history の sweep 合計150は最終Eを含まない。`compare_sinkhorn.py` の最終 E probe が追加50を検証する一方、Python `profile_sinkhorn.py` の `sinkhorn_sweeps_total` は history のみである。

## 4. スレッド数の補足測定

Native+IPO Python、1024点、3次元・rank64・外側3・内側50、各3回の native median 秒。同じ生成 seed/入力 hash を記録。前節の CLI 交互比較とは別セッションなので、旧版との速度比には混用しない。全10ケースの数値ゲート・repeat parity は成功。

| threads | Gaussian | Student-t |
|---:|---:|---:|
| 1 | 2.45478 | 5.26471 |
| 2 | 1.50889 | 2.75059 |
| 4 | 0.78349 | 1.67180 |
| 6 | 0.80452 | 1.31812 |
| 12 | 0.90069 | 1.69923 |

この測定では Gaussian は4、Student-t は6が中央値最小。12論理スレッドは常に速いとは限らず、worker scratch と縮約負荷も増える。まず4、重い Student-t は physical core 数付近も試す。補足測定は各 threads 内の repeat parity を検査し、1/4 threads 間の一致は独立オラクル付き回帰テストで検証した。

再測定例:

```bash
rtk pixi run install-python-fast
rtk pixi run python scripts/profile_sinkhorn.py --sizes 1024 --dimension 3 \
  --rank 64 --outer-iterations 3 --sinkhorn-iterations 50 --threads 6 \
  --repeats 3 --student --output build/results/my_threads6.json

rtk pixi run python scripts/compare_sinkhorn.py \
  --baseline build/perf_opt/baseline/clusterreg_cli \
  --candidate release=build/pixi/clusterreg_cli native=build/pixi-native/clusterreg_cli ipo=build/pixi-fast/clusterreg_cli \
  --sizes 256 1024 2048 --dimension 3 --rank 64 --outer-iterations 3 \
  --sinkhorn-iterations 50 --threads 4 --repeats 3 --student \
  --output build/results/my_comparison.json
```

重い build/test/他ベンチを同時に走らせない。GPU 実機での計測は `pixi run -e cuda profile-sinkhorn-student-cuda` または profiler の `--require-cuda` を使う。warm-up の context 初期化は計測外。現バックエンドは diagnostics/statistics をホストへ戻してから fit が返るため、時間は必要な GPU 同期を含む。

## 5. 検証範囲と残課題

- CPU: 通常/Native/fast の C++23ケース、Python/独立 NumPy UOT oracle、公式2D/3D全反復と paper dense oracle が成功。最終Python再実行は **43 passed / CUDA不可用3 skipped**。追加の1/4threads・1/3/5次元・small-eta・underflow・極端theta・profiler偽成功検出も検証。
- ASan/UBSan: serial と OpenMP ON の C++テスト、および threads4 の Gaussian/Student-t CLI を検証。`ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1` が必要だった。初回 LSan は ptrace 環境制約で fatal。**リーク検査と race 検出は未実施**であり、ASan/UBSan で代替したとはしない。
- CUDA: NVCC12.9 / GCC14.4、standard と Native host + lineinfo をビルドし、CPU側CTest成功。当日 `nvidia-smi` が driver と通信できず、変更後 GPU の parity、compute-sanitizer、速度、メモリピークは **未検証**。README の旧 RTX2070 値は変更前の歴史的結果。
- CUDA の毎 sweep host scalar 同期、E-step ごとの device workspace 構築等は残る。永続 workspace/device-side 停止判定は実機テストできる次段階に延期。今回、GPU の実測高速化率を主張しない。
- CPU の並列同期は独立静的監査で重大問題なし。exp の −700/−745、Student log1p の 1e−4 を直接狙う packet/scalar 境界の専用テストは追加余地がある。今回の tail / small-eta oracle / threads 回帰は成功しているが、全浮動小数点入力を保証するものではない。
- 全点対の算術量は残る。超大規模での近似輸送・低精度・Triton/KeOps 等の追加 backend は別設計と独立精度検証が必要。非凸登録の成功やロバスト性の実データ優位も本速度試験からは結論しない。

作業計画・開始/終了ログ・失敗と処置はローカル `temp/workdoc_Oct01-2026_speed_optimization.md` に記録している。
