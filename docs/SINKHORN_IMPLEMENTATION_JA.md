# UOT-CluReg（Sinkhorn モード）実装仕様

## 1. このモードが解く問題

`algorithm="sinkhorn"` は、ClusterReg の Laplacian 非剛体変形を保ったまま、
クラスタ所属度を両側 KL 緩和つき非平衡最適輸送（UOT）に置き換えるモードである。
本書ではこの実装を **UOT-CluReg** と呼ぶ。

重要な区別として、これは公開されている Sinkhorn-CPD のコードや数式をそのまま
移植した互換モードではない。提供された理論書で定義された、参照測度
`B = b a^T` に対する一般化 KL を使うモデルである。公開 Sinkhorn-CPD の
Shannon 型エントロピーをコスト補正なしで置き換えたものとは、輸送総質量が可変なため
同値ではない。また、公開版で扱われる剛体変換を、ClusterReg の正則化された非剛体場へ
拡張している。Student-t コストも本実装側の拡張である。

## 2. 記号と目的関数

ソース点群とターゲット点群を

```text
Y in R^(Ny x d),  X in R^(Nx x d)
```

とする。既知の対応は入力しない。Laplacian カーネルから作った固定特徴行列
`Phi in R^(Ny x r)` と係数 `A in R^(r x d)` により、変形を

```text
T(Y) = Y + Phi A
```

と表す。標準設定では、ソース参照質量 `b_i = 1/Ny`、ターゲット参照質量
`a_j = 1/Nx`、参照計画 `B = b a^T` を使う。輸送計画
`Gamma in R_+^(Ny x Nx)` の周辺と総質量は

```text
p = Gamma 1,  q = Gamma^T 1,  m_Gamma = sum_ij Gamma_ij
```

である。`p=b`、`q=a`、`m_Gamma=1` は強制しない。

一般化 KL を

```text
KL(u || v) = sum_i [u_i log(u_i / v_i) - u_i + v_i]
```

と定義し、次の共同目的関数を最小化する。

```text
J(Gamma, A, s)
  = <Gamma, C(T(Y), X; s)>
  + eta   KL(Gamma || b a^T)
  + tau_y KL(Gamma 1 || b)
  + tau_x KL(Gamma^T 1 || a)
  + lambda/2 ||A||_F^2,

Gamma >= 0,  sigma_floor <= s <= sigma_ceiling.
```

実装オプションとの対応は、`eta=transport_entropy`、
`tau_y=source_mass_penalty`、`tau_x=target_mass_penalty`、
`lambda=regularization` である。`s` は Gaussian では各座標の分散、
Student-t では密度のスケールである。Python/C++ API では歴史的な名前として
`sigma2` に格納される。

上式の `||A||_F^2` は白色化した Nyström 特徴を使う既定経路の表記である。
`solver="dense"` では完全 Gram 行列 `G` と係数 `C` により
`T(Y)=Y+GC` と表し、同じ RKHS 正則化を
`lambda/2 tr(C^T G C)` として評価する。dense 係数へ Frobenius ノルムを
そのまま適用するわけではない。

### Shannon 型との相違

負の Shannon エントロピーを

```text
H(Gamma) = sum_ij Gamma_ij (log Gamma_ij - 1)
```

とすると、`sum(B)=1` の下で

```text
KL(Gamma || B) = H(Gamma) - <Gamma, log B> + 1
```

である。一様参照質量でも第2項は `m_Gamma log(Ny Nx)` となり、
`m_Gamma` が可変なので加法定数ではない。したがって、本モードは
`<Gamma,C> + eta H(Gamma)` という Shannon 型 Sinkhorn-CPD を厳密再現しない。
同じ問題へ変換するにはコスト側の参照測度補正が必要だが、現在の API はその別モデルを
提供していない。

本実装が扱うのは UOT エネルギーであり、自己輸送項を差し引く
Sinkhorn divergence でもない。

比較対象は [Sinkhorn-CPD 論文（arXiv:2606.16672）](https://arxiv.org/abs/2606.16672)
および公式リポジトリの
[`cf78b1e8e43a4dc73d258dfbb28fb39846557d82` の実装](https://github.com/Theigrams/SinkhornCPD/blob/cf78b1e8e43a4dc73d258dfbb28fb39846557d82/sinkhorn_cpd.py)
で固定している。少なくともこの公開実装との差は次の通りである。

| 項目 | 公開 Sinkhorn-CPD 実装 | 本実装 UOT-CluReg |
|---|---|---|
| 計画正則化 | `H(Gamma)`、係数1 | `eta KL(Gamma || ba^T)` |
| Gibbs 核 | `exp(-C)` | `(ba^T) .* exp(-C/eta)` |
| 周辺 | 両側 KL 緩和 | 両側 KL 緩和 |
| 点対コスト | Gaussian NLL | Gaussian または Student-t NLL |
| 変換 | 剛体 `RY+t` の Procrustes | Laplacian 非剛体場 `Y+Phi A` |
| スケール | `sum Gamma D / (d sum Gamma)`、下限あり | 同式、上下限あり。Student-t は分子だけ `Omega` |

従って「Sinkhorn-CPD の着想を非剛体 ClusterReg へ導入した」とはいえるが、
公開実装の出力互換性や同一目的関数を意味しない。

## 3. Gaussian と Student-t

`D_ij = ||T(y_i)-x_j||^2` とする。`noise_model="gaussian"` のコストは

```text
C_ij = D_ij / (2s) + d/2 log(2 pi s)
```

である。`noise_model="student-t"`、自由度 `nu=student_dof` のコストは

```text
C_ij = lgamma(nu/2) - lgamma((nu+d)/2)
     + d/2 log(nu pi s)
     + (nu+d)/2 log(1 + D_ij/(nu s)).
```

可変質量問題では、どちらの尤度正規化項も総輸送質量に作用するため省略しない。

Student-t の M-step は MM 更新である。現在の残差から

```text
w_ij = (nu+d) / (nu + D_ij/s),  Omega = Gamma .* w
```

を固定し、変形更新には `Omega` の十分統計を使う。一方、スケール更新は

```text
s_new = clip(sum_ij Omega_ij D_ij(new) / (d m_Gamma),
             sigma_floor, sigma_ceiling)
```

である。分子はロバスト重みつきだが、分母は `d sum(Gamma)` であり
`d sum(Omega)` ではない。`transport_mass` と `robust_mass` はこの違いを診断する。

Gaussian では `Omega=Gamma` となる。

## 4. 一般化 Sinkhorn 更新

変形とスケールを固定し、

```text
K = (b a^T) .* exp(-C/eta)
theta_y = tau_y / (tau_y + eta)
theta_x = tau_x / (tau_x + eta)
```

とする。輸送解は `Gamma = diag(u) K diag(v)` で、

```text
u <- (b / (K v))^theta_y
v <- (a / (K^T u))^theta_x
```

を対数領域の log-sum-exp で反復して求める。停止に使う
`transport_residual` は、`rho=theta_y theta_x < 1` に基づく
対数双対固定点までの事後上界である。`kkt_residual` は

```text
C_ij
+ eta   log(Gamma_ij / (b_i a_j))
+ tau_y log(p_i / b_i)
+ tau_x log(q_j / a_j)
```

の最大絶対値を、計画を保存せず双対量と周辺量から評価した値である。

各外側反復は次の順序で進む。

1. 現在の `T,s` に対して UOT 部分問題を解く。
2. 同じ `Gamma` から `Omega` と十分統計を作る。
3. Laplacian 変形係数を更新する。
4. 同じ計画を固定したまま `s` を更新する。
5. 同じ計画に対する更新前後の目的値を比較し、非下降なら受理しない。

返却する変形に対応する周辺質量を得るため、最後に輸送だけをもう一度解く。
この最終評価は `source_mass`、`target_mass`、`transport_mass` に反映されるが、
外側反復の `history` には新しい行を追加しない。

## 5. 変形場と Nyström 求解

Laplacian カーネルは

```text
k(y,z) = exp(-gamma ||y-z||_1)
```

である。`solver="nystrom"` では、k-means ランドマークから白色化特徴
`Q` を作り、`Phi=Q` とする。`D=diag(Omega 1)`、`P=Omega X` とすれば、

```text
(Q^T D Q + lambda s I) A = Q^T (P - D Y)
T(Y) = Y + Q A
```

を Cholesky 分解で解く。`solver="dense"` は完全カーネルを使う小規模の
検証用経路であり、計算量・メモリ量が大きい。

Nyström は変形行列の低ランク化であり、輸送の全点対評価を近似するものではない。
`rank=0` では `ceil(rank_ratio * Ny)` を要求 rank とするため、点数増加時の
計算量を抑えたい場合は固定 `rank` を明示する。

## 6. 共通正規化

Sinkhorn モードで `normalize=True` の場合、両点群に別々の中心・スケールを
適用しない。共通中心と共通 RMS 半径

```text
c = (sum_i y_i + sum_j x_j) / (Ny + Nx)
h = sqrt((sum_i ||y_i-c||^2 + sum_j ||x_j-c||^2) / (Ny + Nx))
```

を用いて、両方へ `(z-c)/h` を適用する。`align_centroids=True` なら、その後に
正規化空間でソース重心をターゲット重心へ平行移動する。この平行移動は結果の
`source_shift` に記録され、`RegistrationResult.transform()` にも同じ前処理が
適用される。

これは clustering モードの「各集合を別々に正規化する」挙動とは異なる。
物理的なスケール差を保存したい場合も、Sinkhorn モードではこの共通正規化を使う。
前処理を完全に無効にするには `normalize=False, align_centroids=False` を指定する。

共通中心、RMS 半径、既定の重心整合はいずれも通常の平均を使い、UOT や Student-t の
重みを計算する前に実行される。このため、極端または片側だけの外れ値は、輸送で質量を
抑制する前に前処理を歪め得る。その条件では `align_centroids=False` との比較、または
外部で決めたロバストな固定前処理を検討する。ただし、別の前処理を使った結果は同じ条件の
実験として明示する。

## 7. アルゴリズム、ノイズ、バックエンドの選択

| 選択 | 値 | 意味 |
|---|---|---|
| `algorithm` | `"clustering"` / `"sinkhorn"` | 論文・公式 ClusterReg の再現には前者、可変質量 UOT には後者 |
| `noise_model` | `"gaussian"` / `"student-t"` | Gaussian 的残差には前者、重い裾を仮定する比較には後者。精度向上の保証はない |
| `backend` | `"cpu"` / `"cuda"` / `"auto"` | 可搬性には CPU、対応 GPU 上の全点対処理には CUDA、環境依存選択を許す場合だけ auto |
| `solver` | `"nystrom"` / `"dense"` | 通常実行には低ランク、独立検証・小規模問題には完全カーネル |

`backend="cpu"` は CUDA を要求しない既定経路で、OpenMP が有効なら
`threads` を輸送縮約に使う。`backend="cuda"` は CUDA 対応ビルドと利用可能な
デバイスを必須とし、条件を満たさなければ例外を返す。`backend="auto"` は
CUDA 対応ビルドかつデバイスが利用可能なら CUDA、それ以外は CPU を選ぶ。
実際の選択は `result.backend_used` で確認する。

CUDA 化されるのは全点対コスト、対数 Sinkhorn 更新、十分統計、KKT 残差、
固定計画目的値である。Nyström 基底作成と変形の線形求解は現在 CPU/Eigen 上で
行うため、「全処理が GPU 常駐」という意味ではない。CUDA 実装も float64 を使い、
各外側反復で診断量と十分統計をホストへ戻す。

`semantics` と `estep` は clustering モード用で、Sinkhorn モードの更新式は
それらで変わらない。`entropy` も clustering 用であり、Sinkhorn の正則化温度には
`transport_entropy` を使う。逆に `noise_model`、`backend`、`transport_*`、
`source_mass_penalty`、`target_mass_penalty` は Sinkhorn 用で、clustering では無視される。

Python は `algorithm="clustering"` と短縮形 `"cluster"` の両方を受け付ける。
CLI は `--algorithm cluster` または `--algorithm sinkhorn` であり、
`--algorithm clustering` は受け付けない。

### Sinkhorn 主要オプション

| Python 名 | 既定値 | 条件・意味 |
|---|---:|---|
| `transport_entropy` | `1.0` | `eta > 0` |
| `source_mass_penalty` | `3.0` | `tau_y > 0` |
| `target_mass_penalty` | `3.0` | `tau_x > 0` |
| `noise_model` | `"gaussian"` | Gaussian または Student-t |
| `student_dof` | `4.0` | Student-t の `nu > 0` |
| `regularization` | `0.1` | 変形正則化 `lambda > 0` |
| `initial_sigma` | `0.0` | `0` は全点対平均二乗距離から初期化 |
| `sigma_floor` | `1e-8` | スケール下限 |
| `sigma_ceiling` | `4.0` | スケール上限 |
| `sinkhorn_iterations` | `500` | 内側反復上限 |
| `sinkhorn_tolerance` | `1e-10` | 対数双対の事後上界閾値。`0` は正の早期終了閾値を無効化 |
| `transport_mass_floor` | `1e-12` | これ未満なら `mass_collapse` |
| `max_iterations` | `50` | 外側反復上限 |
| `tolerance` | `1e-5` | 外側の目的値・変形・スケール停止閾値 |
| `backend` | `"cpu"` | CPU、CUDA、または自動選択 |

`fixed_iterations=True` は外側 tolerance 停止を無効にする。プロファイルのように
内側 Sinkhorn の正の早期終了閾値も無効にする場合は、さらに
`sinkhorn_tolerance=0.0` を指定する（浮動小数点上の変化が厳密に0なら停止し得る）。

## 8. Python 例

### CPU / Student-t

```python
import clusterreg

result = clusterreg.fit(
    source,
    target,
    algorithm="sinkhorn",
    noise_model="student-t",
    backend="cpu",
    student_dof=4.0,
    transport_entropy=1.0,
    source_mass_penalty=3.0,
    target_mass_penalty=3.0,
    regularization=0.08,
    rank=128,
    threads=4,
)

print(result.backend_used, result.stop_reason)
print(result.sigma2, result.transport_mass)
print(result.source_mass.shape, result.target_mass.shape)
registered = result.transformed
```

`source_mass` と `target_mass` の総和は同じ `transport_mass` になるが、一般には
1ではない。互換用の `result.alpha` は `source_mass` の別名であり、確率単体ではない。

### CUDA の可用性と自動選択

```python
import clusterreg

print(clusterreg.cuda_compiled())
print(clusterreg.cuda_available())
print(clusterreg.cuda_device_name())

result = clusterreg.fit(
    source,
    target,
    algorithm="sinkhorn",
    backend="auto",
    noise_model="gaussian",
    rank=128,
)
print(result.backend_used)
```

## 9. CLI 例

CPU 版をビルドして登録する。

```bash
pixi run build-cpp
./build/pixi/clusterreg_cli register \
  --source source.csv --target target.csv --out build/uot_cpu \
  --algorithm sinkhorn --backend cpu --noise-model student-t \
  --rank 128 --threads 4 \
  --transport-entropy 1 --tau-source 3 --tau-target 3 --student-dof 4
```

`--out` は存在しないディレクトリを指定する。CLI は `registered.csv`、
`history.csv`、`metrics.json`、およびソース周辺の互換名である `alpha.csv` を出力する。

CUDA 版は専用 Pixi 環境でビルドする。

```bash
pixi install -e cuda
pixi run -e cuda build-cuda
./build/pixi-cuda-env/clusterreg_cli register \
  --source source.csv --target target.csv --out build/uot_cuda \
  --algorithm sinkhorn --backend cuda --noise-model gaussian --rank 128
```

Pixi の CUDA タスクと CMake の未指定時既定値は
`CMAKE_CUDA_ARCHITECTURES=75`、すなわち compute capability 7.5 を対象にする。
別アーキテクチャでは、その GPU に合う値で CMake を再構成する。

## 10. 診断値

`RegistrationResult` の主な診断値は次の通りである。

| 値 | 意味 |
|---|---|
| `backend_used` | 実際に選ばれた CPU/CUDA |
| `source_mass`, `target_mass` | 返却変形に対して再計算した UOT 周辺質量 |
| `transport_mass` | 最終計画の総質量 |
| `sigma2` | 最終スケール |
| `preparation_seconds` | 正規化・基底準備時間 |
| `iteration_seconds` | 外側ループ時間 |
| `total_seconds` | 最終輸送再評価を含む全 fit 時間 |
| `stop_reason` | `tolerance`、`max_iterations`、`mass_collapse`、`non_descent` |

各 `history` 要素には `objective_before/after`、`transport_mass`、
`robust_mass`、`transport_residual`、`kkt_residual`、`linear_residual`、
`step_rms`、`scale_relative_change`、`transport_iterations`、
`estep_seconds`、`solve_seconds` が入る。

`objective_after` は E-step で得た同じ計画を固定して候補変形・スケールを評価した値である。
許容幅を超えて増加すると `non_descent` で停止し、その候補更新は返却結果へ反映しない。
この場合、`history` の最終行は棄却候補の診断であり、返却状態そのものではない。
質量が `transport_mass_floor` 未満なら、動かなかったことを収束成功とはせず
`mass_collapse` とする。

`transport_mass` は確率でも重なり率でもなく、1を超えることもある。コストの座標尺度、
尤度正規化項、`eta,tau_y,tau_x` を変えると質量の基準も変わるため、異なる設定間で値だけを
直接比較しない。各 `history` の輸送残差はその行の M-step 前に解いた計画に対する値である。
最終周辺は返却変形に対して再計算するが、その輸送の残差を新しい履歴行としては返さない。
Sinkhorn モードでは旧 clustering 用の `after_u`、`after_alpha`、`after_deform`、
`after_variance`、`official_loss`、`row_sum_error` は診断に使わない。

## 11. テストとプロファイリング

CUDA 非依存の完全テストは次で実行する。

```bash
pixi install
pixi run test
```

これには C++ 数値テスト、Python E2E、密な NumPy UOT オラクルとの比較、
Gaussian/Student-t、共通正規化、可変質量、目的値下降、エラー処理が含まれる。
2026-09-30 の確認では C++ と理論検証が PASS、Python は `12 passed, 3 skipped` で、
skip は CPU-only ビルドでの CUDA 一致テストだった。

CUDA 対応ビルドと、利用可能な場合の CPU/CUDA 一致テストは次で実行する。

```bash
pixi install -e cuda
pixi run -e cuda test-cuda
```

同日の CUDA 対応環境では Python `15 passed` と C++ CTest の成功を確認した。
テストを省いて Python API だけを準備する場合は、CPU では `pixi run install-python`、
CUDA では `pixi run -e cuda install-python-cuda` を実行する。

CPU プロファイルは次で作成する。

```bash
pixi run profile-sinkhorn-cpu
```

結果は `build/results/sinkhorn_cpu.json` と同名の CSV に出る。CUDA プロファイルでは
CPU 基準との数値差も同時に記録する。

```bash
pixi run -e cuda profile-sinkhorn-cuda
```

既定プロファイルは初回コンテキスト生成を含む warm-up を計測から除外し、wall time の
中央値・p95、ネイティブ時間、履歴中の E-step 合計、線形求解時間、RSS、KKT/双対残差、
CPU 基準との差を保存する。Student-t も含める任意実行例は次の通りである。

```bash
pixi run profile-sinkhorn-student-cpu
pixi run -e cuda profile-sinkhorn-student-cuda
```

CPU/CUDA の各プロファイルタスクは、対象バックエンドの editable 拡張を明示的に
再ビルドしてから測定する。これにより、以前に別バックエンドをビルドした状態へ依存しない。

最終変形に対する輸送再評価は履歴の E-step 合計には入らず、
`final_estep_and_overhead_seconds` に他のオーバーヘッドとまとめて入る。RSS はプロセスの
累積 high-water mark であり、ケース単体の増分や GPU メモリ使用量ではない。

2026-09-30 に RTX 2070、Gaussian、3次元、rank 64、ソース1024点・ターゲット1025点、
外側3反復・内側50反復で確認した単一ケースでは、wall time 中央値が
CPU `2.6685 s`、CUDA `0.3707 s`（約 `7.20x`）、CPU に対する変形の相対誤差は
`2.25e-16` だった。同じ CUDA 経路の memcheck は0 errors、racecheck は0 hazardsだった。
同じ点数・反復条件で Student-t も含めて再計測した結果は、Student-t の中央値が
CPU `5.6648 s`、CUDA `0.7925 s`（約 `7.15x`）、変形の相対誤差が `1.78e-16` だった。
これは当該 GPU・入力・反復数での確認値であり、一般的な速度保証や受入閾値ではない。

性能値はマシン、GPU、rank、点数、次元、内外反復数に依存する。プロファイラは
速度閾値を正しさの合否条件にせず、数値一致と残差を別に記録する。

## 12. 現在の限界

- 輸送計画 `Gamma` は保存しないが、CPU/CUDA とも全点対を正確に評価する。
  1 Sinkhorn sweep の演算量は `O(Ny Nx d)` であり、近傍疎化や階層化は未実装である。
- Nyström は変形場だけを低ランク化する。輸送計算を線形時間にはしない。
- CUDA は輸送側の高速化であり、基底作成と変形線形系は CPU に残る。
- CUDA の既定ビルド対象は compute capability 7.5 である。異なる GPU、CUDA、
  ドライバ構成での可搬性と性能は、その環境で再検証する必要がある。
- 共通正規化と既定の重心整合は通常の平均に基づくため、輸送前の外れ値に対して
  ロバストではない。
- UOT は遠方点の質量を弱める自由度を持ち、Student-t は大残差の影響を抑えるが、
  それだけで対応や登録精度の改善は保証されない。外れ値率、欠損率、密度差、初期ずれを
  独立に変えた実データ比較が完了するまで、原法よりロバストまたは高速とは主張しない。
- 非凸な交互最適化であり、大域最適性、微分同相性、自己交差の回避は保証しない。
- 公開 Sinkhorn-CPD の Shannon 型、balanced OT、partial OT、Sinkhorn divergence を
  比較対象として切り替える API は現在ない。
