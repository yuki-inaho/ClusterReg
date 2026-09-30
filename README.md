# ClusterReg — 高速 C++ / nanobind / UOT-CUDA 実装

CVPR 2024 論文 [**Correspondence-Free Non-Rigid Point Set Registration Using
Unsupervised Clustering Analysis**](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Correspondence-Free_Non-Rigid_Point_Set_Registration_Using_Unsupervised_Clustering_Analysis_CVPR_2024_paper.pdf) の、理論監査付き C++17 実装です。Python API は
nanobind で C++ コアを直接呼び出します。旧 MATLAB 実装には依存しません。

本実装は、論文の更新式と公式リポジトリ
[`zikai1/ClusterReg`](https://github.com/zikai1/ClusterReg) の固定コミット
`47b6ce98fb53534871da4e4bead739ac7d170091` を照合して作られています。
論文の次元・転置上の曖昧さを修正した `paper` モードと、公式 Python/MATLAB の反復順を
再現する検証用 `official` モードを明確に分けています。

さらに、提供された理論書に基づく `sinkhorn` モードを実装しています。これは
Laplacian 非剛体場に、`B=ba^T` を参照測度とする両側 KL 緩和 UOT を組み合わせた
**UOT-CluReg** です。公開 Sinkhorn-CPD の Shannon 型目的関数そのものではなく、
Gaussian と Student-t、CPU と任意の CUDA バックエンドを選択できます。

## Pixi で実行

Linux x86-64 と Pixi 0.79 以降を想定しています。依存関係、Eigen 3.4、コンパイラ、
nanobind、テストツールは `pixi.toml` / `pixi.lock` で管理します。

```bash
pixi install
pixi run test
```

テストを走らせず対話的に Python API だけ準備する場合は `pixi run install-python` を使います。

`pixi run test` は次を end-to-end で実行します。

- C++ の数式・境界条件テスト
- nanobind を通した人工点群の登録、配列所有権、GIL 解放、1/4スレッド一致
- hash で固定した公式 Python 実装との全反復比較
- 独立 NumPy 完全カーネル基準との `paper` モード比較
- 独立 NumPy UOT 基準との Sinkhorn、Gaussian/Student-t、共通正規化の比較

サニタイザと小規模ベンチは個別に実行できます。

```bash
pixi run test-sanitize
pixi run benchmark-smoke
pixi run profile-sinkhorn-cpu
pixi run profile-sinkhorn-student-cpu
```

CUDA は独立した任意環境です。既定の CPU 環境は CUDA toolkit や GPU を要求しません。

```bash
pixi install -e cuda
pixi run -e cuda test-cuda
pixi run -e cuda profile-sinkhorn-cuda
pixi run -e cuda profile-sinkhorn-student-cuda
```

CUDA Python API だけを準備するタスクは `pixi run -e cuda install-python-cuda` です。

## Python API

```python
import numpy as np
import clusterreg

# 真の対応は評価専用で、fit() には渡りません。
data = clusterreg.make_synthetic(
    count=512,
    dimension=3,
    amplitude=0.4,
    noise=0.003,
    seed=42,
)

model = clusterreg.fit(
    data.source,
    data.target,          # 順序をシャッフル済み
    rank=128,
    threads=4,
    semantics="paper",
)

registered = model.transformed
rmse = np.linalg.norm(registered - data.truth) / np.sqrt(len(data.truth))
warped_query = model.transform(data.source[:20])
print(model.stop_reason, model.sigma2, rmse)
```

UOT-CluReg は `algorithm="sinkhorn"` で選びます。

```python
model = clusterreg.fit(
    data.source,
    data.target,
    algorithm="sinkhorn",
    noise_model="student-t",  # または "gaussian"
    backend="cpu",            # "cuda" / "auto" も選択可能
    rank=128,
    threads=4,
    transport_entropy=1.0,
    source_mass_penalty=3.0,
    target_mass_penalty=3.0,
)
print(model.backend_used, model.transport_mass, model.stop_reason)
```

`fit(source, target)` は `N×d` と `M×d` の array-like を受け取り、対応点や正解ラベルを
使用しません。C/F-order、非連続 view、float32 は C-contiguous float64 へ安全にコピーし、
入力を変更しません。計算中は GIL を解放します。結果配列もモデルから独立した所有コピーです。

clustering モードの主要オプションの既定値は論文・公式実装に合わせています。

| オプション | 既定値 | 意味 |
|---|---:|---|
| `entropy` | `0.5` | エントロピー係数 λ（公式 `beta`） |
| `regularization` | `0.1` | 変形正則化 ζ |
| `gamma` | `2.0` | L1 Laplacian kernel の係数（公式 `theta=0.5` の逆数） |
| `rank_ratio` | `0.3` | `rank=0` 時の Nyström landmark 比率 |
| `rank` | `0` | 固定 landmark 数。大規模データでは明示を推奨 |
| `threads` | `1` | exact streaming E-step の OpenMP スレッド数 |
| `semantics` | `"paper"` | `"paper"` または `"official"` |
| `solver` | `"nystrom"` | `"nystrom"` または検証用 `"dense"` |
| `estep` | `"streaming"` | `"streaming"` または検証用 `"dense"` |

`clusterreg.Options` と enum (`Semantics`, `Solver`, `EStep`) を直接使うこともできます。
Sinkhorn モードではさらに `Algorithm`、`NoiseModel`、`Backend` を利用できます。
主な既定値は `transport_entropy=1.0`、`source_mass_penalty=3.0`、
`target_mass_penalty=3.0`、`student_dof=4.0`、`sinkhorn_iterations=500`、
`sinkhorn_tolerance=1e-10` です。`semantics`、`estep`、`entropy` は clustering
モード用であり、Sinkhorn の更新式には使いません。

## 数式との対応

target を `X∈R^(M×d)`、source を `Y∈R^(N×d)`、membership を
`U∈R^(M×N)` とします。論文式(9)はこの定義のままでは積の向きが合わないため、

```text
D = diag(Uᵀ 1_M),  P = Uᵀ X
```

に統一します。L1 Laplacian Gram 行列を Nyström 近似 `K≈QQᵀ` とすると、変形更新は

```text
(Qᵀ D Q + ζ σ² I) B = Qᵀ(P - D Y)
T = Y + Q B
```

です。これは同じ `QQᵀ` に対する公式 Woodbury 式の代数的な縮約であり、追加近似では
ありません。逆行列や `N×d` の大きな係数を作らず Cholesky で解きます。

`paper` モードでは `U → α → T → σ²` の順に、共通等方分散の目的関数をブロック最小化します。
`official` モードでは公式 Python/MATLAB の epsilon、更新前 `T` による分散、停止用 loss を
再現します。ただし、公式 C++ の source/target 除数・shape バグは再現しません。

Sinkhorn モードでは、輸送計画 `Γ` と一様参照質量 `b,a` に対し、

```text
<Γ,C> + η KL(Γ || baᵀ)
      + τy KL(Γ1 || b) + τx KL(Γᵀ1 || a)
      + λ/2 ||A||²
```

を交互最小化します。総輸送質量を1に固定しないため、一般化 KL の `-p+q` と
Gaussian/Student-t 密度の正規化項も保持します。Sinkhorn モードの正規化は両点群で
同じ中心・RMS 半径を使い、別々のスケール変換を導入しません。

詳しい導出と留保:

- [数式と C++ の対応](docs/THEORY_JA.md)
- [論文・公式実装との差分監査](docs/AUDIT_JA.md)
- [実測結果と再現範囲](docs/RESULTS_JA.md)
- [UOT-CluReg の目的関数・API・CUDA・診断仕様](docs/SINKHORN_IMPLEMENTATION_JA.md)

## 高速化

- `M×N` の membership 行列を保存せず、`Uᵀ1`, `UᵀX`, SSE、entropy を逐次集約
- `r×r` の正定値系を直接解き、明示逆行列と不要な大係数を排除
- SIMD しやすい座標列、スレッド別統計量、OpenMP による exact E-step
- 再現可能な k-means++ と、通常距離に対して正しい Elkan 枝刈り
- 極小確率でも壊れない log-sum-exp と scalar tail `exp`
- UOT の `M×N` 計画を保存せず、対数 Sinkhorn と十分統計を全点対から逐次縮約
- 任意の CUDA バックエンドで点対コスト、LSE、統計量、KKT 残差を float64 計算

近傍打切り、ハード対応、輸送疎化、Gaussian 変形 kernel への置換は行っていません。
clustering E-step と Sinkhorn 輸送はいずれも全点対を評価するため、点対部分の時間計算量
`O(MNd)`（Sinkhorn はさらに内側反復数倍）は残ります。固定 rank なら主要メモリを
抑えられますが、既定の `rank_ratio=0.3` は N と共に rank が増えるため、全体を線形時間・
線形メモリとは主張しません。CUDA は現在、輸送側を担当し、Nyström 基底と線形求解は
CPU/Eigen 上で実行します。

## C++ CLI / ライブラリ

Pixi 内で C++ のみをビルドできます。

```bash
pixi run build-cpp
./build/pixi/clusterreg_cli demo --out build/demo --n 512 --dim 3 --rank 128 --threads 4
./build/pixi/clusterreg_cli register \
  --source source.csv --target target.csv --out build/registered \
  --rank 128 --threads 4 --mode paper

./build/pixi/clusterreg_cli register \
  --source source.csv --target target.csv --out build/registered_uot \
  --algorithm sinkhorn --backend cpu --noise-model student-t \
  --rank 128 --threads 4 --transport-entropy 1 \
  --tau-source 3 --tau-target 3 --student-dof 4
```

CUDA CLI は `pixi run -e cuda build-cuda` で `build/pixi-cuda-env/clusterreg_cli` に
生成されます。Pixi タスクは既定で compute capability 7.5 を対象にします。
`--backend auto` は CUDA 対応ビルドかつデバイス利用可能時だけ CUDA を選択し、
実際の選択を `metrics.json` の `backend` に記録します。

RTX 2070、3次元、rank 64、1024対1025点、外側3・内側50反復の単一プロファイルでは、
中央値が CPU `2.6685 s`、CUDA `0.3707 s`（約 `7.20x`）、変形の相対差は
`2.25e-16` でした。同条件の Student-t は CPU `5.6648 s`、CUDA `0.7925 s`
（約 `7.15x`）、相対差 `1.78e-16` でした。これは環境限定の確認値であり、
速度保証ではありません。

Pixi を使わない場合は CMake 3.18 以上、C++17、Eigen 3.4 が必要です。

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
```

公開 C++ API は `include/clusterreg/clusterreg.hpp` の `fit()` と `transform()` です。
CSV と ASCII PLY は CLI から利用できます。

## 検証範囲と限界

人工 2D/3D、任意 5D、点数不一致、欠損、ノイズ、重複点、極小分散、並列実行をテストします。
Sinkhorn は独立 NumPy オラクル、Gaussian/Student-t、可変質量、共通正規化、目的値下降、
利用可能時の CPU/CUDA 一致も検査します。一方、FAUST/TOSCA の論文表全体、公式 C++/MKL・
公開 Sinkhorn-CPD との速度比較、Windows/macOS は再現していません。非凸問題なので、
大域最適性、正しい物理対応の一意回復、微分同相性、大変形・外れ値に対する成功は保証しません。
UOT の質量抑制と Student-t の影響抑制は実装・数値整合性を確認した機構であり、原法より
ロバストまたは高速という結論には、同一条件のデータセット実験が別途必要です。
また、Sinkhorn の共通正規化と既定の重心整合は通常の平均に基づくため、極端な外れ値は
UOT の質量抑制より前に前処理を歪め得ます。

論文補足の Nyström 誤差証明には、L1 から L2 量子化誤差へ移る際の次元係数に疑義があります。
本実装は印刷された保証を無批判に採用せず、実測成立と定理の妥当性を分けて記録しています。

## ライセンス

AGPL-3.0-only。公式コードの attribution と「学習・非商用用途」の追加記載については
[NOTICE.md](NOTICE.md) を確認してください。本リポジトリは公式コードに新たな商用利用許諾を
与えるものではありません。
