# 全书定理索引

本文档索引正文和附录中的主要定理、命题、推论、引理和猜想。索引目的不是替代正文证明，而是固定每个关键结果在教材中的状态，便于检查“已证内容”和“外部输入”之间的边界。

## 状态约定

- `P`：正文给出证明。
- `S`：正文给出证明草图，完整证明依赖标准背景材料或附录。
- `E`：外部输入定理；来源应在 [SOURCES.md](SOURCES.md) 或附录 E 中可追溯。
- `C`：猜想、纲领性预期或尚未在本书中作为定理使用的期望。

若某条结果的证明依赖另一条外部输入，则按证明中实际使用的外部输入计入依赖栏。

## 第一部分：局部-整体语言

| 编号 | 状态 | 内容 | 主要依赖 |
|---|---:|---|---|
| 1.6 | E | 整体域局部化给出局部域，非 Archimedean 完备化有有限剩余域 | 代数数论基础 |
| 1.8 | S | 乘积公式 | 素理想分解、主除子次数为零 |
| 1.11 | P | adele 环为拓扑环 | restricted product 定义 |
| 1.14 | P | 对角嵌入 $K\to\mathbb A_K$ 良定义 | 分母有限性 |
| 1.15 | E | $K$ 在 $\mathbb A_K$ 中离散且 $\mathbb A_K/K$ 紧 | Minkowski/Riemann-Roch 型定理 |
| 1.19 | P | idele norm 在 $K^\times$ 上平凡 | 乘积公式 |
| 1.21 | E | adeles 的 Pontryagin 自对偶性 | 局部 Fourier 分析、Poisson summation |
| 2.5 | P | Hecke 特征分解为局部 restricted product | idele 群 restricted product |
| 2.9 | E | Tate 局部理论和局部函数方程 | Tate thesis |
| 2.11 | P | 整体 zeta 积分的 Euler 分解 | Fubini、restricted tensor product |
| 2.13 | E | Tate thesis 整体函数方程 | Poisson summation、局部函数方程 |
| 2.15 | E | 全局类域论的表示论形式 | 全局 reciprocity map |
| 3.2 | E | 局部类域论 | 局部 reciprocity map |
| 3.5 | P | 非分歧局部特征与一维 Weil 表示对应 | 3.2 |
| 3.6 | P | `GL(1)` 局部 L 因子相容 | 3.5、Tate 局部因子 |
| 3.9 | E | 一维导子相容 | 局部类域论、Artin 导子 |
| 3.11 | E | 全局类域论 | class formations |
| 3.14 | P | 有限阶 Hecke 特征与有限像一维 Galois 表示 | 3.11 |
| 3.15 | P | 有限阶情形的全局 L 函数相容 | 3.6、3.14 |
| 3.16 | P | `GL(1)` Langlands 的有限阶 Galois 形式 | 3.14、3.15 |
| 3.17 | E | `GL(1)` Langlands 的 Weil 形式 | Weil 群版本类域论 |
| 4.3 | E | Haar 测度存在唯一性 | 局部紧群测度论 |
| 4.6 | P | $C_c^\infty(G)$ 在卷积下成结合代数 | Haar 测度、Fubini |
| 4.8 | P | $e_J$ 为 Hecke 代数中的幂等元 | 开紧子群体积归一化 |
| 4.11 | P | Hecke 代数作用于光滑表示 | 紧支撑局部常值函数积分 |
| 4.12 | P | $\pi(e_J)$ 投影到 $J$-不变量 | 4.8 |
| 4.15 | E | 非分歧 Satake 同构 | 球 Hecke 代数 |
| 4.20 | E | Harish-Chandra 理论接口 | 实还原群表示论 |
| 5.3 | P | 局部 Weil 群短正合列 | Weil 群定义 |
| 5.13 | S | Satake 参数解释为非分歧局部参数 | 4.15 |
| 5.14 | P | `GL(1)` 局部 Langlands | 3.2 |
| 5.15 | E | `GL(n)` 局部 Langlands | Harris-Taylor、Henniart 等 |
| 5.16 | C | 一般还原群局部 Langlands packet 形式 | LLC 纲领 |

## 第二部分：`GL(2)`、模形式和椭圆曲线

| 编号 | 状态 | 内容 | 主要依赖 |
|---|---:|---|---|
| 6.3 | P | slash 算子给出右作用 | 直接矩阵计算 |
| 6.8 | S | 模形式在无穷尖点有 Fourier 展开 | cusp 处全纯性 |
| 6.9 | E | 模形式空间有限维性 | 模曲线和线丛理论 |
| 6.11 | E | Hecke 算子良定义性 | 双陪集与尖点全纯性 |
| 6.12 | S | Hecke 算子的 Fourier 系数公式 | 双陪集分解 |
| 6.15 | E | Hecke 本征形式 Euler 乘积 | Hecke 关系 |
| 6.16 | P | Mellin 级数给出 $L(f,s)$ | 绝对收敛半平面 |
| 6.17 | E | newform 完成 L 函数方程 | Atkin-Lehner 理论 |
| 6.18 | E | Deligne Galois 表示接口 | Deligne 构造 |
| 7.6 | P | $K_0(N),K_1(N)$ 为开紧子群 | 有限 adele 拓扑 |
| 7.9 | E | 经典-adelic 对应 | strong approximation |
| 7.12 | P | 经典尖点对应 adelic 尖点 | 尖点条件比较 |
| 7.15 | E | cuspidal representation 的 restricted tensor product 分解 | 自守表示理论 |
| 7.16 | E | Hecke eigenform 生成自守表示 | newform 与表示生成 |
| 7.19 | P | 好素数 Euler 因子的 Satake 写法 | 非分歧主级数 |
| 7.20 | E | 好素数局部 Langlands 相容 | Deligne 表示、LLC |
| 7.22 | E | `GL(2)` 标准 L 函数解析性质 | Rankin-Selberg/Godement-Jacquet |
| 8.2 | E | 椭圆曲线 Weierstrass 方程 | 椭圆曲线基础 |
| 8.5 | E | 最小模型存在性 | 局部最小模型理论 |
| 8.7 | P | 判别式非零模 $p$ 推出好约化 | Weierstrass 模型 |
| 8.9 | E | Hasse 界 | Weil 猜想的一维情形 |
| 8.10 | P | 好约化局部因子 | 点计数和 Frobenius 多项式 |
| 8.14 | P | $L(E,s)$ 初始绝对收敛 | Hasse 界 |
| 8.17 | E | 导子与约化类型 | Tate 算法、Neron 模型 |
| 8.19 | P | 半稳定导子为乘法坏素数乘积 | 8.17 |
| 8.21 | E | Tate module 基本性质 | 椭圆曲线 torsion 理论 |
| 8.22 | E | Neron-Ogg-Shafarevich 判别准则 | Neron 模型 |
| 8.23 | E | 好约化处 Frobenius 多项式 | Weil pairing、点计数 |
| 8.25 | E | 局部因子相容且独立于 $\ell$ | $\ell$-adic 表示理论 |
| 8.26 | E | 椭圆曲线模性定理接口 | Wiles 等、Breuil-Conrad-Diamond-Taylor |
| 9.7 | C | Fontaine-Mazur 二维预期接口 | p-adic Hodge theory |
| 9.8 | E | 稳定格存在 | 紧群表示 |
| 9.12 | E | Deligne 表示的模性方向 | 模曲线上同调 |
| 9.14 | P | 来自模形式的表示满足 Hecke-Frobenius trace 相容 | 9.12 |
| 9.16 | P | $V_\ell(E)$ 模性推出椭圆曲线模性 | 局部因子相容 |
| 9.17 | E | 椭圆曲线模性定理 | 模性定理 |
| 9.20 | E | 模性提升接口 | Taylor-Wiles 方法 |
| 9.21 | E | $R=T$ 原理接口 | 变形环和 Hecke 代数 |
| 9.23 | E | 半稳定模性定理 | Wiles-Taylor-Wiles |
| 9.24 | P | 半稳定模性是费马应用章所需模性输入 | 90 章逻辑链 |
| 10.1 | E | 局部-整体相容性 | Deligne、Carayol 等 |
| 10.3 | P | 好素数 Frobenius-Hecke 迹相容 | 10.1 |
| 10.6 | E | Serre 模性定理接口 | Khare-Wintenberger、Kisin |
| 10.8 | E | Ribet 降层接口 | Ribet 定理 |
| 10.10 | E | 半稳定椭圆曲线降层接口 | 10.8 |
| 10.11 | E | Frey 曲线局部导子计算 | Tate algorithm |
| 10.13 | P | Frey-Ribet 级数结论 | 9.23、10.8、10.11 |
| 10.14 | P | 级 $2$ 权 $2$ newform 推出 $S_2(\Gamma_0(2))\ne0$ | newform 定义 |

## 第三部分：一般 Langlands 纲领

| 编号 | 状态 | 内容 | 主要依赖 |
|---|---:|---|---|
| 11.9 | E | 极大环面和 Borel 子群存在及共轭 | 代数群结构理论 |
| 11.12 | P | 特征格与余特征格的完美配对 | torus 定义 |
| 11.15 | E | 根子群与余根存在 | 还原群结构理论 |
| 11.17 | P | 根反射公式 | 根资料公理 |
| 11.19 | S | Weyl 群由根反射生成 | 11.15 |
| 11.21 | E | split reductive groups 的根资料分类 | Chevalley-Demazure 理论 |
| 11.29 | P | 对偶群与直积相容 | root datum 对偶 |
| 11.31 | E | pinning 与 Galois 作用 | pinned automorphism 定理 |
| 11.43 | P | L 同态复合覆盖 Weil 群 | L 群定义 |
| 11.44 | C | Langlands 函子性局部接口 | functoriality 纲领 |
| 11.51 | E | 非分歧 Satake 参数 | Satake 同构 |
| 11.53 | P | split `GL(n)` 情形与前文 Satake 参数一致 | 7.19、11.51 |
| 12.3 | P | 不可约光滑表示有中心特征 | Schur 引理 |
| 12.11 | P | `GL(n)` 参数等同于 $n$ 维 WD 表示 | L 群定义 |
| 12.15 | P | `GL(n)` packet 单元素与 component group 平凡相容 | 中心化子计算 |
| 12.17 | C | 局部 Langlands packet 形式 | LLC 纲领 |
| 12.19 | C | 增强局部 Langlands | enhanced LLC |
| 12.30 | E | tori 的局部 Langlands | 局部类域论、Tate-Nakayama |
| 12.32 | E | `GL(n)` 局部 Langlands | Harris-Taylor、Henniart 等 |
| 12.34 | P | `GL(n)` packet 形式与双射形式等价 | 12.32 |
| 12.36 | S | 非分歧主级数的参数 | Satake 同构、局部类域论 |
| 12.39 | E | Archimedean LLC | Langlands 分类 |
| 12.41 | S | tempered 参数对应 tempered 表示的接口 | Archimedean LLC |
| 12.42 | E | 若干已知 LLC 情形 | tori、`GL(n)`、classical groups |
| 12.45 | P | 几乎处处非分歧时 Euler 乘积可形式定义 | 非分歧参数 |
| 13.8 | P | anisotropic modulo center 情形自守形式尖点 | 紧商积分 |
| 13.11 | E | 自守表示 restricted tensor product 分解 | Flath 分解 |
| 13.14 | P | 合适坏位置有限集存在 | 几乎处处非分歧 |
| 13.17 | P | `GL(n)` 标准局部因子由 Satake 参数给出 | 11.51 |
| 13.20 | P | 非分歧局部因子只依赖 spherical Hecke eigencharacter | Satake 参数 |
| 13.21 | E | Euler 乘积初始收敛接口 | 标准 L 函数估计 |
| 13.27 | C | Langlands L 函数解析性质 | functoriality 与 L 函数理论 |
| 13.29 | E | Godement-Jacquet 标准 L 函数 | Godement-Jacquet |
| 13.30 | E | Rankin-Selberg 与 Langlands-Shahidi 接口 | 积分表示、局部系数 |
| 13.33 | E | 强重数一 for `GL(n)` | Jacquet-Shalika 等 |
| 13.34 | P | 几乎所有 Satake 参数确定 `GL(n)` cuspidal 表示 | 13.33 |
| 14.1 | P | `GL(n)` L-packet 为单元素 | centralizer 结构 |
| 14.3 | E | `GL(n)` 局部 Langlands | 12.32 |
| 14.5 | P | 非分歧主级数的参数公式 | Satake 同构 |
| 14.6 | E | Bernstein-Zelevinsky 与 Langlands 分类 | BZ 分类 |
| 14.10 | E | Godement-Jacquet 标准 L 函数 | Godement-Jacquet |
| 14.11 | E | Rankin-Selberg L 函数 | Jacquet-Piatetski-Shapiro-Shalika |
| 14.13 | E | 强重数一 | `GL(n)` 自守理论 |
| 14.14 | P | Satake 参数唯一决定 cuspidal 表示 | 14.13 |
| 14.15 | P | 几乎所有 Euler 因子相同推出表示相同 | 14.13 |
| 14.16 | E | Converse theorem 接口 | Cogdell-Piatetski-Shapiro |
| 14.19 | E | 函数域 `GL(n)` 全局 Langlands | Lafforgue |
| 14.21 | P | 函数域中 Euler 因子同时决定双方 | 14.19、Chebotarev |
| 14.23 | E | 数域 regular algebraic 情形的 Galois 表示构造 | Harris-Taylor、Clozel 等 |
| 14.26 | C | 数域 `GL(n)` 全局 Langlands 粗略形式 | 纲领性陈述 |
| 15.2 | P | L 同态推前局部参数 | L 同态定义 |
| 15.7 | P | 强转移推出弱转移 | 定义比较 |
| 15.8 | P | `GL(N)` 目标弱转移唯一 | 强重数一 |
| 15.10 | C | Langlands 函子性弱形式 | functoriality 纲领 |
| 15.11 | C | Langlands 函子性强形式 | packet 和字符恒等式 |
| 15.13 | P | 弱转移推出部分 L 函数相容 | Euler 因子定义 |
| 15.15 | P | 参数复合构成强转移 | 局部 LLC 相容 |
| 15.16 | E | converse theorem 的函子性用途 | converse theorem |
| 15.20 | E | solvable base change 与 automorphic induction | Arthur-Clozel |
| 15.23 | E | 若干低阶 functorial lifts | Gelbart-Jacquet、Kim-Shahidi 等 |
| 15.25 | P | symmetric power lift 存在时的 L 函数相容 | functoriality 定义 |
| 15.28 | E | Arthur-Mok 型分类的函子性接口 | Arthur、Mok |
| 15.29 | S | 函子性与 Galois 表示复合相容 | Frobenius-Satake 相容 |
| 16.2 | P | 测试函数卷积保持右有限自守函数 | 紧支撑积分 |
| 16.5 | E | 紧商 trace formula 接口 | Selberg/Arthur trace formula |
| 16.7 | E | Arthur trace formula 粗略接口 | Arthur trace formula |
| 16.13 | S | `GL(n)` 中 regular semisimple 稳定共轭等于普通共轭 | 特征多项式 |
| 16.18 | E | endoscopic transfer 的存在接口 | transfer factors |
| 16.20 | E | fundamental lemma | Ngô |
| 16.22 | E | 稳定 trace formula 接口 | Arthur |
| 16.24 | S | trace formula 比较的函子性含义 | 稳定谱分解 |
| 16.26 | E | twisted trace formula 的 base change 接口 | Arthur-Clozel |
| 17.4 | C | Ramanujan-tempered 预期 | temperedness 纲领 |
| 17.15 | E | Arthur multiplicity formula | Arthur 分类 |
| 17.17 | E | Arthur 标准转移 | Arthur-Mok |
| 17.20 | P | tempered Arthur 参数给出 tempered Langlands 参数 | 参数定义 |
| 17.21 | S | 非平凡 $\operatorname{SL}_2$ 因子预期给出非 tempered 分量 | Arthur 参数解释 |
| 17.23 | E | L 函数判别 self-duality 类型 | classical groups L 函数理论 |

## 第四部分：几何 Langlands

| 编号 | 状态 | 内容 | 主要依赖 |
|---|---:|---|---|
| 18.5 | E | $\operatorname{Bun}_G$ 的代数栈性 | 代数栈理论 |
| 18.8 | E | Weil uniformization 接口 | Beauville-Laszlo、adelic uniformization |
| 18.12 | P | Hecke stack 给出 $\operatorname{Bun}_G$ 上 correspondence | Hecke stack 定义 |
| 18.14 | E | affine Grassmannian 的 Schubert 分解 | loop group 几何 |
| 19.2 | E | Schubert 分解 | affine Grassmannian |
| 19.6 | E | 卷积保持 perversity | 几何 Satake 技术核心 |
| 19.10 | E | Satake 范畴的 Tannakian 性 | Tannakian formalism |
| 19.11 | E | 几何 Satake 等价 | Mirkovic-Vilonen 等 |
| 19.13 | P | 卷积对应张量积 | 19.11 |
| 19.15 | P | Hecke 函子的张量相容 | 卷积定义 |
| 19.16 | S | 几何 Satake 的函数迹恢复经典 Satake | sheaf-function dictionary |
| 20.3 | P | local system 给出表示范畴上的张量函子 | 关联丛构造 |
| 20.6 | P | eigensheaf 对直和与张量积的相容 | Hecke 本征定义 |
| 20.7 | C | 几何 Langlands 朴素本征层形式 | 几何 Langlands |
| 20.9 | E | 几何类域论接口 | Abel-Jacobi、Fourier-Mukai |
| 20.11 | S | eigensheaf 的 trace 给出 Hecke eigenfunction | sheaf-function dictionary |
| 21.7 | C | 几何 Langlands 范畴形式 | categorical geometric Langlands |
| 21.9 | S | 范畴形式推出本征层形式 | skyscraper sheaf 与 Hecke 作用 |
| 22.2 | S | $\operatorname{Bun}_G(\mathbb F_q)$ 与 adelic 双商 | Weil uniformization |
| 22.3 | E | sheaf-function dictionary | Grothendieck-Lefschetz trace formula |
| 22.5 | S | eigensheaf trace 给出 eigenfunction | 20.11、22.3 |
| 22.6 | E | Drinfeld 函数域 `GL(2)` 定理 | shtukas |
| 22.7 | E | Lafforgue 函数域 `GL(n)` 定理 | shtukas |
| 22.11 | E | shtuka cohomology 接口 | Drinfeld-Lafforgue |
| 22.14 | E | Ngô 支持定理与 fundamental lemma 接口 | Hitchin fibration |

## 应用章和附录

| 编号 | 状态 | 内容 | 主要依赖 |
|---|---:|---|---|
| 90.1 | P | 费马大定理，作为最终应用命题 | 90.10 |
| 90.2 | P | 指数归约 | 初等数论 |
| 90.5 | E | Frey 曲线性质 | Frey-Hellegouarch、Tate algorithm |
| 90.7 | E | 半稳定模性定理 | Wiles-Taylor-Wiles |
| 90.8 | E | Ribet 降层定理当前形式 | Ribet |
| 90.9 | P | $S_2(\Gamma_0(2))=0$ | 附录 D |
| 90.10 | P | 接受三项外部输入推出费马大定理 | 90.2、90.5、90.7、90.8、90.9 |
| A.3 | S | 乘积公式复习 | 素理想分解/除子次数 |
| A.9 | E | 局部类域论 | class field theory |
| A.10 | E | 全局类域论 | class field theory |
| A.12 | E | Chebotarev 密度 | analytic algebraic number theory |
| A.13 | S | Frobenius characteristic polynomial 决定半单 $\ell$-adic 表示 | Chebotarev、Brauer-Nesbitt |
| A.15 | P | 非零元的素理想赋值只有有限支撑 | Dedekind 域分式理想分解 |
| A.16 | P | 数域乘积公式的有限素数部分 | 理想范数 |
| A.17 | P | 数域乘积公式 | A.16、Archimedean norm |
| A.19 | S | Ray class group 有限 | 理想类群有限、Dirichlet 单位定理 |
| A.21 | S | Ray class group 的 idele class 商描述 | idele 到理想的映射 |
| A.24 | E | Norm subgroup theorem | 全局类域论存在定理 |
| A.26 | P | Artin 导子对直和可加 | ramification filtration |
| A.27 | P | 非分歧表示导子为 $0$ | Artin 导子定义 |
| A.28 | S | 一维 Artin 导子与乘法导子相容 | 局部类域论导子相容 |
| B.2 | E | Van Dantzig 定理 | tdlc 群理论 |
| B.4 | E | Haar 测度存在唯一性 | 测度论 |
| B.6 | S | 紧群和 reductive $p$-adic groups unimodular | Haar 测度、代数群 |
| B.9 | P | 开紧子群特征函数为幂等元 | Haar 归一化 |
| B.10 | P | unimodular locally profinite group 上卷积结合律 | Fubini、Haar 换元 |
| B.11 | P | 开紧平均算子的像为 $J$-不变量 | B.9 |
| B.12 | E | 商测度和 Weil 积分公式 | 局部紧群测度论 |
| B.14 | S | 离散闭子群商存在右不变测度 | B.12 |
| B.15 | P | restricted product 张量函数积分分解 | 有限维 Fubini |
| B.16 | P | restricted product 开紧体积公式 | B.15 |
| C.4 | S | Hecke 代数积分作用良定义 | 光滑性与紧支撑 |
| C.5 | P | 双不变 Hecke 函数保持 $J$-不变量 | Haar 左不变性 |
| C.9 | E | Jacquet functor 基本性质 | Bernstein-Zelevinsky 理论 |
| C.13 | P | 有限维 Schur 引理 | 代数闭域上线性代数 |
| C.14 | E | admissible smooth 表示的 Schur 引理 | Bernstein 理论 |
| C.16 | P | 不可约 admissible smooth 表示有中心特征 | C.14 |
| C.18 | P | smooth dual 是 smooth representation | stabilizer 共轭 |
| C.19 | S | $J$-不变 smooth dual 与 $(V^J)^\vee$ 比较 | 开紧平均算子 |
| C.21 | P | admissibility 在扩张下稳定 | $J$-不变量左正合 |
| C.22 | P | 有限长度表示的 admissibility 判准 | C.21 |
| C.23 | E | reductive $p$-adic groups 中不可约 smooth 表示可容许 | Bernstein 理论 |
| D.3 | E | 模曲线代数化与微分形式 | 模曲线理论 |
| D.4 | P | $\dim S_2(\Gamma)=g(X(\Gamma))$ | D.3、Riemann surface 理论 |
| D.5 | E | $X_0(N)$ genus formula | Riemann-Hurwitz |
| D.6 | P | $S_2(\Gamma_0(2))=0$ | D.4、D.5 |
| D.8 | E | Atkin-Lehner-Li newform theory | newform theory |
| D.10 | P | $[\operatorname{SL}_2(\mathbb Z):\Gamma_0(2)]=3$ | $\mathbb P^1(\mathbb F_2)$ 作用 |
| D.11 | P | $X_0(2)$ cusp 数为 $2$ | cusp 轨道矩阵构造 |
| D.12 | S | $X_0(2)$ 椭圆点数 $e_2=1,e_3=0$ | 椭圆点同余方程 |
| D.13 | P | $X_0(2)$ genus 为 $0$ | D.10-D.12 |
| D.14 | S | 权 $2$ cusp form 给出全纯微分 | slash 变换和 $q$-展开 |
| D.15 | P | genus $0$ 推出 $S_2(\Gamma)=0$ | D.3、D.4 |
| F.3 | P | annihilator 为闭子群 | compact-open topology |
| F.4 | E | Pontryagin duality | LCA 群理论 |
| F.5 | E | 闭子群对偶正合列 | Pontryagin duality |
| F.6 | S | cocompact lattice 的 annihilator 离散且余商紧 | F.5 |
| F.8 | E | Fourier inversion and Plancherel | LCA Fourier 分析 |
| F.10 | P | 非 Archimedean $\mathbf 1_{\mathcal O_F}$ Fourier 变换 | 紧群非平凡特征积分为零 |
| F.11 | P | conductor 为 $\mathcal O_F$ 时 $\mathbf 1_{\mathcal O_F}$ 自 Fourier | F.10 |
| F.13 | S | Fourier 变换保持 Schwartz-Bruhat 空间 | 局部 Fourier 分析 |
| F.15 | P | adele Schwartz 纯张量 Fourier 变换逐处分解 | restricted product、Fubini |
| F.16 | E | adeles 的自对偶性 | Tate-Weil Fourier 分析 |
| F.17 | P | $\widehat{\mathbb A_K/K}\simeq K$ | F.5、F.16 |
| F.19 | E | LCA Poisson summation | Fourier inversion |
| F.20 | P | adele Poisson summation | 1.15、F.16、F.19 |
| F.21 | P | Tate 整体 zeta 积分的局部分解 | B.15、绝对收敛 |
| F.22 | E | Tate 整体函数方程的 Fourier 分析核心 | Tate thesis |
| G.2 | P | split torus 的 character/cocharacter lattices | 直接计算 |
| G.3 | P | `GL(n)` 的根、simple roots 和 coroots | adjoint action on matrix units |
| G.4 | P | `GL(n)` Weyl group 为 $S_n$ | monomial matrices |
| G.5 | P | `GL(n)` 自对偶 | root datum 对偶 |
| G.7 | P | `SL(n)` character/cocharacter lattices | determinant 约束 |
| G.8 | P | `SL(n)` 与 `PGL(n)` 互为对偶 | root/weight lattice 对偶 |
| G.11 | S | $\operatorname{Sp}_{2n}$ root system 为 type $C_n$ | Lie algebra weights |
| G.12 | E | classical groups 的对偶表 | root datum classification |
| G.14 | P | split `GL(n)` 的 L 群 | G.5 |
| G.15 | P | split `SL(n)` 和 `PGL(n)` 的 L 群 | G.8 |
| G.17 | S | $\operatorname{Res}_{E/F}\mathbb G_m$ 的 character lattice | 基变换到 $\overline F$ |
| G.18 | P | restriction of scalars torus 的 L 群 | G.17 |
| G.20 | P | determinant 给出 L 同态 | split L 群定义 |
| G.21 | P | central embedding 给出 L 同态 | split L 群定义 |
| G.22 | P | symmetric square 给出 L 同态 | 代数群表示 |
| H.2 | P | 双陪集算子代表无关并保持 $\Gamma$-不变性 | slash 右作用 |
| H.4 | S | $\ell\nmid N$ 的 $\Gamma_0(N)$ Hecke 双陪集分解 | 指数 $\ell$ 子格分类 |
| H.5 | P | $\ell\mid N$ 的 $U_\ell$ 代表族 | 第六章定义 |
| H.7 | P | $\alpha_b$ 代表族给出 $a_{\ell n}$ 项 | Fourier 根求和 |
| H.8 | P | $\beta$ 代表给出 $\ell^{k-1}a_{n/\ell}$ 项 | slash 计算 |
| H.9 | S | 好素数 Hecke Fourier 公式 | H.4、H.7、H.8 |
| H.10 | P | 坏素数 $U_\ell$ Fourier 公式 | H.5、H.7 |
| H.12 | S | Petersson 内积绝对收敛 | cusp 指数衰减 |
| H.13 | E | Hecke 算子的 Petersson 正规性 | Hecke correspondence 理论 |
| H.16 | S | adelic 球 Hecke 作用对应经典 $T_p$ | strong approximation、H.4 |
| H.17 | S | $K_p$-球向量线上 Hecke 特征值为 $a_p$ | newform theory、H.16 |
| H.18 | P | Hecke 多项式等于 Satake 参数 characteristic polynomial | 代数展开 |
| I.2 | E | 局部 Godement-Jacquet 理论 | Godement-Jacquet |
| I.3 | E | Godement-Jacquet 非分歧计算 | 球函数计算 |
| I.6 | S | Godement-Jacquet 全局积分 Euler 分解 | restricted tensor product、Fubini |
| I.7 | E | Godement-Jacquet 全局定理 | Poisson summation on matrix space |
| I.10 | E | Whittaker uniqueness for `GL(n)` | Gelfand-Kazhdan/Jacquet theory |
| I.12 | E | 局部 Rankin-Selberg 理论 | JPS-Shalika |
| I.13 | E | 全局 Rankin-Selberg unfolding | Rankin-Selberg method |
| I.14 | E | Rankin-Selberg 解析性质 | JPS-Shalika |
| I.17 | E | Converse theorem 的积分表示背景 | Cogdell-Piatetski-Shapiro |
| I.19 | P | 函子性转移推出非分歧 L 函数相容 | Satake 参数推前 |

## 主线依赖链

费马应用章实际使用的最短定理链为
$$
90.5,\ 90.7,\ 90.8,\ 90.9 \Longrightarrow 90.10 \Longrightarrow 90.1.
$$
其中 $90.9$ 依赖附录 D 的 genus 计算；$90.7$ 属模性输入；$90.8$ 属降层输入；$90.5$ 属 Frey 曲线局部计算输入。

数论 Langlands 主线的核心链为
$$
1.15,\ 1.21 \Longrightarrow 2.13,\qquad
3.2,\ 3.11 \Longrightarrow 3.16,\qquad
4.15,\ 5.15 \Longrightarrow 12.32,
$$
并继续进入
$$
13.11,\ 13.29,\ 13.33,\ 14.3,\ 14.19,\ 15.10,\ 15.11.
$$
这里最后两项为猜想，不能在证明中当作无条件定理使用。

几何 Langlands 主线的核心链为
$$
18.5,\ 18.14 \Longrightarrow 19.11 \Longrightarrow 20.7,
$$
而范畴化版本为
$$
19.11,\ 21.7 \Longrightarrow 21.9.
$$
当工作在有限域上时，$22.3$ 把 sheaf 侧结果投影到函数侧，连接第十三、十四章的函数域自守表示。
