# 习题覆盖表

本文档记录 [SOLUTIONS.md](SOLUTIONS.md) 已覆盖哪些主线能力。它不是新增习题集，而是收口审查工具：若一条 Langlands 主线没有可计算练习或没有解答，就不能算作教材闭环。

## 覆盖标准

每条主线至少需要三类题：

1. 基础对象题：检查定义是否能实际使用。
2. 参数或 L 因子题：检查 Galois/Weil 参数、Satake 参数、Hecke 本征值或 L 因子能否互相转换。
3. 应用链题：检查读者能否把外部输入和本书引理拼成逻辑结论。

## 四条主线覆盖矩阵

| 主线 | 已有解答 | 覆盖能力 | 当前缺口 |
|---|---|---|---|
| `GL(1)` 与类域论 | 1.1-1.4、2.1-2.4、3.1-3.5、5.1-5.2、F.1-F.8、V.1、V.3、V.5 | restricted product、乘积公式、局部 zeta integral、Dirichlet character 到 Hecke character、几何 Frobenius、局部 reciprocity、ray class conductor、Fourier/Poisson 接口 | 基本覆盖完成 |
| `GL(2)`、椭圆曲线与费马应用 | 6.1-6.2、7.1-7.2、7.6、8.1、9.1、10.1、90.1-90.4、H.1、H.3-H.4、J.3-J.5、T.1、T.3、T.5、W.1、W.4-W.5、AD.1-AD.5、AE.1-AE.5 | slash action、Hecke 关系、adelic 尖点条件、classical/unitary normalization 平移、好约化 Frobenius、稳定格半单化、Ribet 降层位置、Frey 曲线判别式、级 `2` 矛盾、`GL(2)` 局部 LLC 模型 | 基本覆盖完成 |
| 一般算术 Langlands | 4.1-4.2、11.1、12.1、13.1、15.1、15.3、G.1、G.3-G.5、I.1-I.2、I.4-I.5、L.1、L.4-L.5、M.1-M.2、M.5、N.1、N.4-N.5、P.1、P.3、P.5、Q.1、Q.3、Q.5、R.1、R.3、R.5、X.1-X.3、Z.1、Z.3、Z.5、AA.1、AA.3-AA.4 | Hecke 幂等元、L-packet 单元素、部分 L 函数、强/弱转移、非分歧 L 因子相容、根资料、Satake 推前、Rankin-Selberg 因子、trace formula 局部角色、Arthur 参数、hyperspecial 和 spherical depth | 基本覆盖完成 |
| 几何 Langlands 与函数域桥梁 | 16.1、19.1、20.1、20.4、22.1、O.2-O.4、P.5、S.1、S.3-S.4、Y.1、Y.3、Y.5、AB.1-AB.5、AC.1-AC.5 | trace formula 与函子性动机、dominant coweight 与对偶群权、Hecke eigenvalue 张量函子、Frobenius trace 到 Hecke eigenfunction、函数域的 adelic/几何双描述、kernel formalism、几何 Satake 范畴化、shtuka/excursion operator、derived/IndCoh 接口、Fargues-Fontaine 局部几何接口 | 基本覆盖完成 |

## 关键闭环题

以下题目是主线收口时必须保留的最短题组。

### `GL(1)`

| 题号 | 作用 |
|---|---|
| 1.1 | 检查乘积公式和绝对值归一化 |
| 2.1 | 检查局部 zeta integral 对分歧特征的行为 |
| 2.4 | 区分 `GL(1)` 的 Galois 侧、自守侧和 L 函数相容 |
| 3.2 | 检查几何 Frobenius convention 对 L 因子的影响 |
| V.1 | 用局部 reciprocity 计算非分歧 character 的 Frobenius 值 |

### 费马应用

| 题号 | 作用 |
|---|---|
| 8.1 | 检查椭圆曲线好约化局部因子的 Frobenius 表达式 |
| 10.1 | 检查 Ribet 降层为什么依赖局部-整体相容 |
| 90.1 | 检查指数归约 |
| 90.3 | 检查 $X_0(2)$ genus 计算 |
| 90.4 | 检查“费马由 Langlands 证明”的精确含义 |
| AD.4 | 检查 Frey 曲线判别式计算 |
| W.5 | 检查最终级 `2` 矛盾 |

### 一般算术 Langlands

| 题号 | 作用 |
|---|---|
| 11.1 | 检查 torus 的 character/cocharacter lattice |
| 12.1 | 检查 `GL(n)` packet 单元素现象 |
| 13.1 | 检查部分 L 函数为什么需要坏位置集合 |
| 15.1 | 检查强转移推出弱转移 |
| G.5 | 检查 symmetric square 的 Satake 参数推前 |
| I.2 | 检查 Rankin-Selberg 局部因子 |
| P.3 | 检查 `GL(2)` 非分歧主级数的标准 L 因子 |
| X.3 | 检查 standard transfer 在非分歧 Satake 参数上的公式 |

### 几何 Langlands

| 题号 | 作用 |
|---|---|
| 19.1 | 检查 dominant coweight 与对偶群 dominant weight |
| 20.1 | 检查 Hecke eigenvalue 的张量函子性 |
| 22.1 | 检查函数域同时具有 adelic 和曲线几何描述 |
| O.2 | 检查 Hecke functor 的 kernel transform 形式 |
| P.5 | 检查几何 Satake 是经典 Satake 的范畴化 |
| S.1 | 检查 $\operatorname{Bun}_G(\mathbb F_q)$ 与 adelic 双商 |
| Y.5 | 检查 Hecke eigensheaf 条件对所有 $V$ 的张量相容 |

## 第一轮收口新增题目

以下四项已补入正文或解答，因此习题层面的基本收口可判定为完成。

1. `3.5`：有限阶 Hecke character 的 conductor 与 ray class factorization。
2. `7.6`：classical normalization 与 unitary automorphic normalization 的变量平移。
3. `15.3`：弱转移推出非分歧部分 L 函数相容。
4. `20.4`：Frobenius trace 如何把 Hecke eigensheaf 变成 Hecke eigenfunction。
