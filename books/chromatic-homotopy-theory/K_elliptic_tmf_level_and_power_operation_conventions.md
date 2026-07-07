# 附录 K：Elliptic cohomology、tmf、level structure 与 power operation 约定

## K.1 椭圆曲线和形式群

**定义 K.1.** 椭圆曲线 $C/S$ 是带单位截面的 proper smooth genus one curve。单位截面处的形式完备化 $\widehat C$ 是 $S$ 上一维交换形式群。

**定义 K.2.** 若 $S=\operatorname{Spec}k$ 且 $\operatorname{char}k=p$，则 $C$ ordinary 当且仅当 $\widehat C$ 高度为 $1$；$C$ supersingular 当且仅当 $\widehat C$ 高度为 $2$。

**外部输入 K.3.** 椭圆曲线的 $p$-divisible group 高度为 $2$，其 connected formal part 高度只能为 $1$ 或 $2$。这是代数几何输入。

## K.2 Elliptic cohomology 的强弱定义

**定义 K.4.** 一个弱 elliptic cohomology datum 包括 even periodic ring spectrum $E$、椭圆曲线 $C/E_0$、以及形式群同构
$$
G_E\simeq \widehat C.
$$

**定义 K.5.** 一个 sheaf-theoretic elliptic cohomology theory 是在椭圆曲线模栈某个 etale 或 derived site 上的 sheaf of $\mathbb E_\infty$-ring spectra，其 stalks/sections 满足 K.4 的局部形式群条件。

**警告 K.6.** K.4 不自动给出 K.5。Power operations、descent 和 $\mathbb E_\infty$ 结构都需要额外定理。

## K.3 tmf、TMF 和 level structure

**约定 K.7.** 本书使用：

- $tmf$：connective topological modular forms；
- $TMF$：periodic topological modular forms；
- $TMF(\Gamma)$：带 level structure $\Gamma$ 的版本；
- $TMF_{K(2)}$：$K(2)$-localization，通常与 supersingular locus 的 Morava E-theory descent 相关。

**警告 K.8.** 不同文献对 compactified/noncompactified moduli、connective/periodic、level structure 的记号不同。任何计算必须声明使用哪个版本。

## K.4 Supersingular local model

**外部输入 K.9.** 在 supersingular 点附近，$TMF$ 的 $K(2)$-local 部分可由相应高度 $2$ Morava E-theory 对 automorphism group 的 descent 描述。精确形式依赖素数、supersingular 点数量和 level structure。

**例 K.10.** 在某些素数和 level 下，supersingular locus 分解为有限多个点，每个点贡献一个高度 $2$ Lubin-Tate deformation theory。global $TMF$ 通过 descent/gluing 汇总这些局部数据。

**警告 K.11.** $TMF_{K(2)}$ 与单个 $E_2^{hG}$ 的等价只在特定素数、level 和 supersingular 点条件下成立。全局 $TMF$ 不是单个 local model。

## K.5 Power operations

**定义 K.12.** 对 $\mathbb E_\infty$-ring spectrum $E$，power operations 是由 $\mathbb E_\infty$ 结构诱导在 $E^0(X)$ 或相关 cohomology groups 上的操作，通常与有限子群、isogenies 或 Hecke correspondences 相关。

**外部输入 K.13.** Morava E-theory 和 elliptic cohomology 的 power operations 可由形式群的有限子群模问题描述。高度 $2$ 情形与 modular curves 和 isogeny correspondences 密切相关。

**警告 K.14.** Power operations 不是复定向形式群律本身。它们依赖 $\mathbb E_\infty$ 结构，不能从 homotopy commutative ring spectrum 自动得到。

## 本附录小结

Elliptic cohomology 和 tmf 是 height $2$ chromatic theory 的几何入口。正式教材必须区分弱 elliptic datum、sheaf of $\mathbb E_\infty$-rings、connective/periodic tmf、level structure、supersingular local model 和 power operations。
