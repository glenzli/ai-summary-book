# 第十七章：Quiver varieties 与 Nakajima 表示构造

## 本章目标

本章定义 quiver varieties 的基本数据，说明它们如何产生 Kac-Moody algebras 和 quantum groups 的表示。完整 Nakajima theorem 作为外部输入。

## 依赖前置知识

需要基本 quiver representations、Hamiltonian reduction 和 Borel-Moore homology 或 equivariant K-theory。

## 17.1 Quiver 数据

**定义 17.1.** 一个 quiver $Q=(I,H)$ 由顶点集 $I$ 和箭头集 $H$ 构成。给定 dimension vectors $v,w\in\mathbb Z_{\ge0}^I$，取向量空间
$$
V_i\simeq\mathbb C^{v_i},\qquad W_i\simeq\mathbb C^{w_i}.
$$

**定义 17.2.** framed doubled quiver representation space 为
$$
\mathbf M(v,w)=
\bigoplus_{h\in H}\operatorname{Hom}(V_{s(h)},V_{t(h)})
\oplus
\bigoplus_{h\in H}\operatorname{Hom}(V_{t(h)},V_{s(h)})
\oplus
\bigoplus_i\operatorname{Hom}(W_i,V_i)
\oplus
\bigoplus_i\operatorname{Hom}(V_i,W_i).
$$
群
$$
G_v=\prod_i GL(V_i)
$$
按基变换作用。

**定义 17.3.** $\mathbf M(v,w)$ 有自然 symplectic form，moment map 记为
$$
\mu:\mathbf M(v,w)\to\mathfrak g_v^\ast.
$$
Nakajima quiver variety 定义为 GIT quotient
$$
\mathfrak M_\theta(v,w)=\mu^{-1}(0)^{\theta\text{-ss}}/\!/G_v.
$$

**定义 17.4.** 写一个点为
$$
(B_h,B_{\bar h},i,j),
$$
其中 $B_h:V_{s(h)}\to V_{t(h)}$，$B_{\bar h}:V_{t(h)}\to V_{s(h)}$，$i:W_i\to V_i$，$j:V_i\to W_i$。在顶点 $i$ 处，moment map 分量形式为
$$
\mu_i=\sum_{h:t(h)=i}B_hB_{\bar h}
-\sum_{h:s(h)=i}B_{\bar h}B_h+i_i j_i.
$$
符号依 quiver orientation convention 可能整体改变。

**命题 17.5.** moment map 方程在每个顶点 $i$ 处具有“入箭头复合减出箭头复合加 framing 项”的形式。

**证明.** $G_v$ 的 Lie algebra 为 $\bigoplus_i\operatorname{End}(V_i)$。对 $\xi_i\in\operatorname{End}(V_i)$，infinitesimal action 在所有以 $i$ 为端点的箭头映射上给出左乘或右乘项，在 framing maps 上给出相应基变换项。moment map 由 symplectic form 与 infinitesimal action 配对定义，因此每个 $\xi_i$ 的系数即为该顶点处的矩阵方程。$\square$

## 17.2 基本例子

**例 17.6.** 对单顶点无边 quiver，取 $\dim V=v$、$\dim W=w$。则
$$
\mathbf M(v,w)=\operatorname{Hom}(W,V)\oplus\operatorname{Hom}(V,W)
$$
点记作 $(i,j)$，moment map 为
$$
\mu(i,j)=ij\in\operatorname{End}(V).
$$
在合适稳定性条件下，$\mathfrak M(v,w)$ 与 $T^\ast\operatorname{Gr}(v,W)$ 同构。

**命题 17.7.** 在上述例子中，若稳定性要求 $i:W\to V$ 满射，则 quotient by $GL(V)$ 的底空间为 Grassmannian $\operatorname{Gr}(v,W)$。

**证明.** 满射 $i:W\to V$ 的 kernel 是 $W$ 中 codimension $v$ 的子空间，等价于选择 $W$ 的 $v$ 维 quotient。两个满射在 $GL(V)$ 作用下等价当且仅当它们有同一 kernel。故商为 quotient Grassmannian，也可与 $v$ 维子空间 Grassmannian 对偶识别。加入 $j$ 和方程 $ij=0$ 后得到 cotangent direction。$\square$

## 17.3 表示构造

**外部输入定理 17.8.** Nakajima quiver varieties 的 Borel-Moore homology 或 middle homology 携带 Kac-Moody algebra $\mathfrak g_Q$ 的表示；在固定 framing $w$ 后，直和
$$
\bigoplus_v H_\ast(\mathfrak L(v,w))
$$
给出最高权为 $w$ 的 integrable representation。

**定义 17.9.** Hecke correspondence between quiver varieties 通过改变 $v$ by simple root $\alpha_i$ 定义，产生 Chevalley generators $e_i,f_i$ 的几何算子。其点通常包括两个 quiver variety 点和一个 $I$-graded subspace
$$
V'\subset V
$$
使得 $\dim V/V'$ 为 simple root 方向。

**外部输入定理 17.10.** 这些 Hecke correspondences 满足 Kac-Moody relations，并实现相应 highest weight representation。

## 17.4 与 Springer 和 Satake 的接口

**边界说明 17.11.** 某些 quiver varieties 同构于 affine Grassmannian slices、Slodowy slices 或 instanton moduli spaces。这些同构支撑了 quiver variety、geometric Satake、symplectic duality 和 Coulomb branch 之间的联系，但每个同构都需要独立定理。

## 本章小结

本章定义了 Nakajima quiver varieties 的基本 Hamiltonian reduction 数据，写出 moment map 分量公式，计算单顶点无边 quiver 与 cotangent Grassmannian 的关系，并把表示构造作为外部输入。

## 练习

**练习 17.1.** 对单顶点无边 quiver，写出 $\mathbf M(v,w)$ 和 moment map。

**练习 17.2.** 对 $A_1$ quiver，解释 quiver variety 与 cotangent bundle of Grassmannian 的关系。

**练习 17.3.** 写出 Hecke correspondence 改变 dimension vector 的基本图。

**练习 17.4.** 对单顶点无边 quiver，证明 $ij=0$ 正是 cotangent vector annihilates tangent direction 的矩阵形式。
