# 第五十章：Stacks、Gerbes 与非阿贝尔上同调

集合值 sheaf 的粘合只要求元素相等；几何对象往往只有局部同构，且同构本身还要满足 cocycle 相干。Groupoid-valued stack 保留这一级自同构，higher stack 则继续保留更高同伦。Torsor 由 classifying stack $BG$ 分类，gerbe 可视为局部非空、局部连通的 stack，并在适当交换或带带状结构下对应二阶上同调。本章从 fibered category 与 descent data 开始，逐步走向非阿贝尔上同调的类型正确陈述。

背景包括 Grothendieck topology、Cech nerve、2-categories 与 higher topoi。我们会区分 prestack、stack、hyperstack 和 gerbe，并明确 $H^1$ 是 pointed set、$H^2$ 需要何种系数结构；不把非阿贝尔上同调写成普通群。

## 50.1 预 Stack 与 Stack

**定义 50.1.** 设 $(\mathcal C,J)$ 为站点。一个 groupoid-valued prestack 是伪函子

$$
F:\mathcal C^{op}\to\mathbf{Grpd}.
$$

**定义 50.2.** Prestack $F$ 称为 stack，若对每个覆盖 $U_\bullet\to U$，自然函子

$$
F(U)\to\operatorname{Desc}(F,U_\bullet)
$$

为 groupoids 的等价。

**命题 50.3.** Set-valued sheaf 可视为离散 groupoid-valued stack。

**证明.** 集合 $S$ 可视为只有恒等态射的离散 groupoid。若 $F$ 是 sheaf，则覆盖上的匹配族唯一粘合为全局截面；这正是离散 groupoid 情形的 descent 等价。反过来，离散 stack 的 descent 等价在对象集上给出 sheaf 条件。$\square$

## 50.2 Descent data

**定义 50.4.** 覆盖 $\{U_i\to U\}$ 上对象的 descent datum 由对象 $x_i\in F(U_i)$、同构

$$
\phi_{ij}:x_i|_{U_{ij}}\xrightarrow{\sim}x_j|_{U_{ij}}
$$

以及三重交上的 cocycle 条件

$$
\phi_{jk}\phi_{ij}=\phi_{ik}
$$

组成。

**命题 50.5.** Stack 条件保证带 descent datum 的局部对象可有效粘合。

**证明.** Stack 条件说 $F(U)\to\operatorname{Desc}(F,U_\bullet)$ 是等价。给定 descent datum 即 $\operatorname{Desc}(F,U_\bullet)$ 的对象。等价的本质满性给出 $x\in F(U)$ 映到该 descent datum；完全忠实性保证粘合对象在唯一同构意义下唯一。$\square$

## 50.3 Classifying stacks 与 torsors

**定义 50.6.** 设 $G$ 为站点上的 sheaf of groups。$G$-torsor 是带右 $G$-作用的 sheaf $P$，局部同构于 $G$ 自身的右正则作用。

**定义 50.7.** Classifying stack $BG$ 把 $U$ 送到 $U$ 上 $G|_U$-torsors 的 groupoid。

**外部输入定理 50.8.** $BG$ 是 stack，且

$$
H^1(U,G)
$$

可识别为 $G$-torsors 的同构类，即 $\pi_0 BG(U)$。

**命题 50.9.** 若 $G$ 为平凡群 sheaf，则 $BG$ 等价于终 stack。

**证明.** 平凡群作用的 torsor 局部同构于单点 sheaf。由于没有非平凡 automorphisms 或 twisting，任意 torsor 全局同构于单点 sheaf。故每个 $U$ 上 $BG(U)$ 为终 groupoid，自然地给出终 stack。$\square$

## 50.4 Gerbes

**定义 50.10.** Stack $\mathcal G$ 称为 gerbe，若它局部非空，且任意两个局部对象在进一步覆盖后局部同构。

**定义 50.11.** 若 $A$ 为 abelian sheaf of groups，$A$-banded gerbe 是每个对象的 automorphism sheaf 与 $A$ 相容识别的 gerbe。

**外部输入定理 50.12.** 若 $A$ 是 abelian sheaf of groups，则 $A$-banded gerbes 的等价类由通常的 sheaf cohomology group

$$
H^2(U,A)
$$

分类。这里的 $H^2(U,A)$ 是阿贝尔上同调；只有把 band 推广为非交换群、crossed module 或更高群对象后，才进入相应的非阿贝尔 $H^2$ 型分类。

## 50.5 Cech cocycles

**命题 50.13.** 对 sheaf of groups $G$，覆盖 $\{U_i\to U\}$ 上的 $G$-torsor descent datum 给出 1-cocycle

$$
g_{ij}\in G(U_{ij})
$$

满足 $g_{ij}g_{jk}=g_{ik}$。

**证明.** 局部平凡化 $P|_{U_i}\cong G|_{U_i}$ 后，重叠 $U_{ij}$ 上两个平凡化的比较为 $G$-equivariant automorphism of $G$，由右乘某个 $g_{ij}\in G(U_{ij})$ 给出。三重交上 descent cocycle 条件正是

$$
g_{ij}g_{jk}=g_{ik}.
$$

$\square$

## 50.6 Higher stacks

**定义 50.14.** Higher stack 是满足超下降的 space-valued 或 $\infty$-groupoid-valued sheaf

$$
F:\mathcal C^{op}\to\mathcal S.
$$

**命题 50.15.** Groupoid-valued stacks 经 nerve 全忠实嵌入 hypercomplete space-valued stacks 的 $1$-截断部分。

**证明.** Groupoid 的 nerve 是 1-truncated space。把 groupoid-valued stack $F$ 逐点取 nerve，得到 space-valued presheaf $NF$。Groupoid descent 等价在 nerve 后变为 1-truncated spaces 的 descent 等价。$\infty$-topos 中每个 $n$-截断对象都 hypercomplete，所以 $NF$ 自动满足 hyperdescent；nerve 在 groupoids 上全忠实，故得到所述嵌入。$\square$

**命题 50.16.** 若 $F$ 是 groupoid-valued stack，则任意两个对象 $x,y\in F(U)$ 的 isomorphism presheaf

$$
V\longmapsto \operatorname{Iso}_{F(V)}(x|_V,y|_V)
$$

是 $U$ 上的 sheaf。

**证明.** Stack 条件中函子

$$
F(U)\to\operatorname{Desc}(F,U_\bullet)
$$

是 groupoids 的等价，特别是完全忠实。对覆盖 $\{U_i\to U\}$，一族局部同构 $\alpha_i:x|_{U_i}\to y|_{U_i}$ 若在重叠上相容，就给出 descent groupoid 中从 $x$ 的 descent datum 到 $y$ 的 descent datum 的态射。完全忠实性给出唯一全局同构 $\alpha:x\to y$ 粘合这些 $\alpha_i$。这正是 isomorphism presheaf 的 sheaf 条件。$\square$

## 50.7 从同构粘合到非阿贝尔上同调

Stacks 把 sheaf 条件从元素提升到对象和同构；torsors 由 classifying stacks 表示；gerbes 是高一阶的局部对象粘合结构；非阿贝尔上同调用 cocycles、torsors、gerbes 和 higher stacks 统一描述局部到整体的 obstruction。它是 descent、几何栈和高阶 topos 的核心桥梁。

## 练习

**练习 50.1.** 定义 groupoid-valued prestack。

**练习 50.2.** 定义 stack 条件。

**练习 50.3.** 证明 set-valued sheaf 是离散 stack。

**练习 50.4.** 写出 descent datum。

**练习 50.5.** 说明 stack 条件如何给有效粘合。

**练习 50.6.** 定义 $G$-torsor。

**练习 50.7.** 定义 classifying stack $BG$。

**练习 50.8.** 说明 $H^1(U,G)$ 与 torsors 的关系。

**练习 50.9.** 证明平凡群的 classifying stack 为终 stack。

**练习 50.10.** 定义 gerbe 和 $A$-banded gerbe。

**练习 50.11.** 说明 $H^2(U,A)$ 与 gerbes 的关系。

**练习 50.12.** 从 torsor descent datum 写出 Cech 1-cocycle。

**练习 50.13.** 定义 higher stack。

**练习 50.14.** 说明 1-stack 如何嵌入 higher stack。

**练习 50.15.** 证明 stack 中两个对象之间的 isomorphism presheaf 是 sheaf。
