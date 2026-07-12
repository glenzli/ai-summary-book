# 序章：范围、严格性和 Langlands 主线

## 本章目标

本章说明本书的写作范围、严格性标准、资料源口径和全书路线。本章不发展技术理论；正式定义从第一章开始。

## 0.1 目标问题

Langlands 纲领的核心目标不是证明某一个孤立定理，而是组织以下两类对象之间的深层对应：

1. 数论侧：整体域和局部域上的 Galois 群、Weil 群、Weil-Deligne 群及其表示。
2. 分析和表示论侧：还原群在局部域或 adele 环上的表示，尤其是自守表示。

最粗略的口号是：Galois 参数应当控制自守表示，而两侧的 L 函数、局部因子、epsilon 因子和函子性应当相容。本书不把这句话作为定义；本书的任务是逐步定义其中每个词。

## 0.2 为什么必须从 adeles 开始

若只从经典模形式进入 Langlands 纲领，会很快遇到三个障碍。

第一，经典模形式把所有有限素数的信息压缩到 Fourier 系数和 Hecke 算子中，而 Langlands 纲领需要逐素数地分解局部表示。

第二，Galois 表示天然有局部化：对每个素数 $p$ 有分解群和惯性群。自守侧也必须有相应的局部分量；adele 群 $G(\mathbb A_K)$ 正是把所有局部群 $G(K_v)$ 组织为一个整体对象的语言。

第三，类域论已经告诉我们，交换化 Galois 理论不是由 $K^\times$ 本身控制，而是由 idele class group
$$
C_K=K^\times\backslash\mathbb A_K^\times
$$
控制。`GL(1)` Langlands 就是把这一事实翻译为表示论语言。

因此本书从整体域、局部域、adeles、ideles 和 Tate thesis 开始，而不是直接从椭圆曲线或模形式开始。

## 0.3 严格性标准

本书采用如下标准：

- 定义必须给出完整数据和条件。
- 命题必须说明精确假设和结论。
- 证明必须说明使用的定理、自然性、局部-整体分解或泛性质。
- 暂不证明的大定理必须标注为“外部输入定理”。
- 同一对象的不同模型必须区分。例如，模形式、adelic automorphic forms 和自守表示不是同一个定义。
- “Langlands 对应”必须说明是哪一个群、哪个域、局部还是整体、已知定理还是猜想。

例如，不能只说“`GL(1)` Langlands 就是类域论”。必须写出 reciprocity map、idele class characters 和一维 Galois/Weil 表示之间的对应，并说明局部 L 因子如何相容。

同样，不能只说“费马大定理来自 Langlands 纲领”。本书会把逻辑链写成：

$$
\text{Frey 曲线}
\quad+\quad
\text{半稳定椭圆曲线模性}
\quad+\quad
\text{Ribet 降层}
\quad\Longrightarrow\quad
S_2(\Gamma_0(2))\ne 0,
$$

再用 $S_2(\Gamma_0(2))=0$ 得出矛盾。

## 0.4 本书路线

第一部分建立局部-整体语言：

- 整体域、局部域、赋值和完备化。
- Adele 环、idele 群和 idele class group。
- Haar 测度、Fourier 变换和 Poisson summation。
- Tate thesis 和 `GL(1)` L 函数。
- 类域论作为 `GL(1)` Langlands。

第二部分进入 `GL(2)`：

- 经典模形式和 Hecke 算子。
- Adelic 模形式和 `GL(2)` 自守表示。
- 椭圆曲线的 Hasse-Weil L 函数。
- Galois 表示、导子和局部因子。
- 模性定理、降层和费马大定理。

第三部分进入一般形式：

- 还原群、root datum、对偶群和 L 群。
- 局部 Langlands 参数。
- 全局自守表示和标准 L 函数。
- 函子性、trace formula、endoscopy 和 Arthur 参数。

第四部分进入几何 Langlands：

- 曲线上的 $G$-bundles。
- Hecke 修改和 Hecke 算子。
- 几何 Satake。
- Hecke eigensheaves。
- 谱侧和自守侧的范畴化对应。

## 0.5 关于证明的边界

完整 Langlands 纲领不是一本书能完全证明的定理集合。许多核心陈述仍是猜想；许多已知定理的证明需要整套专门技术。本书采取以下处理方式：

- 基础性构造尽量证明，例如 restricted product 的基本性质、Euler 乘积分解和简单的维数计算。
- 中型定理给出外部输入的证明路线，并列出缺失引理。
- 大型定理作为外部输入，例如 Tate thesis 的完整函数方程、全局类域论、Wiles-Taylor-Wiles 模性定理、Ribet 降层、局部 Langlands for `GL(n)`、Arthur 分类。
- 每次使用外部输入时，必须说明它在当前论证中承担的精确角色。

这种写法的目标是让读者知道自己已经理解了什么、还借用了什么，而不是把大定理压缩成不可检查的口号。

## 0.6 本章小结

Langlands 纲领把 Galois/Weil 参数、自守表示和 L 函数组织成一个局部-整体系统。理解它的最低入口不是某个单独定义，而是一组相容语言：adeles、ideles、Haar 测度、表示论、Galois 表示和 L 函数。本书按这条主线展开。

## 练习

**练习 0.1.** 解释为什么 $K^\times$ 本身不足以作为全局类域论的主角，而 idele class group $C_K$ 更合适。此题只要求给出结构性理由，不要求证明类域论。

**练习 0.2.** 给出一个局部对象和一个整体对象的例子，并说明二者之间的关系。

**练习 0.3.** 查找一个你熟悉的 L 函数，写出它的 Euler 乘积形式，并指出其中的局部因子。
