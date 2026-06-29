# 附录 B：形式化蓝图

## 目标

本附录给出把本书内容迁移到 Coq-HoTT、UniMath 或 Cubical Agda 的蓝图。它不是可直接运行的代码，而是检查依赖和定义口径的清单。

## B.1 路径代数蓝图

目标：形式化第二章和附录 A。

依赖：

- identity type / path type；
- path induction；
- path inverse and concat；
- `ap`；
- transport。

检查点：

- 复合方向是否与库一致；
- 反身路径命名是否为 `idpath`、`refl` 或 `refl` 变体；
- 路径复合记号是否为 `@`、`∙` 或其他。

## B.2 等价蓝图

目标：形式化第五章。

依赖：

- fiber；
- contractible；
- equivalence；
- quasi-inverse / half-adjoint equivalence；
- equivalence composition and inverse。

检查点：

- 库中 `IsEquiv` 的基准定义；
- `Equiv` 是否为 record、sigma type 或 typeclass；
- 等价证明是否是命题；
- 是否启用 univalence。

## B.3 单值性蓝图

目标：形式化第六至第七章。

依赖：

- `idtoequiv`；
- univalence axiom or cubical univalence；
- function extensionality；
- propositional extensionality；
- transport of structures。

检查点：

- 公理化 HoTT 中 `ua` 的计算只给路径；
- Cubical Agda 中 path abstraction 和 Glue 可能给出更强计算；
- 结构等同性是否已有库定理可用。

## B.4 圆的基本群蓝图

目标：形式化第十一章。

子任务：

1.  定义整数 $\mathbb Z$；
2.  证明后继函数是等价；
3.  用圆递归定义 $\mathsf{code}:\mathbb S^1\to\mathcal U$；
4.  定义 encode；
5.  定义 decode；
6.  证明 encode/decode 互逆；
7.  把 loop space 等价转化为基本群同构。

检查点：

- 圆的 HIT 是否可用；
- transport 沿 $\mathsf{loop}$ 的计算规则强度；
- 整数加法与路径复合的相容性；
- 基本群是否定义为集合截断。

## B.5 单值范畴蓝图

目标：形式化第十三至第十四章。

依赖：

- set-level Hom；
- precategory；
- isomorphism；
- `idtoiso`；
- univalent category；
- functor, natural transformation；
- Yoneda and Rezk completion。

检查点：

- 对象类型是否允许高阶；
- Hom 集合性如何编码；
- 自然变换相等是否使用函数外延性；
- essentially surjective 是否使用命题截断。
