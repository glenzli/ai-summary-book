# 第十一章：基本群、覆盖空间与圆的计算

“圆有一条生成 loop”并不能单独证明其基本群是 $\mathbb Z$：还必须证明每条 loop 都由唯一整数次幂表示。HoTT 的 encode–decode 方法把这个唯一性问题变成依赖类型计算。先让圆的生成 loop 在整数 fiber 上作用为后继，再把任意路径编码成沿路径 transport 后的整数；反向则把整数解码为 loop 的正幂或负幂。

这一构造同时调用前面几乎全部工具：圆的 HIT 递归定义覆盖族，单值性把整数后继等价变成宇宙路径，路径归纳定义 encode，依赖消去定义 decode，集合截断最后把 loop space 变成普通群。整数及其群律在附录 M、W 中构造，圆规则在附录 L 中固定，逐项的两侧逆证明位于附录 N。

## 11.1 Loop space 与基本群

**定义 11.1.** 基点类型为二元组 $(X,x_0)$，其中 $X:\mathcal U$ 且 $x_0:X$。其 loop space 为
$$
\Omega(X,x_0)\coloneqq(x_0=x_0).
$$

**定义 11.2.** 基本群定义为 loop space 的集合截断：
$$
\pi_1(X,x_0)\coloneqq\|\Omega(X,x_0)\|_0.
$$
乘法由路径复合诱导，单位由 $\mathsf{refl}_{x_0}$ 诱导，逆由路径逆诱导。

**命题 11.3（基本群是群）.** $\pi_1(X,x_0)$ 具有群结构。

**证明（书内证明核）.** 见附录 V.1-V.2。路径复合满足结合律、左右单位律和逆律；由于目标是集合截断，这些运算和律可由集合截断递归/归纳下降到 $\|\Omega(X,x_0)\|_0$。$\square$

## 11.2 覆盖族

**定义 11.4.** 在 HoTT 中，基于 $X$ 的覆盖可表示为类型族
$$
P:X\to\mathcal U
$$
使每个 fiber $P(x)$ 是集合，常在圆的计算中取具体族 $P:\mathbb S^1\to\mathcal U$。

**定义 11.5（总空间）.** 覆盖 $P:X\to\mathcal U$ 的总空间为
$$
\sum_{x:X}P(x).
$$

**例 11.6（圆的 universal cover 思路）.** 通过圆的递归原则定义族
$$
\mathsf{code}:\mathbb S^1\to\mathcal U
$$
使 $\mathsf{code}(\mathsf{base})\equiv\mathbb Z$，而沿 $\mathsf{loop}$ 的 transport 对应整数后继自等价。

**构造 11.7（code 覆盖的输入数据）.** 取
$$
\mathbb Z\coloneqq\mathbb Z_{\mathsf{ind}}
$$
为附录 M 的归纳整数，并取 successor 自等价
$$
\mathsf{succEquiv}_{\mathbb Z}:\mathbb Z\simeq\mathbb Z
$$
为定义 M.10。由单值性得到宇宙中的路径
$$
\mathsf{ua}(\mathsf{succEquiv}_{\mathbb Z}):\mathbb Z=\mathbb Z.
$$
用圆递归原则 L.15 在宇宙中定义
$$
\mathsf{code}:\mathbb S^1\to\mathcal U
$$
并令
$$
\mathsf{code}(\mathsf{base})=\mathbb Z,
\qquad
\mathsf{ap}_{\mathsf{code}}(\mathsf{loop})
=
\mathsf{ua}(\mathsf{succEquiv}_{\mathbb Z}).
$$
沿 $\mathsf{loop}$ 的 transport 与 $\mathsf{succ}_{\mathbb Z}$ 一致，依赖的是单值性计算定理
$$
\mathsf{idtoequiv}(\mathsf{ua}(e))=e.
$$
该定理在第六章作为单值性的等价方向使用；若采用 cubical 口径，应替换为 Glue/univalence 的对应计算规则。

## 11.3 Encode-decode 方法

**定义 11.8.** 对圆，设
$$
\mathsf{code}:\mathbb S^1\to\mathcal U
$$
为上节覆盖。定义
$$
\mathsf{encode}:\prod_{x:\mathbb S^1}(\mathsf{base}=x)\to\mathsf{code}(x)
$$
为沿路径 transport 整数 $0$。

**定义 11.9.** 定义
$$
\mathsf{decode}:\prod_{x:\mathbb S^1}\mathsf{code}(x)\to(\mathsf{base}=x)
$$
使用圆的依赖消去。基点处的函数
$$
\mathbb Z\to(\mathsf{base}=\mathsf{base})
$$
由附录 M.15 的 loop 幂给出，把整数 $n$ 送到 $\mathsf{loop}^n$。关键相干条件由整数 successor 与右复合 $\mathsf{loop}$ 的关系给出。

**定理 11.10（圆的 loop space）.** 有等价
$$
(\mathsf{base}=\mathsf{base})\simeq\mathbb Z.
$$

**证明（书内证明核）.** 见附录 N.11。附录 N.2-N.3 构造 $\mathsf{code}$ 并计算 loop transport，N.4-N.7 构造 $\mathsf{encode}$ 与 $\mathsf{decode}$，N.8-N.10 证明两侧互逆，再由附录 G 中“准逆推出等价”得到所述等价。本书圆规则规定点构造子上的 judgmental 计算和 loop 构造子上的 propositional 计算；附录 N 的端点对齐按这一固定口径进行。$\square$

**推论 11.11.** 基本群 $\pi_1(\mathbb S^1,\mathsf{base})$ 同构于 $\mathbb Z$。

**证明（书内证明核）。** 见附录 V.10。由定理 11.10 得到底层等价；附录 V.5 证明 loop 幂保持整数加法：
$$
\mathsf{loop}^{z+w}=\mathsf{loop}^z\cdot\mathsf{loop}^w.
$$
因此映射 $z\mapsto[\mathsf{loop}^z]$ 保持群运算，并与 encode 诱导的反向映射互逆。$\square$

## 11.4 证明边界

**警告 11.12.** $\pi_1(\mathbb S^1)\cong\mathbb Z$ 的完整证明不是一句“圆有一条环路”即可推出。关键是构造覆盖 $\mathsf{code}$，证明 transport 沿 $\mathsf{loop}$ 是后继，并建立 encode/decode 的逆性。

## 11.5 从 loop 数据到群同构

覆盖族 $\mathsf{code}$ 把圆的几何生成元变成整数后继的可计算作用，encode 与 decode 再证明 loop space 本身等价于整数。集合截断不是证明的替代，而是把已经算出的高阶路径对象送到群所在的集合层；附录 V 额外核对 loop 复合对应整数加法。这个例子给出后续合成同伦论的基本方法：选择能记录路径信息的依赖族，再用消去原则证明编码完备。

## 练习

**练习 11.1.** 写出 loop space 上路径复合给出的乘法，并证明左单位律。

**练习 11.2.** 说明为什么基本群需要集合截断。

**练习 11.3.** 按附录 M 的归纳整数定义整数后继函数，并逐项展开它与 predecessor 的互逆证明。

**练习 11.4.** 在 encode-decode 证明中，指出 decode 的依赖消去原则需要的相干条件。
