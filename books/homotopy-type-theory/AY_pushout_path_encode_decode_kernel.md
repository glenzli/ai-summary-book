# 附录 AY：Pushout 路径空间的 Encode-Decode 证明核

Blakers-Massey 证明的技术核心之一是 pushout 中的路径空间刻画。本附录给出 pushout path 的 encode-decode 证明架构，用于支撑附录 AU 的 flattening 与 gap map 计算。

固定 span
$$
B\xleftarrow{f}A\xrightarrow{g}C
$$
和 pushout
$$
P\coloneqq B\sqcup_A C
$$
构造子为 $\mathsf{inl}$、$\mathsf{inr}$、$\mathsf{glue}$。

## AY.1 Code family

**定义 AY.1（left-right code）.** 对 $b:B$、$c:C$，定义类型
$$
\mathsf{Code}_{LR}(b,c)
$$
为由 zigzag 生成的类型，基本生成元为
$$
(a:A)\times(f(a)=b)\times(g(a)=c).
$$
严格版本取该 span-fiber 的适当 join/截断闭包，以便表达 pushout 中任意路径
$$
\mathsf{inl}(b)=\mathsf{inr}(c).
$$

**定义 AY.2（dependent code）.** 固定 $b_0:B$。定义族
$$
\mathsf{code}_{b_0}:P\to\mathcal U
$$
使
$$
\mathsf{code}_{b_0}(\mathsf{inl}(b))\coloneqq
(\mathsf{inl}(b_0)=\mathsf{inl}(b)),
$$
并在右侧点
$$
\mathsf{code}_{b_0}(\mathsf{inr}(c))
$$
取 $\mathsf{Code}_{LR}(b_0,c)$ 的规范表示。

**构造义务 AY.3（glue 相容）.** 对每个 $a:A$，需给出沿
$$
\mathsf{glue}_a:\mathsf{inl}(f(a))=\mathsf{inr}(g(a))
$$
的 transport 等价
$$
\mathsf{transport}^{\mathsf{code}_{b_0}}(\mathsf{glue}_a):
\mathsf{code}_{b_0}(\mathsf{inl}(f(a)))
\simeq
\mathsf{code}_{b_0}(\mathsf{inr}(g(a))).
$$
该等价把左侧路径 $p:\mathsf{inl}(b_0)=\mathsf{inl}(f(a))$ 送到由 $p$ 后接 $\mathsf{glue}_a$ 得到的 left-right zigzag。

## AY.2 Encode and decode

**定义 AY.4（encode）.** 对 $x:P$，定义
$$
\mathsf{encode}_x:
(\mathsf{inl}(b_0)=x)\to\mathsf{code}_{b_0}(x)
$$
为
$$
\mathsf{encode}_x(p)\coloneqq
\mathsf{transport}^{\mathsf{code}_{b_0}}(p)(\mathsf{refl}_{\mathsf{inl}(b_0)}).
$$

**定义 AY.5（decode）.** 定义
$$
\mathsf{decode}_x:
\mathsf{code}_{b_0}(x)\to(\mathsf{inl}(b_0)=x)
$$
对 $x$ 作 pushout 依赖消去。左侧 $x=\mathsf{inl}(b)$ 时，$\mathsf{code}$ 已定义为路径空间，取恒等。右侧 $x=\mathsf{inr}(c)$ 时，把 zigzag 逐段解释为 pushout 路径：生成元 $(a,p,q)$ 给出
$$
\mathsf{ap}_{\mathsf{inl}}(p)^{-1}\cdot
\mathsf{glue}_a\cdot
\mathsf{ap}_{\mathsf{inr}}(q).
$$

## AY.3 互逆性

**定理 AY.6（decode-after-encode）.** 对任意 $x:P$ 和 $p:\mathsf{inl}(b_0)=x$，
$$
\mathsf{decode}_x(\mathsf{encode}_x(p))=p.
$$

**证明（证明核）.** 对 $p$ 作路径归纳，归约到 $x=\mathsf{inl}(b_0)$ 和 $p=\mathsf{refl}$。此时 encode 为 $\mathsf{refl}$ 的 transport，decode 为恒等，故成立。$\square$

**定理 AY.7（encode-after-decode）.** 对任意 $x:P$ 和 $z:\mathsf{code}_{b_0}(x)$，
$$
\mathsf{encode}_x(\mathsf{decode}_x(z))=z.
$$

**证明状态.** 对 $x$ 作 pushout 依赖消去。左侧 $\mathsf{inl}(b)$ 情形由路径归纳。右侧 $\mathsf{inr}(c)$ 情形对 code 的 zigzag 生成方式归纳：生成元由 AY.3 的 transport 相容和 pushout path constructor 的计算规则给出；复合和逆由路径代数附录 A 处理。若 code 采用 join/截断闭包，还需对 join 和截断构造子分别检查相干。$\square$

## AY.4 Blakers-Massey 中的使用

**命题 AY.8（gap fiber code）.** 在 pushout 方块中，gap map
$$
A\to B\times_PC
$$
在 $(b,c,p)$ 处的 fiber 等价于某个由
$$
\mathsf{fib}_f(b)
\quad\text{和}\quad
\mathsf{fib}_g(c)
$$
组成的 join-code fiber。

**证明核.** 用 AY.4-AY.7 把路径 $p:\mathsf{inl}(b)=\mathsf{inr}(c)$ 替换为 left-right code；再展开 gap fiber 的 $\Sigma$-路径。得到的数据正是选择一个左 fiber 点、右 fiber 点以及连接 zigzag 的 join 数据。$\square$

**剩余证明义务 AY.9.** 需要固定 code 的具体定义：free zigzag HIT、join-code 或 truncation-code。不同定义给出等价证明，但证明项结构的复杂度不同。
