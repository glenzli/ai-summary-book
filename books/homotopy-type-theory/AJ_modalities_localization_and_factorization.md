# 附录 AJ：模态、局部化与正交分解系统

本附录补上 HoTT 教材中不可缺少的模态层。它服务于三个目标：第一，把命题截断和 $n$-截断放入统一框架；第二，为合成同伦论中的连通/截断分解给出精确口径；第三，为局部化、cohesive/modal HoTT 和近期 coslice colimit 结果建立教材接口。

## AJ.1 反射子宇宙

**定义 AJ.1（反射子宇宙）.** 一个反射子宇宙由以下数据组成：

1.  一个类型谓词
    $$
    \mathsf{isLocal}:\mathcal U\to\mathsf{Prop};
    $$
2.  对每个类型 $A$，一个类型 $L A$ 和单位映射
    $$
    \eta_A:A\to L A;
    $$
3.  $L A$ 是 local；
4.  对任意 local 类型 $B$，预合成 $\eta_A$ 给出等价
    $$
    (L A\to B)\simeq(A\to B).
    $$

**定义 AJ.2（$L$-local 映射与 $L$-等价）.** 映射 $f:A\to B$ 称为 $L$-等价，若对任意 local 类型 $X$，预合成
$$
f^\ast:(B\to X)\to(A\to X)
$$
是等价。

**命题 AJ.3（单位映射是 $L$-等价）.** 对任意 $A$，$\eta_A:A\to L A$ 是 $L$-等价。

**证明.** 这正是定义 AJ.1 的泛性质。令 $X$ 为 local 类型，则预合成 $\eta_A$ 给出
$$
(L A\to X)\simeq(A\to X).
$$
因此 $\eta_A$ 满足 $L$-等价定义。$\square$

**命题 AJ.4（local 类型上的单位为等价）.** 若 $A$ 是 local，则 $\eta_A:A\to L A$ 是等价。

**证明.** 因 $A$ local，定义 AJ.1 对 $B=A$ 给出
$$
(L A\to A)\simeq(A\to A).
$$
令 $r:L A\to A$ 为 $\mathsf{id}_A$ 在该等价下的扩张。于是
$$
r\circ\eta_A\sim\mathsf{id}_A.
$$
另一方面，$L A$ local，两个映射 $L A\to L A$ 若预合成 $\eta_A$ 后同伦，则由 $\eta_A$ 的泛性质和函数外延性相等。现在
$$
(\eta_A\circ r)\circ\eta_A\sim\eta_A\circ(r\circ\eta_A)\sim\eta_A
$$
且 $\mathsf{id}_{L A}\circ\eta_A=\eta_A$，故 $\eta_A\circ r\sim\mathsf{id}_{L A}$。所以 $\eta_A$ 有准逆 $r$，是等价。$\square$

## AJ.2 模态

**定义 AJ.5（模态）.** 反射子宇宙 $L$ 称为模态，若它对依赖和封闭：对任意 $A$ 和族 $B:L A\to\mathcal U$，若每个 $B(y)$ local，则
$$
\prod_{y:L A}B(y)
$$
满足相应 dependent elimination；等价地，$L$-local 类型在依赖函数类型下稳定，并且反射满足依赖消去原则。

**定义 AJ.6（left exact 模态）.** 模态 $L$ 称为 left exact，若它保持有限极限；教材层可采用等价形式：
$$
L\Big(\sum_{x:A}B(x)\Big)\to
\sum_{u:L A}L(B_u)
$$
在合适 transport 后为等价，并且 $L\mathbf 1\simeq\mathbf 1$。

这里 $B_u$ 表示把族 $B$ 沿 $\eta_A$ 延拓到 $L A$ 后的 fiber。严格形式需要在 universe 层固定 extension 数据；因此本书把 left exactness 当作额外结构，不从普通模态推出。

**命题 AJ.7（命题截断是模态）.** 命题截断
$$
\|-\|_{-1}:\mathcal U\to\mathcal U
$$
是模态。

**证明（证明核）.** local 类型为 mere propositions。单位为 $|-|:A\to\|A\|_{-1}$。若 $P$ 是命题，则非依赖消去给出
$$
(\|A\|_{-1}\to P)\to(A\to P),
$$
反向由命题截断递归给出；二者互逆由函数外延性和 $P$ 的命题性证明。依赖消去情形中，若 $B:\|A\|_{-1}\to\mathcal U$ 且每个 $B(z)$ 是命题，则由截断归纳从 $A$ 上的数据延拓，并由命题性保证相干唯一。因此命题截断是模态。$\square$

**命题 AJ.8（$n$-截断是模态）.** 对每个 $n\ge -1$，$n$-截断 $\|-\|_n$ 是模态。

**证明状态.** 书内证明依赖一般 $n$-截断 HIT 的依赖消去和 h-level 稳定性。附录 L 给出输入规则，附录 AB 给出 h-level 向上闭包，故本书可把该结论作为由这些输入推出的证明核。

## AJ.3 连通映射与正交分解

**定义 AJ.9（$n$-连通映射）.** 映射 $f:A\to B$ 称为 $n$-连通，若对每个 $b:B$，其 fiber
$$
\mathsf{fib}_f(b)\coloneqq\sum_{a:A}f(a)=b
$$
的 $n$-截断可收缩：
$$
\mathsf{isContr}\bigl(\|\mathsf{fib}_f(b)\|_n\bigr).
$$

**定义 AJ.10（$n$-截断映射）.** 映射 $f:A\to B$ 称为 $n$-截断，若每个 fiber 是 $n$-型。

**定理 AJ.11（连通/截断分解）.** 对任意映射 $f:A\to B$，定义
$$
I_f\coloneqq\sum_{b:B}\|\mathsf{fib}_f(b)\|_n.
$$
有分解
$$
A\xrightarrow{c_f} I_f\xrightarrow{t_f}B,
$$
其中
$$
c_f(a)\coloneqq(f(a), |(a,\mathsf{refl}_{f(a)})|)
,\qquad
t_f(b,u)\coloneqq b.
$$
则 $c_f$ 是 $n$-连通，$t_f$ 是 $n$-截断。

**证明（证明核）.** 对 $t_f$，其在 $b:B$ 处的 fiber 等价于 $\|\mathsf{fib}_f(b)\|_n$：fiber 元是 $((b',u),p:b'=b)$，沿 $p$ transport 后化为 $u:\|\mathsf{fib}_f(b)\|_n$。因此它是 $n$-型。

对 $c_f$，固定 $(b,u):I_f$。其 fiber 展开为
$$
\sum_{a:A}(f(a),|(a,\mathsf{refl})|)=(b,u).
$$
由 $\Sigma$-路径等价，这等价于
$$
\sum_{(a,q):\mathsf{fib}_f(b)} |(a,q)|=u.
$$
对目标取 $n$-截断。因 $\|\mathsf{fib}_f(b)\|_n$ 是 $n$-型，类型
$$
\sum_{v:\|\mathsf{fib}_f(b)\|_n} v=u
$$
可收缩，中心为 $(u,\mathsf{refl})$。截断归纳把 $u$ 归约到 $|(a,q)|$ 的代表元情形，得到所需可收缩性。$\square$

**定理 AJ.12（正交性，外部输入 / 证明核）.** 若 $e:A\to B$ 是 $n$-连通，$m:X\to Y$ 是 $n$-截断，则任意交换方块
$$
\begin{array}{ccc}
A&\to&X\\
\downarrow e&&\downarrow m\\
B&\to&Y
\end{array}
$$
的 filler 类型是可收缩的。

**证明状态.** 证明把 filler 类型化为对 $b:B$ 的 fiber 中心选择问题；$e$ 的 $n$-连通性给出 fiber 的 $n$-截断可收缩，$m$ 的 fiber 是 $n$-型，因此从截断上的数据唯一延拓。完整文本展开需要使用依赖函数外延性、fiber transport 和截断归纳。本书把它作为标准正交分解证明核，来源为模态与 factorization systems 文献。

## AJ.4 局部化

**定义 AJ.13（关于映射族的 local 类型）.** 给定映射族
$$
S_i:A_i\to B_i\qquad(i:I),
$$
类型 $X$ 称为 $S$-local，若每个预合成
$$
(B_i\to X)\to(A_i\to X)
$$
是等价。

**输入 AJ.14（局部化 HIT）.** 对映射族 $S$，存在反射
$$
\eta_A:A\to L_S A
$$
到 $S$-local 类型的反射子宇宙，并满足 AJ.1 的泛性质。

**使用边界.** 该构造通常用 localization HIT 给出。它不是本书基础类型形成规则；在使用时必须列入 HIT 输入或引用相应论文来源。

**命题 AJ.15（局部化保持 $S$-等价的目标泛性质）.** 若 $X$ 是 $S$-local，则
$$
(L_S A\to X)\simeq(A\to X).
$$

**证明.** 这是输入 AJ.14 的反射泛性质。$\square$

**定理 AJ.16（素数局部化的同伦群口径，外部输入）.** 对 pointed simply connected 类型 $X$，素数 $p$ 处局部化映射
$$
X\to X_{(p)}
$$
在所有同伦群上诱导代数意义的 $p$-局部化。

**证明状态.** 这是 Christensen-Opie-Rijke-Scoccola 的 HoTT 局部化定理。书内可使用它作为高级合成同伦论输入，但不能把它回流为基础层定理。

## AJ.5 Coslice colimit 与模态交互

**定义 AJ.17（coslice colimit）.** 固定对象 $A:\mathcal U$。coslice $\mathcal U_{A/}$ 的对象为映射 $A\to X$。给定图形
$$
D:J\to\mathcal U_{A/},
$$
其 coslice colimit 是对象 $A\to C$，使得对任意 $A\to Y$，诱导的锥空间满足相应泛性质。

**事实 AJ.18（树形图的遗忘函子创建余极限，外部输入）.** 对图形为树的情形，遗忘函子
$$
\mathcal U_{A/}\to\mathcal U
$$
创建相应余极限。

**使用边界.** Hart-Hou 的 2024/2026 版本证明了 coslice colimit 与普通 colimit 的连接。其与正交分解系统的交互推出若干 pointed colimit 保持 $n$-连通性。本书把它作为第十章 pushout 后的进阶余极限工具，而非基础 HIT 规则。

## AJ.6 本附录的接口

1.  第八章的截断由 AJ.5-AJ.8 统一解释为模态。
2.  第十章的 pushout 与 cofiber 可在 AJ.9-AJ.12 的连通/截断分解下研究。
3.  第十二章的上同调和稳定方向可使用 AJ.13-AJ.18 的局部化与 coslice colimit 结果作为高级输入。
4.  附录 AT 继续展开 left exact 模态、cohesive HoTT 和 modal induction。
5.  第十七章讨论 cohesive/modal HoTT 时必须区分普通模态、left exact 模态、局部化和具体模型。
