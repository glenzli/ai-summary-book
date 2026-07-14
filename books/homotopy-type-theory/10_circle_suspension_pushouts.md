# 第十章：圆、悬挂、Pushout 与同伦余极限

圆用一条 loop 记录一维粘合；悬挂把每个 $a:A$ 变成南北两点之间的一条 meridian；pushout 则为 span 的每个输入点加入一条 glue 路径。三者看似是不同空间，实际共享同一种构造原则：向目标 $X$ 给映射，等价于在所有生成元上给数据，并证明这些数据沿粘合路径相容。

本章把第九章的 HIT 规则用于这些具体构造。圆的映射由“点加环路”描述，悬挂把一族路径装进一个类型，pushout 则实现基本的同伦余极限；wedge、cofiber 与 smash product 随后由 pushout 组合得到。所用路径代数、等价和截断均已建立，具体 HIT 规则固定在附录 L.14–L.20。

## 10.1 圆的递归与映射空间

**命题 10.1（圆到类型的映射数据）.** 对任意类型 $A$，从 $\mathbb S^1$ 到 $A$ 的函数由点 $a:A$ 和环路 $\ell:a=a$ 给出。

**验证状态：HIT 递归原则。** 这是圆的非依赖递归原则。若要证明映射类型
$$
(\mathbb S^1\to A)\simeq\sum_{a:A}(a=a)
$$
还需要函数外延性和圆的依赖消去原则。完整证明需要逐项处理圆的依赖消去、transport 计算和整数后继相容性。

**例 10.2.** 取 $A\equiv\mathbb S^1$，点 $\mathsf{base}$ 与环路 $\mathsf{loop}$ 给出恒等映射的候选数据。

## 10.2 悬挂

**规则 10.3（悬挂）.** 对类型 $A$，悬挂 $\mathsf{susp}(A)$ 由两个点和一族路径生成：
$$
\mathsf{north}:\mathsf{susp}(A),\qquad
\mathsf{south}:\mathsf{susp}(A),
$$
$$
\mathsf{merid}:A\to(\mathsf{north}=\mathsf{south}).
$$

**规则 10.4（悬挂递归）.** 要定义 $f:\mathsf{susp}(A)\to B$，需给出
$$
n:B,\qquad s:B,\qquad m:\prod_{a:A}(n=s).
$$

**例 10.5.** $\mathsf{susp}(\mathbf 2)$ 与圆等价。

**证明（书内证明核）。** 见附录 AD。两个方向的函数分别由悬挂和圆的递归原则定义：悬挂的两个 meridian 分别送到 $\mathsf{refl}_{\mathsf{base}}$ 和 $\mathsf{loop}$；反向把 $\mathsf{loop}$ 送到 $\mathsf{merid}(1)\cdot\mathsf{merid}(0)^{-1}$。两个复合的同伦由圆和悬挂的依赖消去以及路径代数给出。$\square$

## 10.3 Pushout

**规则 10.6（Pushout）.** 给定
$$
f:A\to B,\qquad g:A\to C,
$$
pushout $\mathsf{pushout}(f,g)$ 由以下构造子生成：
$$
\mathsf{inl}:B\to\mathsf{pushout}(f,g),
$$
$$
\mathsf{inr}:C\to\mathsf{pushout}(f,g),
$$
$$
\mathsf{glue}:\prod_{a:A}\mathsf{inl}(f(a))=\mathsf{inr}(g(a)).
$$

**命题 10.7（Pushout 的递归泛性质）.** 要定义 $\mathsf{pushout}(f,g)\to X$，等价于给出 $u:B\to X$、$v:C\to X$ 和同伦
$$
\prod_{a:A}u(f(a))=v(g(a)).
$$

**验证状态：HIT 递归原则。** 完整等价形式需要函数外延性和依赖消去原则。

## 10.4 同伦余极限

**定义 10.8.** 在 HoTT 中，许多同伦余极限由 HIT 给出。Pushout 是基本二元同伦余极限；cofiber、wedge、smash product 可由 pushout 和悬挂组合构造。

**例 10.9（Cofiber）.** 对 $f:A\to B$，其 cofiber 可定义为 pushout
$$
\mathsf{cofib}(f)\coloneqq\mathsf{pushout}(f,!),
$$
其中 $!:A\to\mathbf 1$。这把 $A$ 的像在 $B$ 中压到一个点。

**例 10.10（Wedge）.** 基点类型 $(A,a_0)$ 与 $(B,b_0)$ 的 wedge 可把两个基点识别：
$$
A\vee B\coloneqq\mathsf{pushout}(\mathbf 1\xrightarrow{a_0}A,\mathbf 1\xrightarrow{b_0}B).
$$

**例 10.10.1（Smash product）.** 对 pointed 类型 $X,Y$，smash product $X\wedge Y$ 定义为 wedge 嵌入
$$
X\vee Y\to X\times Y
$$
的 cofiber。它是稳定同伦论和 cup product 几何来源的基础构造；递归泛性质、球面 smash 和对称幺半结构见附录 AM。

## 10.5 证明纪律

**警告 10.11.** 许多传统拓扑语句在 HoTT 中需要重写为 HIT 的泛性质。例如“把子空间压成点”不是集合商直觉，而是 cofiber HIT 的递归和消去原则。

**命题 10.12（HIT 构造的等价不变性）.** 若输入图中的类型和映射被等价替换，则由 HIT 定义的同伦余极限等价。

**证明（pushout 证明核）。** 见附录 AI。对 pushout，使用两个 span 之间的等价和相干方块构造双向递归函数，再用 pushout 依赖消去证明复合与恒等同伦。cofiber、wedge 等基础同伦余极限由 pushout 表达，逐次应用 pushout 情形。更一般的同伦余极限函子性属于高阶 HIT 元理论。$\square$

## 10.6 粘合数据如何成为泛性质

圆、悬挂和 pushout 都把几何粘合翻译成构造子与消去原则。Cofiber 不是一句“把像压到一点”，wedge 也不是未经定义的子空间并；它们是具体 pushout，因而带有可类型检查的递归泛性质。下一章会对圆选择一个宇宙值目标，把整数后继等价作为 loop 上的粘合数据，由此得到覆盖族并开始真正的同伦群计算。

## 练习

**练习 10.1.** 用 pushout 定义两个基点类型的 smash product。

**练习 10.2.** 写出 $\mathsf{susp}(\mathbf 0)$ 的点构造和路径构造，并猜测其等价类型。

**练习 10.3.** 证明 pushout 到集合 $X$ 的函数由相容的一对函数给出。

**练习 10.4.** 构造从 $\mathsf{susp}(\mathbf 2)$ 到 $\mathbb S^1$ 的函数。
