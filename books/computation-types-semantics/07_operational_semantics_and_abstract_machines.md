# 第 7 章：操作语义、唯一分解与 CEK 机

类型规则约束程序的静态形状，运行规则则描述配置如何变化；两者是不同关系。操作语义还可以有不同粒度：大步语义只记录终止结果，小步语义保留中间配置，抽象机进一步把求值上下文实现为显式栈和环境。本章先为一个 while 语言证明大小步在终止运行上的双向等价，再证明唯一分解蕴含确定性，最后用卸载函数逐步比较 CBV λ 演算与 CEK 机。

## 7.1 While 语言的大小步语义

**定义 7.1（命令与状态）。**
$$
c::=\mathsf{skip}\mid x:=a\mid c;c\mid
\mathsf{if}\ b\ \mathsf{then}\ c\ \mathsf{else}\ c\mid
\mathsf{while}\ b\ \mathsf{do}\ c.
$$
状态 $\sigma:\mathsf{Var}\to\mathbb Z$ 是全映射；算术、布尔表达式解释为全函数
$\llbracket a\rrbracket\sigma\in\mathbb Z$、
$\llbracket b\rrbracket\sigma\in\{\mathsf{true},\mathsf{false}\}$。

**定义 7.2（小步规则）。** 除赋值外，规则完整列为
$$
\langle x:=a,\sigma\rangle\to
\langle\mathsf{skip},\sigma[x\mapsto\llbracket a\rrbracket\sigma]\rangle,
$$
$$
\frac{\langle c_1,\sigma\rangle\to\langle c_1',\sigma'\rangle}
{\langle c_1;c_2,\sigma\rangle\to\langle c_1';c_2,\sigma'\rangle},
\qquad
\langle\mathsf{skip};c_2,\sigma\rangle\to\langle c_2,\sigma\rangle,
$$
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{true}}
{\langle\mathsf{if}\ b\ \mathsf{then}\ c_1\ \mathsf{else}\ c_2,\sigma\rangle
\to\langle c_1,\sigma\rangle},
$$
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{false}}
{\langle\mathsf{if}\ b\ \mathsf{then}\ c_1\ \mathsf{else}\ c_2,\sigma\rangle
\to\langle c_2,\sigma\rangle},
$$
$$
\langle\mathsf{while}\ b\ \mathsf{do}\ c,\sigma\rangle
\to
\langle\mathsf{if}\ b\ \mathsf{then}\ (c;\mathsf{while}\ b\ \mathsf{do}\ c)
\ \mathsf{else}\ \mathsf{skip},\sigma\rangle.
$$

**定义 7.3（大步规则）。** 判断
$\langle c,\sigma\rangle\Downarrow\sigma'$ 由以下规则生成：
$$
\langle\mathsf{skip},\sigma\rangle\Downarrow\sigma,\qquad
\langle x:=a,\sigma\rangle\Downarrow
\sigma[x\mapsto\llbracket a\rrbracket\sigma],
$$
$$
\frac{\langle c_1,\sigma\rangle\Downarrow\sigma_1\quad
\langle c_2,\sigma_1\rangle\Downarrow\sigma_2}
{\langle c_1;c_2,\sigma\rangle\Downarrow\sigma_2}.
$$
条件真/假规则分别选择对应分支。循环规则为
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{false}}
{\langle\mathsf{while}\ b\ \mathsf{do}\ c,\sigma\rangle\Downarrow\sigma},
$$
$$
\frac{\llbracket b\rrbracket\sigma=\mathsf{true}\quad
\langle c,\sigma\rangle\Downarrow\sigma_1\quad
\langle\mathsf{while}\ b\ \mathsf{do}\ c,\sigma_1\rangle\Downarrow\sigma_2}
{\langle\mathsf{while}\ b\ \mathsf{do}\ c,\sigma\rangle\Downarrow\sigma_2}.
$$

**引理 7.4（顺序提升）。** 若
$\langle c,\sigma\rangle\to^*\langle\mathsf{skip},\tau\rangle$，则
$$
\langle c;d,\sigma\rangle\to^*\langle d,\tau\rangle.
$$

**证明。** 对给定多步长度归纳。零步时 $c=\mathsf{skip}$、$\sigma=\tau$，用一次
skip-sequence 规则。正长度时首步
$\langle c,\sigma\rangle\to\langle c',\sigma'\rangle$ 由 sequence 上下文规则提升，再对余下多步使用归纳假设。证毕。

**引理 7.5（一步向后封闭大步）。** 若
$\langle c,\sigma\rangle\to\langle c',\sigma'\rangle$ 且
$\langle c',\sigma'\rangle\Downarrow\tau$，则
$\langle c,\sigma\rangle\Downarrow\tau$。

**证明。** 对小步推导归纳，逐条处理定义 7.2。

- 赋值：后继是 skip 与更新状态；反演其大步推导得 $\tau$ 等于更新状态，再用赋值大步规则。
- sequence 上下文：后继为 $c_1';c_2$。反演大步 sequence，得
  $\langle c_1',\sigma'\rangle\Downarrow\rho$ 与
  $\langle c_2,\rho\rangle\Downarrow\tau$。归纳假设给出
  $\langle c_1,\sigma\rangle\Downarrow\rho$，再用大步 sequence。
- skip-sequence：后继为 $c_2$，在其大步推导前添加 skip 的大步推导。
- 条件真、假：后继分支的大步推导与相应布尔前提正好组成对应大步条件规则。
- while 展开：反演展开后 if 的大步推导。若布尔值为假，所选分支是 skip，故
  $\tau=\sigma$，用 while-false。若为真，所选分支是
  $c;\mathsf{while}\ b\ \mathsf{do}\ c$；反演 sequence 得循环体与余下循环的两个大步前提，用 while-true。

六种小步规则模式均已覆盖。证毕。

**定理 T7.1（终止时大小步等价）。**
$$
\langle c,\sigma\rangle\Downarrow\tau
\quad\Longleftrightarrow\quad
\langle c,\sigma\rangle\to^*\langle\mathsf{skip},\tau\rangle.
$$

**证明。** 正向对大步推导归纳。skip 为零步，赋值为一步。sequence 情形先用归纳假设得到
$c_1$ 的多步，再用引理 7.4 把它提升到 $c_1;c_2$，随后拼接 $c_2$ 的归纳所得多步。
条件情形先走选择分支的一步，再拼接该分支的归纳多步。
while-false 依次走 while 展开、if-false 两步到 skip。while-true 依次走展开和 if-true，
再用引理 7.4 提升循环体的归纳多步，最后拼接余下循环的归纳多步。

反向对多步长度归纳。零步迫使 $c=\mathsf{skip}$ 且 $\sigma=\tau$，用 skip 大步规则。
正长度写成
$\langle c,\sigma\rangle\to\langle c_1,\sigma_1\rangle
\to^*\langle\mathsf{skip},\tau\rangle$。
归纳假设给出 $\langle c_1,\sigma_1\rangle\Downarrow\tau$，引理 7.5 再给出
$\langle c,\sigma\rangle\Downarrow\tau$。证毕。

**例 7.6（两次赋值的完整配置轨迹）。** 设 $\sigma_0(x)=0$，则
$$
\begin{aligned}
\langle x:=x+1;x:=x+1,\sigma_0\rangle
&\to\langle\mathsf{skip};x:=x+1,\sigma_1\rangle\\
&\to\langle x:=x+1,\sigma_1\rangle\\
&\to\langle\mathsf{skip},\sigma_2\rangle,
\end{aligned}
$$
其中 $\sigma_1(x)=1,\sigma_2(x)=2$。T7.1 因而给出相应大步判断。

## 7.2 求值上下文与确定性

**定义 7.7（唯一分解）。** 给定表达式语言的值 $v$、redex $r$ 和单孔求值上下文
$E[-]$。若每个非值表达式恰有一对 $(E,r)$ 使 $e=E[r]$，称其满足唯一分解。
若每个 redex 至多收缩到一个结果，写 $r\rightsquigarrow r'$。

**定理 T7.2（唯一分解蕴含小步确定）。** 定义
$E[r]\to E[r']$ 当且仅当 $r\rightsquigarrow r'$。若分解唯一且 redex 收缩是偏函数，则
$e\to e_1$ 与 $e\to e_2$ 推出 $e_1=e_2$。

**证明。** 两个步骤分别给出分解 $e=E_1[r_1]=E_2[r_2]$。唯一分解给出
$E_1=E_2$ 且 $r_1=r_2$；redex 收缩唯一给出两个 $r'$ 相等，代回同一上下文即得
$e_1=e_2$。证毕。

## 7.3 CEK 机与卸载模拟

本节对象语言是第 3 章无类型 λ 项配上第 4 章从左到右 CBV 规则（去掉类型标注）。

**定义 7.8（闭包、续延与状态）。** 值闭包为
$w=(\lambda x.e,\rho)$，其中环境 $\rho$ 把变量映到值闭包。续延为
$$
K::=\mathsf{mt}\mid\mathsf{arg}(e,\rho,K)\mid\mathsf{fun}(w,K).
$$
求值状态写 $\langle e,\rho,K\rangle_E$，返回状态写 $\langle K,w\rangle_R$。

**定义 7.9（CEK 转移）。**
$$
\begin{aligned}
\langle x,\rho,K\rangle_E&\leadsto\langle K,\rho(x)\rangle_R,\\
\langle\lambda x.e,\rho,K\rangle_E&\leadsto
  \langle K,(\lambda x.e,\rho)\rangle_R,\\
\langle e_1e_2,\rho,K\rangle_E&\leadsto
  \langle e_1,\rho,\mathsf{arg}(e_2,\rho,K)\rangle_E,\\
\langle\mathsf{arg}(e_2,\rho,K),w_1\rangle_R&\leadsto
  \langle e_2,\rho,\mathsf{fun}(w_1,K)\rangle_E,\\
\langle\mathsf{fun}((\lambda x.e,\rho_0),K),w_2\rangle_R&\leadsto
  \langle e,\rho_0[x\mapsto w_2],K\rangle_E.
\end{aligned}
$$
第一条只用于 $x\in\mathrm{dom}(\rho)$。闭初态的可达状态都满足该条件。

**定义 7.10（卸载函数）。** 令 $\mathsf{close}(e,\rho)$ 为把
$e$ 的自由变量同时替换为环境中闭包递归展开所得闭值；进入抽象时删除同名环境项。
可达环境只指向更早创建的闭包，故该递归良定义。定义上下文
$$
\begin{aligned}
\mathcal E_{\mathsf{mt}}[-]&=[-],\\
\mathcal E_{\mathsf{arg}(e,\rho,K)}[-]
&=\mathcal E_K[[-]\ \mathsf{close}(e,\rho)],\\
\mathcal E_{\mathsf{fun}(w,K)}[-]
&=\mathcal E_K[\mathsf{close}(w)\ [-]].
\end{aligned}
$$
于是
$$
\mathsf{unload}(\langle e,\rho,K\rangle_E)
=\mathcal E_K[\mathsf{close}(e,\rho)],
\quad
\mathsf{unload}(\langle K,w\rangle_R)
=\mathcal E_K[\mathsf{close}(w)].
$$

**引理 7.11（每个机器步卸载为零或一个对象步）。** 若
$S\leadsto S'$，则
$$
\mathsf{unload}(S)=_\alpha\mathsf{unload}(S')
\quad\text{或}\quad
\mathsf{unload}(S)\to\mathsf{unload}(S').
$$
后一种情形恰是定义 7.9 的最后一条转移。

**证明。** 逐条代入五条转移。变量查找由
$\mathsf{close}(x,\rho)=\mathsf{close}(\rho(x))$ 给出相等；抽象返回相等；应用转移把
$e_1e_2$ 分解成 arg frame，卸载定义两边相等；arg 返回把函数闭值放入 fun frame，两边仍相等。
最后一条转移的左侧卸载为
$$
\mathcal E_K[(\lambda x.\mathsf{close}(e,\rho_0\setminus x))\,
\mathsf{close}(w_2)],
$$
它一步 βv 到
$\mathcal E_K[\mathsf{close}(e,\rho_0[x\mapsto w_2])]$，即右侧卸载。证毕。

**引理 7.12（一个对象步可由有限机器步实现）。** 若可达状态 $S$ 的卸载项满足
$\mathsf{unload}(S)\to e'$，则存在 $S'$ 使
$S\leadsto^+S'$ 且 $\mathsf{unload}(S')=_\alpha e'$。

**证明。** CBV 一步有唯一分解
$\mathsf{unload}(S)=E[(\lambda x.e_0)v]$。对 $E$ 的结构归纳。
孔上下文时，机器依次把应用拆成 arg frame、把函数抽象返回、进入实参、把实参返回到 fun frame，
随后执行最后一条转移；引理 7.11 的计算情形给出目标 βv 结果。
若 $E=E_0[-]\,e_2$，应用转移增加 arg frame，归纳假设在该 frame 下实现 $E_0$ 中的步骤；
若 $E=v_1E_0[-]$，机器已把 $v_1$ 保存为 fun frame，归纳假设在该 frame 下实现实参步骤。
这三种上下文构造穷尽左到右 CBV 求值上下文。变量查找与抽象返回只插入有限个卸载不变步，因为项和当前上下文均为有限语法树。证毕。

**定理 T7.3（CEK 与 CBV 求值等价）。** 对闭项 $e$ 与闭值 $v$，
$$
e\to^*v
\quad\Longleftrightarrow\quad
\exists w.\

\langle e,\varnothing,\mathsf{mt}\rangle_E
\leadsto^*\langle\mathsf{mt},w\rangle_R
\ \land\ \mathsf{close}(w)=_\alpha v.
$$

**证明。** 正向对对象多步长度归纳。每个对象一步由引理 7.12 实现；到达值后，定义 7.9 的抽象返回一步到最终返回状态。反向对机器步数归纳，逐步使用引理 7.11；卸载相等步不改变对象项，计算步贡献一个 βv 步。初态卸载为 $e$，最终状态卸载为
$\mathsf{close}(w)$，拼接即得。证毕。

**例 7.13（CEK 的完整状态轨迹）。** 对
$e=(\lambda x.x)(\lambda y.y)$，令
$w_x=(\lambda x.x,\varnothing)$、$w_y=(\lambda y.y,\varnothing)$：
$$
\begin{aligned}
\langle e,\varnothing,\mathsf{mt}\rangle_E
&\leadsto\langle\lambda x.x,\varnothing,
  \mathsf{arg}(\lambda y.y,\varnothing,\mathsf{mt})\rangle_E\\
&\leadsto\langle\mathsf{arg}(\lambda y.y,\varnothing,\mathsf{mt}),w_x\rangle_R\\
&\leadsto\langle\lambda y.y,\varnothing,\mathsf{fun}(w_x,\mathsf{mt})\rangle_E\\
&\leadsto\langle\mathsf{fun}(w_x,\mathsf{mt}),w_y\rangle_R\\
&\leadsto\langle x,[x\mapsto w_y],\mathsf{mt}\rangle_E\\
&\leadsto\langle\mathsf{mt},w_y\rangle_R.
\end{aligned}
$$
前四步卸载不变，第五步对应唯一 βv 步，最后查找环境。

**外部输入 EI-7（通用 SOS 格式结果）。** 对带标签转移系统的特定 GSOS/ntyft 等规则格式，
可推出双模拟是同余等一般元定理。其精确格式与结论登记于 SOURCES.md；本章 T7.1--T7.3
均由具体规则直接证明，不调用 EI-7。

## 7.4 操作语义证明的责任边界

while 大小步等价、唯一分解下的确定性与本节 CEK 双向对应均在书内完整证明。
EI-7 只说明向并发或带标签语言扩张时可用的格式理论，不承担本章任何有限规则检查。

## 练习

**练习 E7.1.** 重建例 7.6，并写出对应的大步推导树。

**练习 E7.2.** 由定义 7.2 直接证明 while 语言小步确定。

**练习 E7.3.** 对 $(\lambda x.\lambda z.x)(\lambda y.y)$ 写出完整 CEK 轨迹及每步卸载项。

**练习 E7.4.** 在引理 7.5 中完整写出 while 展开后布尔值为真的反演子树。
