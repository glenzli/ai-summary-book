# 第 9 章：Hoare 逻辑、循环不变式与相对完备性

指称语义回答命令表示哪个偏状态变换，程序逻辑则组织关于所有初态与终态的证明。三元组还存在两个容易混淆的层级：语义有效性量化实际运行，语法可证性由有限规则生成；soundness 只从后者走向前者。由于本章采用部分正确性，发散运行不制造反例，相应的语义谓词是 weakest liberal precondition。本章完整证明基本 Hoare 系统 sound，并把 Cook 相对完备性保留为精确 EI-9。

## 9.1 语义三元组与证明规则

**定义 9.1（断言与替换）。** 断言是状态集合 $S$ 上的谓词。写
$\sigma\models P$。定义语义断言替换
$$
\sigma\models Q[x:=a]
\quad\Longleftrightarrow\quad
\sigma[x\mapsto\llbracket a\rrbracket\sigma]\models Q.
$$
断言蕴含 $P\Rightarrow Q$ 表示每个满足 $P$ 的状态都满足 $Q$。

**定义 9.2（部分正确性）。**
$$
\models\{P\}\,c\,\{Q\}
$$
表示对所有 $\sigma,\tau$，
$$
\sigma\models P\ \land\
\langle c,\sigma\rangle\Downarrow\tau
\quad\Longrightarrow\quad
\tau\models Q.
$$
若 $c$ 从 $\sigma$ 发散，定义不要求后置条件。

**定义 9.3（基本 Hoare 系统）。** 可证判断
$\vdash_H\{P\}c\{Q\}$ 由下列规则生成：
$$
\frac{}{\{P\}\mathsf{skip}\{P\}}\;\textsc{H-Skip},
\qquad
\frac{}{\{Q[x:=a]\}\ x:=a\ \{Q\}}\;\textsc{H-Assign},
$$
$$
\frac{\{P\}c_1\{R\}\quad\{R\}c_2\{Q\}}
{\{P\}c_1;c_2\{Q\}}\;\textsc{H-Seq},
$$
$$
\frac{\{P\land b\}c_1\{Q\}\quad
\{P\land\neg b\}c_2\{Q\}}
{\{P\}\mathsf{if}\ b\ \mathsf{then}\ c_1\ \mathsf{else}\ c_2\{Q\}}
\;\textsc{H-If},
$$
$$
\frac{\{I\land b\}c\{I\}}
{\{I\}\mathsf{while}\ b\ \mathsf{do}\ c\{I\land\neg b\}}
\;\textsc{H-While},
$$
$$
\frac{P\Rightarrow P'\quad\{P'\}c\{Q'\}\quad Q'\Rightarrow Q}
{\{P\}c\{Q\}}\;\textsc{H-Consequence}.
$$
规则中的未标 $\vdash_H$ 三元组都是证明系统前提；蕴含则在选定断言语义中有效。

**引理 9.4（语义不变式引理）。** 若
$\models\{I\land b\}c\{I\}$，则
$$
\models\{I\}\mathsf{while}\ b\ \mathsf{do}\ c\{I\land\neg b\}.
$$

**证明。** 固定 $\sigma\models I$ 以及终止推导
$\langle\mathsf{while}\ b\ \mathsf{do}\ c,\sigma\rangle\Downarrow\tau$，
对该推导归纳。若末规则是 while-false，则 $\tau=\sigma$ 且
$\llbracket b\rrbracket\sigma=\mathsf{false}$，所以
$\tau\models I\land\neg b$。若末规则是 while-true，则
$\sigma\models I\land b$，且存在 $\rho$ 使
$\langle c,\sigma\rangle\Downarrow\rho$ 和余下循环从 $\rho$ 到 $\tau$。
前提三元组给出 $\rho\models I$；对余下循环推导使用归纳假设，得
$\tau\models I\land\neg b$。两种 while 大步末规则均已覆盖。证毕。

**定理 T9.1（Hoare 规则 soundness）。** 若
$\vdash_H\{P\}c\{Q\}$，则 $\models\{P\}c\{Q\}$。

**证明。** 对 Hoare 推导归纳，逐条处理定义 9.3。

- \textsc{H-Skip}：大步终态等于初态，所以 $P$ 保持。
- \textsc{H-Assign}：若初态满足 $Q[x:=a]$，定义 9.1 说明更新后的状态满足
  $Q$；赋值大步规则的终态正是该更新。
- \textsc{H-Seq}：反演 sequence 大步推导，得到中间状态 $\rho$。第一个归纳假设从
  $P$ 得 $\rho\models R$，第二个归纳假设再得终态满足 $Q$。
- \textsc{H-If}：布尔值唯一。真时初态满足 $P\land b$，反演大步规则得到
  $c_1$ 的运行，使用第一个归纳假设；假时对 $c_2$ 使用第二个归纳假设。
- \textsc{H-While}：归纳假设先把前提三元组变为语义有效；引理 9.4 给出结论。
- \textsc{H-Consequence}：若初态满足 $P$，由 $P\Rightarrow P'$ 得
  $P'$；中间归纳假设给出终态 $Q'$，再由 $Q'\Rightarrow Q$ 得 $Q$。

六种 Hoare 末规则均已覆盖。证毕。

## 9.2 一个完整循环证明

**例 9.5（递减循环的 Hoare 推导）。** 在整数状态上验证
$$
\{x\ge0\}\

\mathsf{while}\ x>0\ \mathsf{do}\ x:=x-1\
\{x=0\}.
$$
取 $I\equiv x\ge0$。赋值公理给出
$$
\{x-1\ge0\}\ x:=x-1\ \{x\ge0\}.
$$
算术蕴含
$x\ge0\land x>0\Rightarrow x-1\ge0$，故 \textsc{H-Consequence} 得
$$
\{x\ge0\land x>0\}\ x:=x-1\ \{x\ge0\}.
$$
\textsc{H-While} 随后给出
$$
\{x\ge0\}\ W\ \{x\ge0\land\neg(x>0)\}.
$$
最后用整数序上的蕴含
$x\ge0\land x\le0\Rightarrow x=0$ 和 \textsc{H-Consequence} 得目标。
这条推导只证明部分正确性。总正确性还需在真分支中证明自然数变元 $x$ 严格下降。

## 9.3 WLP 与相对完备性

**定义 9.6（weakest liberal precondition）。**
$$
\sigma\models\mathrm{wlp}(c,Q)
\quad\Longleftrightarrow\quad
\forall\tau.\

\langle c,\sigma\rangle\Downarrow\tau\Rightarrow\tau\models Q.
$$
“liberal”表示不要求从 $\sigma$ 终止；总正确性的 weakest precondition 还必须合取终止性。

**命题 9.7（WLP 的语义特征）。**
$$
\models\{P\}c\{Q\}
\quad\Longleftrightarrow\quad
P\Rightarrow\mathrm{wlp}(c,Q).
$$

**证明。** 正向：若 $\sigma\models P$，对任意
$\langle c,\sigma\rangle\Downarrow\tau$，三元组有效性给出
$\tau\models Q$，所以 $\sigma\models\mathrm{wlp}(c,Q)$。
反向：若 $\sigma\models P$ 且命令终止到 $\tau$，蕴含前提给出
$\sigma\models\mathrm{wlp}(c,Q)$；展开定义便有 $\tau\models Q$。证毕。

**例 9.8（发散揭示 WLP 与 WP 的差别）。** 令
$L=\mathsf{while}\ \mathsf{true}\ \mathsf{do}\ \mathsf{skip}$。不存在
$\langle L,\sigma\rangle\Downarrow\tau$，故对任意 $Q$ 和 $\sigma$，
$\sigma\models\mathrm{wlp}(L,Q)$；于是
$\mathrm{wlp}(L,Q)=\mathsf{true}$。总正确性的
$\mathrm{wp}(L,Q)$ 则为 $\mathsf{false}$，因为没有初态使 $L$ 终止。

**外部输入 EI-9（Cook 相对完备性）。** 对确定 while/ALGOL 型语言的部分正确性，
若断言语言能表达相关最弱自由前置条件（或等价的运行关系），并允许把解释结构中所有真断言蕴含作为背景理论使用，
则每个语义有效 Hoare 三元组均可在相应 Hoare 系统中证明。这里的“相对”以断言理论为 oracle，
不提供一阶算术真理或循环不变式的总判定器。

**证明路线（不计作书内证明）。** 来源对程序结构递归构造可表达的最弱前置断言，
在 while 情形把可达状态关系编码成不变式，再由背景理论中的真蕴含完成 consequence 步。
过程调用等扩张需要来源中的额外规则；SOURCES.md 精确登记版本与定理位置。

## 9.4 程序逻辑的证明边界

T9.1、引理 9.4 与命题 9.7 在本章完整证明。EI-9 只承担从语义有效性回到语法可证性的相对完备方向；
它不参与 soundness，也不意味着自动生成证明或判定程序正确性。

## 练习

**练习 E9.1.** 用赋值与 consequence 规则完整证明
$\{x=0\}\ x:=x+1\ \{x=1\}$。

**练习 E9.2.** 对
$z:=x;\mathsf{while}\ y>0\ \mathsf{do}\ (z:=z+1;y:=y-1)$
给出含初始值逻辑常量的循环不变式并完成部分正确性推导。

**练习 E9.3.** 证明例 9.8 的 WLP 等式，并说明总正确性为何不同。

**练习 E9.4.** 在 T9.1 的 while 情形中写出外层 Hoare 推导归纳与内层运行推导归纳各自的归纳对象。
