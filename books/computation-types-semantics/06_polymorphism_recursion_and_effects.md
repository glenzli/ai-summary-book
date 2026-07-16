# 第 6 章：System F、递归类型与效应接口

简单类型为每个函数固定一个输入类型；System F 则允许项对类型本身抽象，由此表达表示无关的多态程序。递归类型和效应又是两种不同扩张：前者解类型方程，后者改变计算结果的组织方式。若把三者混成一个语言，纯 System F 的正规化与参数性结论会被错误外推。本章先完整定义纯 CBV System F 并证明 preservation，再分别以可执行的 fold/unfold 和异常 monad 计算说明两个接口。

## 6.1 纯 CBV System F

**定义 6.1（类型与良构）。** 类型变量上下文 $\Delta$ 是互异类型变量的有限序列。类型为
$$
A::=\alpha\mid A\to B\mid\forall\alpha.A.
$$
良构判断由
$$
\frac{\alpha\in\Delta}{\Delta\vdash\alpha\ \mathsf{type}},\qquad
\frac{\Delta\vdash A\ \mathsf{type}\quad\Delta\vdash B\ \mathsf{type}}
{\Delta\vdash A\to B\ \mathsf{type}},
$$
$$
\frac{\Delta,\alpha\vdash A\ \mathsf{type}}
{\Delta\vdash\forall\alpha.A\ \mathsf{type}}
$$
生成。类型替换 $A[\alpha:=B]$ 捕获避免地定义；进入 $\forall\beta$ 时先令 $\beta$ 避开
$\mathrm{FTV}(B)$。

**定义 6.2（项、上下文与类型规则）。**
$$
e::=x\mid\lambda x:A.e\mid e\,e\mid\Lambda\alpha.e\mid e[A].
$$
值为 $\lambda x:A.e$ 或 $\Lambda\alpha.e$。项上下文 $\Gamma$ 是变量到
$\Delta$-良构类型的有限映射。判断 $\Delta;\Gamma\vdash e:A$ 由 STLC 三条规则（每个类型都要求在
$\Delta$ 下良构）及
$$
\frac{\Delta,\alpha;\Gamma\vdash e:A\qquad\alpha\notin\mathrm{FTV}(\Gamma)}
{\Delta;\Gamma\vdash\Lambda\alpha.e:\forall\alpha.A}\;\textsc{T-TAbs},
$$
$$
\frac{\Delta;\Gamma\vdash e:\forall\alpha.A\qquad\Delta\vdash B\ \mathsf{type}}
{\Delta;\Gamma\vdash e[B]:A[\alpha:=B]}\;\textsc{T-TApp}
$$
生成。所有绑定项按 α-等价识别。

**定义 6.3（CBV 求值）。** 第 4 章三条应用规则扩展到上述两类值，并增加
$$
\frac{e\to e'}{e[B]\to e'[B]}\;\textsc{E-TApp},
\qquad
(\Lambda\alpha.e)[B]\to e[\alpha:=B]\;\textsc{E-TBeta}.
$$
类型参数本身不求值。

**引理 6.3A（System F weakening）。**

1. 若 $\Delta;\Gamma\vdash e:A$，$\Gamma\subseteq\widehat\Gamma$，且
   $\widehat\Gamma$ 的每个声明都在 $\Delta$ 下良构，则
   $\Delta;\widehat\Gamma\vdash e:A$。
2. 若 $\Delta;\Gamma\vdash e:A$ 且 $\beta\notin\Delta$，则在对项内绑定类型变量作必要的
   α-改名后，$\Delta,\beta;\Gamma\vdash e:A$。

**证明。** 两项分别对给定类型推导归纳。变量、项应用和类型应用情形直接在扩张上下文中重用末
规则；项抽象情形先把绑定项变量改名到扩张上下文之外，再对主体使用归纳假设。

对类型抽象末规则，设待提升项为 $\Lambda\alpha.e_0$。先把 $\alpha$ 改名，使其不在
$\mathrm{FTV}(\widehat\Gamma)$ 中，并在第二项中还使
$\alpha\ne\beta$。第一项的归纳假设把
$\Delta,\alpha;\Gamma\vdash e_0:C$ 提升为
$\Delta,\alpha;\widehat\Gamma\vdash e_0:C$；第二项的归纳假设把它提升为
$\Delta,\beta,\alpha;\Gamma\vdash e_0:C$。两种情形的类型抽象侧条件均由新鲜性成立，重用
\textsc{T-TAbs} 即得结论。五种末规则均已覆盖。证毕。

**引理 6.4（类型替换）。** 若
$\Delta,\alpha,\Delta';\Gamma\vdash e:A$ 且
$\Delta\vdash B\ \mathsf{type}$，则
$$
\Delta,\Delta';\Gamma[\alpha:=B]
\vdash e[\alpha:=B]:A[\alpha:=B].
$$
这里 $\Delta'$ 只是互异类型变量的序列，不含可被替换的类型声明；结论上下文中删除 $\alpha$，其余变量保持原序列。

**证明。** 先同时证明类型良构替换：若
$\Delta,\alpha,\Delta'\vdash C\ \mathsf{type}$，则
$\Delta,\Delta'\vdash C[\alpha:=B]\ \mathsf{type}$。对良构推导归纳：变量末规则中，
被替换变量 $\alpha$ 使用前提 $\Delta\vdash B\ \mathsf{type}$；其余变量仍在替换后的类型上下文中；
箭头情形对两个前提使用归纳假设；全称情形先把绑定变量改名，使其不同于 $\alpha$ 且不自由出现于
$B$，再对主体使用归纳假设并重建全称类型。于是三种类型良构末规则全部成立。

再对项类型推导归纳。

- 变量规则：变量声明 $x:C$ 替换为 $x:C[\alpha:=B]$，仍由变量规则得到。
- 项抽象：前提中的参数类型、主体和结果类型同时替换；先把项绑定变量取新鲜，再对主体用归纳假设并重用抽象规则。
- 项应用：对函数和实参两个前提使用归纳假设，再用应用规则。
- \textsc{T-TAbs}：设其绑定类型变量为 $\beta$。先 α-改名使
  $\beta\ne\alpha$ 且 $\beta\notin\mathrm{FTV}(B)$。归纳假设作用于主体；原侧条件
  $\beta\notin\mathrm{FTV}(\Gamma)$ 替换后仍成立，故重用 \textsc{T-TAbs}。
- \textsc{T-TApp}：归纳假设给出函数项的替换类型；由刚证明的类型良构替换可得
  $\Delta,\Delta'\vdash C[\alpha:=B]\ \mathsf{type}$。重用
  \textsc{T-TApp}，并使用捕获避免替换结合律
  $A[\beta:=C][\alpha:=B]=_\alpha
  A[\alpha:=B][\beta:=C[\alpha:=B]]$。

五种项类型末规则均已覆盖。证毕。

**引理 6.5（项替换）。** 若
$\Delta;\Gamma,x:A\vdash e:C$ 且 $\Delta;\Gamma\vdash v:A$，则
$$
\Delta;\Gamma\vdash e[x:=v]:C.
$$

**证明。** 对第一项类型推导归纳。变量情形分变量等于 $x$ 与不等于 $x$；前者使用第二前提，
后者重用上下文声明。项应用对两个前提使用归纳假设。项抽象先 α-改名其绑定变量，使其不在
$\mathrm{FV}(v)$ 中；由引理 6.3A(1)把 $v$ 的推导扩到该变量上下文，再对主体使用归纳假设并重建
项抽象。类型抽象先把绑定类型变量 $\alpha$ 改名，使
$\alpha\notin\mathrm{FTV}(v)\cup\mathrm{FTV}(\Gamma)\cup\mathrm{FTV}(A)$；由引理 6.3A(2)得到
$\Delta,\alpha;\Gamma\vdash v:A$，对主体使用归纳假设，再以原侧条件重建类型抽象。类型应用对
主项使用归纳假设，类型实参不含项变量。五种末规则均已覆盖。证毕。

**引理 6.6（System F 反演）。**

1. 若 $\Delta;\Gamma\vdash e_1e_2:C$，则存在 $A$ 使
   $\Delta;\Gamma\vdash e_1:A\to C$ 且 $\Delta;\Gamma\vdash e_2:A$。
2. 若 $\Delta;\Gamma\vdash\lambda x:A.e:C$，则
   $C=A\to D$ 且 $\Delta;\Gamma,x:A\vdash e:D$，对某个 $D$。
3. 若 $\Delta;\Gamma\vdash e[B]:C$，则存在 $A,\alpha$ 使
   $\Delta;\Gamma\vdash e:\forall\alpha.A$ 且 $C=A[\alpha:=B]$。
4. 若 $\Delta;\Gamma\vdash\Lambda\alpha.e:C$，则
   $C=\forall\alpha.A$ 且 $\Delta,\alpha;\Gamma\vdash e:A$，对某个 $A$。

**证明。** 四种语法头分别只能由项应用、项抽象、\textsc{T-TApp}、
\textsc{T-TAbs} 作为末规则产生，读取规则前提即得。证毕。

**定理 T6.1（System F Preservation）。** 若
$\Delta;\Gamma\vdash e:A$ 且 $e\to e'$，则
$\Delta;\Gamma\vdash e':A$。

**证明。** 对求值推导归纳。

- 应用左、应用右两条上下文规则：用引理 6.6(1)反演，再对发生步骤的前提使用归纳假设，最后重建项应用。
- 项 β：反演整个应用，再用引理 6.6(2)反演函数抽象；箭头构造子的单射性使参数类型与结果类型对齐，随后应用引理 6.5。
- \textsc{E-TApp}：用引理 6.6(3)反演类型应用，对主项步骤使用归纳假设，再用 \textsc{T-TApp}。
- \textsc{E-TBeta}：反演整个类型应用，再用引理 6.6(4)反演类型抽象，得到
  $\Delta,\alpha;\Gamma\vdash e_0:C$ 和
  $\alpha\notin\mathrm{FTV}(\Gamma)$。引理 6.4 给出
  $\Delta;\Gamma\vdash e_0[\alpha:=B]:C[\alpha:=B]$。

定义 6.3 的五种规则模式均已覆盖。证毕。

**例 6.7（多态恒等的推导与两次 β）。**
$$
\mathsf{id}=\Lambda\alpha.\lambda x:\alpha.x
:\forall\alpha.\alpha\to\alpha.
$$
推导从 $\alpha;x:\alpha\vdash x:\alpha$ 开始，依次使用项抽象和
\textsc{T-TAbs}。对任意闭类型 $B$ 与值 $v:B$，
$$
\mathsf{id}[B]\,v
\to(\lambda x:B.x)\,v
\to v.
$$
第一步是 \textsc{E-TBeta}，第二步是项 β；T6.1 分别保持类型 $B\to B$ 与 $B$。

**外部输入 EI-6（纯 System F 关系参数性）。** 对定义 6.1--6.3 的纯 System F，
给每对类型解释指定关系并按箭头、全称类型递归提升。若
$\Delta;\Gamma\vdash e:A$，则 $e$ 的两个关系相关实例落在 $A$ 的关系解释中
（abstraction theorem）。特别地，闭项
$f:\forall\alpha.\alpha\to\alpha$ 与多态恒等函数扩张相等。

**证明路线（不计作书内证明）。** 证明同时归纳类型推导和关系环境，在
\textsc{T-TAbs}/\textsc{T-TApp} 情形量化所有类型及关系。非终止、异常、引用、类型反射或
seq 会改变定理的关系类别与结论；完整来源由 SOURCES.md 的 EI-6 定位承担。

## 6.2 同构递归类型

**定义 6.8（同构递归口径）。** 在一个另行声明含 $1,+$ 的类型语言中，
$\mu X.F(X)$ 带
$$
\mathsf{fold}:F(\mu X.F(X))\to\mu X.F(X),\qquad
\mathsf{unfold}:\mu X.F(X)\to F(\mu X.F(X))
$$
及计算规则
$\mathsf{unfold}(\mathsf{fold}\,v)\to v$。本节不采用把两种类型判断等同的等递归口径。

**例 6.9（递归自然数的一步展开）。** 令
$N_\mu=\mu X.(1+X)$，
$$
\mathsf{zero}=\mathsf{fold}(\mathsf{inl}\,*),\qquad
\mathsf{one}=\mathsf{fold}(\mathsf{inr}\,\mathsf{zero}).
$$
于是实际求值为
$$
\mathsf{unfold}\,\mathsf{one}
\to\mathsf{inr}\,\mathsf{zero}:1+N_\mu.
$$
递归类型本身不等于项级一般递归。若另加
$\mathsf{fix}_A:(A\to A)\to A$，则
$\mathsf{fix}_A(\lambda x:A.x)$ 沿第 4 章例 4.6 的轨迹无限求值，虽仍可保持类型。

## 6.3 Monad 与异常计算

**定义 6.10（Kleisli 三元组形式的 monad）。** 在集合范畴中，给每个集合 $A$ 一个集合
$TA$、函数 $\eta_A:A\to TA$，并给每个 $f:A\to TB$ 一个扩张
$f^*:TA\to TB$。要求
$$
(\eta_A)^*=\mathrm{id}_{TA},\qquad
f^*\circ\eta_A=f,\qquad
g^*\circ f^*=(g^*\circ f)^*.
$$
最后一式中 $f:A\to TB,g:B\to TC$。

**定理 T6.2（Kleisli 合成）。** 定义
$g\odot f=g^*\circ f:A\to TC$。则 $\eta$ 是单位，$\odot$ 结合。

**证明。** 左单位：
$f\odot\eta_A=f^*\circ\eta_A=f$。右单位：
$\eta_B\odot f=(\eta_B)^*\circ f=\mathrm{id}_{TB}\circ f=f$。结合律为
$$
h\odot(g\odot f)
=h^*\circ g^*\circ f
=(h^*\circ g)^*\circ f
=(h\odot g)\odot f,
$$
中间等式是第三条 monad 律。证毕。

**例 6.11（异常绑定的两条执行轨迹）。** 固定异常集合 $E$，令 $TA=A+E$，
$\eta_A(a)=\mathsf{inl}(a)$，并定义
$$
f^*(\mathsf{inl}(a))=f(a),\qquad
f^*(\mathsf{inr}(\epsilon))=\mathsf{inr}(\epsilon).
$$
若 $f(2)=\mathsf{inl}(3)$，则
$f^*(\mathsf{inl}(2))=\mathsf{inl}(3)$；而对任意 $f$，
$f^*(\mathsf{inr}(\epsilon))=\mathsf{inr}(\epsilon)$。第二条轨迹显示异常跳过后续函数。

## 6.4 多态结果与扩张边界

T6.1、T6.2 及所需替换引理均在本章完整证明。EI-6 只覆盖纯 System F 的关系参数性；
递归类型的域方程模型、带状态参数性和 step-indexed logical relations 均不进入本章证明链。

## 练习

**练习 E6.1.** 写出例 6.7 的完整类型推导树。

**练习 E6.2.** 对异常 monad 分正常值与异常值验证三条 monad 律。

**练习 E6.3.** 写出 $\mathsf{unfold}(\mathsf{fold}(\mathsf{inr}\,\mathsf{zero}))$ 的类型与一步轨迹。

**练习 E6.4.** 指出 T6.1 的类型 β 情形为何必须使用类型替换，而不能调用第 4 章的项替换。
