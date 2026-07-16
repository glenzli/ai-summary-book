# 符号与判断约定

本表固定全书符号。若某章临时改变符号，必须在该章局部声明，并不得反向污染全书。

## 集合、字和编码

| 符号 | 含义 |
| --- | --- |
| $\mathbb{N}$ | 包含 $0$ 的自然数集合 |
| $\Sigma$ | 有限字母表 |
| $\Sigma^\*$ | $\Sigma$ 上有限字串集合 |
| $\epsilon$ | 空字 |
| $\langle x_1,\ldots,x_n\rangle$ | 有效配对/编码；默认为可计算双射的一个固定选择 |
| $\ulcorner e\urcorner$ | 语法对象 $e$ 的 Gödel 编码 |
| $\varphi_e$ | 第 $e$ 个偏可计算函数 |
| $A \le_m B$ | 集合 $A$ many-one 归约到 $B$ |
| $A \le_T B$ | 集合 $A$ Turing 归约到 $B$ |

## 程序、状态与求值

| 符号 | 含义 |
| --- | --- |
| $e,e_1,e_2$ | 表达式或项，具体语法由章节给定 |
| $v$ | 值 |
| $\sigma$ | 程序状态，通常为变量到值的有限映射 |
| $\langle c,\sigma\rangle$ | 命令 $c$ 在状态 $\sigma$ 下的运行配置 |
| $e \to e'$ | 单步归约或小步求值 |
| $e \to^\* e'$ | $0$ 步或多步归约 |
| $e \Downarrow v$ | 大步求值到值 $v$ |
| $E[-]$ | 求值上下文 |
| $M \leadsto M'$ | 抽象机一步转移 |

## λ 演算与替换

| 符号 | 含义 |
| --- | --- |
| $x,y,z$ | 变量 |
| $\lambda x.e$ | 抽象 |
| $e_1\,e_2$ | 应用 |
| $\mathrm{FV}(e)$ | 自由变量集合 |
| $e[x:=s]$ | 捕获避免替换 |
| $e =_\alpha e'$ | α-等价 |
| $e \to_\beta e'$ | 一步 β 归约 |
| $\mathrm{nf}(e)$ | 正规形；存在时才使用 |

## 类型与上下文

| 符号 | 含义 |
| --- | --- |
| $A,B,C$ | 类型 |
| $\Gamma,\Delta$ | 类型上下文，有限变量声明序列 |
| $\Gamma \vdash e:A$ | 在上下文 $\Gamma$ 下项 $e$ 具有类型 $A$ |
| $\Gamma \vdash A\ \mathsf{type}$ | 在依赖类型上下文中 $A$ 是类型 |
| $\Gamma \vdash A\equiv B\ \mathsf{type}$ | $A,B$ 判断等价且都是类型 |
| $\Gamma \vdash a\equiv b:A$ | $a,b$ 在类型 $A$ 上判断等价 |
| $\Pi x:A.B$ | 依赖函数类型 |
| $\Sigma x:A.B$ | 依赖配对类型 |
| $\mathsf{Id}_A(a,b)$ | 恒等类型 |
| $\forall \alpha.e$ | System F 的类型抽象，章节中也写作 $\Lambda\alpha.e$ |
| $\mu X.F(X)$ | 递归类型或不动点类型；必须说明等递归或同构递归口径 |

## 语义域与逻辑

| 符号 | 含义 |
| --- | --- |
| $\llbracket e \rrbracket$ | 表达式或类型的指称语义 |
| $D,E$ | 偏序、cpo 或语义域 |
| $\bot$ | 最小元素，表示未定义或发散 |
| $\sqsubseteq$ | 信息序或近似序 |
| $\bigsqcup_n d_n$ | ω-链上确界 |
| $\mathrm{fix}(f)$ | 连续函数 $f$ 的最小不动点 |
| $\{P\}\ c\ \{Q\}$ | Hoare 三元组 |
| $\models \{P\}c\{Q\}$ | 三元组语义有效 |
| $\vdash_H \{P\}c\{Q\}$ | 在 Hoare 系统中可证明 |
| $\mathrm{wlp}(c,Q)$ | 命令 $c$ 关于后置条件 $Q$ 的最弱自由前置条件（部分正确性） |

## 状态标签

| 标签 | 含义 |
| --- | --- |
| `内部证明` | 正文给出完整、可逐项核对的证明 |
| `外部输入` | 固定演算、假设和结论并登记来源的大型结果 |
| `边界说明` | 限制或反例，不作为证明链前提 |
