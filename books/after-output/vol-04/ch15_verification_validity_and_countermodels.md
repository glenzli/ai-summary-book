# 第十五章 有效性、反模型与逻辑形式

考虑一段熟悉的诊断：

> 如果服务器过载，延迟就会上升。现在延迟上升了，所以服务器已经过载。

两句前提和结论都可能为真，论证仍可能失败。缓存失效、网络拥塞或下游服务变慢都能使延迟上升。要精确指出
失败之处，我们不必先判断服务器实际发生了什么；只需构造一种让前提全真而结论假的情形。这个情形就是
反模型。

本章会把这段自然语言先翻译为公式，再用赋值检验其结构，最后说明加入什么前提才可能完成修复。反模型的价值
不在于“唱反调”，而在于它能把论证失败定位到一个具体赋值。

## V2.1 语法层

固定可数原子命题集 $\mathsf{At}=\{p_0,p_1,\ldots\}$。命题公式集合 $\mathsf{Form}$ 是满足下列生成规则的最小集合：

1. 若 $p\in\mathsf{At}$，则 $p\in\mathsf{Form}$；
2. 若 $A,B\in\mathsf{Form}$，则 $\neg A,(A\land B),(A\lor B),(A\to B)\in\mathsf{Form}$。

“最小”排除由规则不能有限生成的字符串。括号与优先级属于具体语法；后文只对合式公式定义真值。

## V2.2 语义层

**定义 V2.1（赋值与满足）.** 赋值是函数

$$
v:\mathsf{At}\to\{0,1\}.
$$

它唯一递归延拓到所有公式：

$$
\begin{aligned}
v\models p &\Longleftrightarrow v(p)=1,\\
v\models\neg A &\Longleftrightarrow v\not\models A,\\
v\models A\land B &\Longleftrightarrow v\models A\text{ 且 }v\models B,\\
v\models A\lor B &\Longleftrightarrow v\models A\text{ 或 }v\models B,\\
v\models A\to B &\Longleftrightarrow v\not\models A\text{ 或 }v\models B.
\end{aligned}
$$

这里的“或”是相容析取：两者同时真时 $A\lor B$ 仍真。

**定义 V2.2（可满足、有效与语义蕴涵）.** 公式集 $\Gamma$ 可满足，是指存在 $v$ 使 $v\models A$ 对所有 $A\in\Gamma$ 成立。公式 $A$ 有效，记 $\models A$，是指每个赋值都满足 $A$。写 $\Gamma\models A$，是指每个满足 $\Gamma$ 的赋值都满足 $A$。

若 $\Gamma$ 不可满足，则 $\Gamma\models A$ 对每个 $A$ 都成立。这是语义定义的真空情形，不说明任意现实前提都可接受。

## V2.3 反模型判据

**定义 V2.3（反模型）.** 若 $v\models\Gamma$ 且 $v\not\models A$，则 $v$ 称为 $\Gamma\models A$ 的反模型。

**定理 V2.4（反模型判据）.** 对任意 $\Gamma\subseteq\mathsf{Form}$ 与 $A\in\mathsf{Form}$，

$$
\Gamma\not\models A
\quad\Longleftrightarrow\quad
\text{存在 }\Gamma\models A\text{ 的反模型}.
$$

**证明.** 由定义，$\Gamma\models A$ 是全称命题

$$
\forall v\,(v\models\Gamma\Rightarrow v\models A).
$$

在经典逻辑中否定该式，得到

$$
\exists v\,(v\models\Gamma\land v\not\models A),
$$

这恰是反模型的定义。两方向由同一等价给出。证毕。

## V2.4 基本有效式

**定理 V2.5（modus ponens 的语义有效性）.**

$$
A,A\to B\models B.
$$

**证明.** 任取满足两个前提的赋值 $v$。若 $v\not\models B$，则因 $v\models A$，蕴含的语义给出 $v\not\models A\to B$，与前提矛盾。因此 $v\models B$。赋值任意，故结论成立。证毕。

**定理 V2.6（经典逆否等价）.** 对每个赋值 $v$，

$$
v\models A\to B
\quad\Longleftrightarrow\quad
v\models\neg B\to\neg A.
$$

**证明.** 左式为假当且仅当 $v\models A$ 且 $v\not\models B$。右式为假当且仅当 $v\models\neg B$ 且 $v\not\models\neg A$；按二值否定语义，这同样等价于 $v\not\models B$ 且 $v\models A$。二式恰在同一组赋值下为假，故逐赋值同真值。证毕。

逆否命题不是逆命题。一般没有 $A\to B\models B\to A$。

## V2.5 两个无效形式

肯定后件

$$
A\to B,\ B\therefore A
$$

无效：取 $v(A)=0,v(B)=1$，前提全真而结论假。否定前件

$$
A\to B,\ \neg A\therefore\neg B
$$

也由同一赋值反驳。

自然语言中的对应错误是把某机制的一个后果当成该机制的唯一来源，或把一个充分条件的失败当成结果必然失败。
在开头的例子中，令 $A$ 表示“服务器过载”，$B$ 表示“延迟上升”；赋值 $v(A)=0,v(B)=1$ 就描述了由其他原因
造成延迟的情形。若监控系统另能建立 $B\to A$，例如在已经排除所有其他来源的受控诊断模型中，结论便可由
modus ponens 得到。这个新增前提本身需要系统证据，不能靠把“所以”写得更肯定来获得。

## V2.6 析取规则与自然语言翻译

**命题 V2.7（析取三段论有效）.**

$$
A\lor B,\ \neg A\models B.
$$

**证明.** 任取满足前提的赋值。由 $\neg A$ 得 $A$ 假；而 $A\lor B$ 真。按析取语义，至少一个析取支为真，故只能有 $B$ 真。证毕。

因此，一旦自然语言“或”已经正确翻译为形式公式 $A\lor B$，穷尽性就在该公式的真值条件中，不需要额外前提。真正的风险发生在翻译层：说话者列出的 $A,B$ 可能没有覆盖现实可能性，因而自然语言前提本身为假；这属于健全性失败，不是析取规则无效。

## V2.7 一阶模型中的反模型

命题赋值只能处理无内部结构的原子句。对带量词语言，反模型是一个结构 $\mathcal M$ 加变量赋值 $s$：结构给出论域、常量、函数和关系符号的解释，满足关系写 $\mathcal M,s\models A$。例如要反驳

$$
\forall x\,(P(x)\to Q(x)),\quad \exists x\,Q(x)
\models \exists x\,P(x),
$$

取单元素论域 $\{a\}$，令 $P^{\mathcal M}=\varnothing$、$Q^{\mathcal M}=\{a\}$。两前提真而结论假。

本书不在此重建一阶逻辑递归语义；第三章只使用量词满足条件，完整元理论列为外部输入。

## V2.8 反模型的证明力边界

反模型精确证明 $\Gamma\not\models A$，或以单个对象反驳全称命题。它不自动证明：

- 哪个现实前提实际上为假；
- 某个替代理论为真；
- 一个概率主张的数值错误；
- 某方法“通常”失败。

对概率或平均主张，需要给出概率空间、分布、估计量与不确定性。对因果主张，需要比较满足观察事实但干预预测不同的模型，第七章将给出完整例子。

## V2.9 推导层暂不等于语义层

本章证明的是 $\models$ 层面的结论。符号 $\vdash_D$ 还需要明确推导系统 $D$。要由 $\Gamma\vdash_D A$ 推到 $\Gamma\models A$，必须证明 $D$ 可靠；要由语义蕴涵反推存在推导，则需要完备性。第五章严格处理这两个层级。

反模型把开头那段诊断修成了一个更诚实的分叉：延迟上升与过载相容，却没有单独识别过载；若要得到过载结论，
需要排除替代原因的额外前提或直接观测。这里完成的是逻辑结构诊断，而不是服务器故障诊断。第三章将进一步
处理另一个常见失败：有些句子甚至还没有形成类型正确、量词明确的命题，因而尚无反模型可谈。

## 练习

**练习 V2.1.** 用递归语义验证 $(A\land B)\to A$ 有效。

**练习 V2.2.** 为“如果下雨地面湿；地面湿；所以下雨”写命题反模型，并给出一个使现实前提成立的替代原因。

**练习 V2.3.** 判断 $A\to B,\neg B\models\neg A$，并只用语义定义证明。

**练习 V2.4.** 给出一个形式有效但相对于现实解释不健全的技术论证，指出假前提。

**练习 V2.5.** 分别说明反驳全称主张、存在主张和概率主张需要什么对象。
