# 《Deep Agents 实战》学习笔记｜第 3 章 虚拟文件系统

实验用的模型是 DeepSeek（deepseek-flash，走 OpenAI 兼容接口），环境是 deepagents 0.7.8、langchain 1.4.3。


## 1. 为什么用文件系统：信息放在上下文外面 + 按需取用

传统做法把文件内容、搜索结果、中间计算全部塞进对话历史。这样历史越来越长，每次调用模型都要整份重发。第 2 章我在 LangSmith 里看过一次研究任务，模型的输入从 3.15K token 涨到了 57.27K。

第 3 章的思路是给 Agent 一个文件系统：资料存进文件，需要时再读；用搜索定位；中间结果写下来。关键在"按需"：上下文里只放当前这一步用得上的内容，其余都放在文件里。比如研究助手每次的搜索结果如果写进文件，上下文里只留一个路径，后面每一轮就不用重复携带全部内容。

这对应书第 2 章上下文工程里的按需加载和大结果卸载。书讲的是原则，Deep Agents 把它落成了三样东西：一套文件工具、可以替换的存储后端、自动的上下文管理。

## 2. 七个文件工具：改局部用 edit_file + 调用成功不等于拿到全集 + 分片读和三种 grep

内置的文件工具有 7 个：

- `ls`：列目录；
- `read_file`：读文件；
- `write_file`：新建文件，或者整份覆盖已有文件；
- `edit_file`：对已有文件做精确的字符串替换；
- `delete`：删除文件或目录，v0.7 新增；
- `glob`：按文件名模式找文件；
- `grep`：按内容搜索。

`write_file` 会把同一路径的文件整份覆盖，所以只改一部分时要用 `edit_file`。

工具调用成功，不代表拿到了全部内容：

- `read_file` 不写范围时默认只读前 100 行。`offset` 是从第几行开始（从 0 算），`limit` 是读几行，返回里会告诉模型下一段从哪读。
- `grep` 命中超过上限（默认 1000 条）或者超时、`glob` 超时，都会停在中途，返回里附一句提示，说结果不完整；输出太长也会被剪掉，并提示缩小范围。

这就是书第 4 章讲的显式截断：截断时要明确告诉模型"还有更多"。拿到截断的结果，应该缩小范围继续搜。

`read_file` 也能读图片、音视频和 PDF。但这只说明工具能构造多模态内容块，还要模型本身接受这种输入才行。

`grep` 有三种输出模式：

- `files_with_matches`：只返回哪些文件里有匹配；
- `content`：按文件分组返回命中的那几行，带行号，不带前后文；
- `count`：只返回匹配数量。

可以先用 `count` 或 `files_with_matches` 定位，再用 `content` 拿到命中行的行号，最后用 `read_file` 的 `offset`、`limit` 读附近几行。这样每一步进上下文的内容都很少，第 3 节卸载实验里的模型就是这样取数的。

工具决定 Agent 能做什么；某一次操作能不能做，由权限决定，见第 6 节。

## 3. 上下文自动管理：单次太大就卸载 + 累积太多就总结

上一节的工具要模型主动去用。这一节的两件事由 Harness 自动完成，不靠模型自觉。

**大结果卸载**

一次工具调用返回的结果超过约 2 万 token 时，`FilesystemMiddleware` 会在工具执行完、结果进入对话之前做这几件事：

1. 把完整内容写进后端的 `/large_tool_results/调用编号`；
2. 对话里只留三样：文件路径、分片读取的提示、首尾各 5 行的预览；
3. 模型需要时，自己用工具读回。

所以模型从头到尾没见过全文。

超没超阈值是按字符数估算的。源码里的判断是：文字部分超过 4 × 阈值。默认阈值是 2 万，也就是 8 万字符。

几个边界：

- 只看工具的返回结果，不看调用参数。
- `ls`、`glob`、`grep`、`read_file`、`edit_file`、`write_file`、`delete` 这 7 个工具不卸载：
  - 前三个自己会截断；
  - `read_file` 如果被卸载，模型又会用 `read_file` 去读，一直绕圈；
  - 后三个只返回一句确认。
- 写进后端失败时，保留原消息。
- 超长的用户消息另有一个阈值，默认约 5 万 token。

**对话历史总结**

每次的结果都不大，但轮数多了，历史累积起来照样会超。上下文达到阈值时，Harness 自动总结：

- 阈值：模型带窗口大小信息时，是窗口的 85%；没有时，用固定的 17 万 token。我用的 deepseek-flash 拿不到窗口信息，走的是 17 万。
- 总结做三件事：
  1. 把要总结掉的旧消息写进后端留档；
  2. 用模型生成一份结构化摘要；
  3. 把这一次发给模型的内容换成"摘要 + 最近的消息"。最近的消息保留多少：有窗口信息时，保留窗口的 10%；没有时，保留 6 条。

要分清两样东西：状态里保存的历史，和模型这次看到的内容。原始消息在状态里正常还在，模型看到的是摘要加近期消息。

一句话：卸载管"单次太大"，总结管"累积太多"。

**卸载实验**

我做了一个实验：

- 一个取报告的工具返回约 12.5 万字符的英文报告，约 2.6 万 token。"Project Falcon 的上线日期"藏在报告正中间，问 Agent 这个日期。
- A 组：什么都不配的 `create_deep_agent`，默认开着卸载。
- B 组：传 `middleware=[FilesystemMiddleware(tool_token_limit_before_evict=None)]` 关掉卸载。同名的中间件会原地替换掉默认装的那个。

每组跑 2 次，结论一致：A 组两次都卸载、都答对，B 组两次都没卸载、都答对。A 组第 2 次多搜了一次 `grep`，token 略有出入。下面是第 1 次：

```
A 开卸载   模型调用 4 次   每次输入 2802, 3275, 3425, 3927 token
           fetch_report → grep → read_file      报告进对话 1782 字符     答对
B 关卸载   模型调用 2 次   每次输入 2802, 29271 token
           fetch_report                         报告进对话 125727 字符   答对
```

A 组的过程：

- 报告是 Harness 自动存进 `/large_tool_results/` 的，不是模型存的。
- 之后的 `grep` 是模型自己决定调用的。`grep` 的工具说明里写了卸载结果放在哪个目录。
- 模型用 `content` 模式搜到答案那一行，又用 `read_file` 分片读了附近十几行核对，没有整份读回。

两组对比：

- 第 2 次调用时，A 的输入约 3.3K，B 约 29.3K。
- A 多调了两次模型，但每次运行的输入合计，A 约 1.35 万，B 约 3.21 万。
- 这次 B 组只有一轮带着整份报告。轮数越多，差距越大。
- 按 B 组的增量算，这份英文报告大约 4.75 个字符算一个 token。

实验的设计：

- 两组用同一份报告，只差开不开卸载。如果对照组换成短报告，它低于阈值，根本不会卸载。两组就同时差了"内容长短"和"有没有卸载"两个条件，token 的差别分不清是哪一个造成的。
- 关键数据放在报告正中间，是因为卸载后只留首尾预览。放在头尾，模型从预览里就直接看到了，不会去读回卸载的文件，要观察的"模型怎么从卸载文件里取数"也就看不到了。

## 4. 后端决定文件存在哪：StateBackend 是草稿纸 + 磁盘后端不是沙箱

文件工具只是一套接口，文件到底存在哪里，由后端决定。后端可以换，工具不变。

**StateBackend（默认）**

不用任何配置。文件存在 Agent 的状态里，也就是 `invoke()` 返回结果里的 `files`：一个"路径 → 内容"的字典，不会在磁盘上生成真实文件。"虚拟"文件系统的"虚拟"就是这个意思。状态本身放在内存还是数据库，由 Checkpointer 决定，见第 5 节。我在卸载实验里查过：状态里有 `/large_tool_results/…`，磁盘上没有这个目录。

状态按对话线程分开，也就是按 `thread_id` 分，这里的线程不是操作系统的线程。

- 新线程从空状态开始，不继承旧线程的文件。
- 旧文件没被删，换回原来的 `thread_id` 就能接上。
- 本地多次 `invoke()` 想延续文件，要接上 Checkpointer 并复用同一个 `thread_id`，或者自己把 `files` 传回去。
- 主 Agent 和子 Agent 共享这些文件。

它适合当草稿纸。

**FilesystemBackend**

直接读写本地磁盘。`root_dir` 是根目录。改动直接落盘，不会自动回滚。

`virtual_mode=True`（v0.7 默认）只是一条路径规则：路径里有 `..`、以 `~` 开头、解析后跑出根目录的，都拒绝。但根目录里的所有文件它都读得到，包括 `.env` 和密钥。这和"venv 不是沙箱"是同一类事。

**LocalShellBackend**

在 FilesystemBackend 的基础上多给一个 `execute` 工具，命令在本机直接跑，没有任何隔离，`root_dir` 只是命令的工作目录。它有个原文没提的参数 `inherit_env`，默认 `False`，也就是不把终端里的环境变量传给命令。原文把它定为极高风险，生产环境和多用户系统都不要用。

既要执行命令又要隔离，用沙箱后端（Modal、Daytona、Runloop 等），第 10 章讲。

**从"模型不可信"看后端**

后端决定的是模型出错时最坏能碰到什么：只是状态里的一个字典、真实磁盘，还是整台机器的 shell。研究助手本来就读网页、调外部搜索，再换成 FilesystemBackend 让它读得到 `.env`，书里说的致命三要素就凑齐了。

## 5. 跨对话和撑过重启：StoreBackend 按 namespace 分 + CompositeBackend 按路径分 + 两件事不要混

```python
backend = CompositeBackend(
    default=StateBackend(),                                              # 其余路径：线程内的草稿
    routes={"/memories/": StoreBackend(namespace=lambda rt: ("local-user",))},  # 跨对话保留
)
agent = create_deep_agent(model=model, backend=backend,
                          checkpointer=InMemorySaver(), store=InMemoryStore())
```

**StoreBackend**

文件存进 LangGraph 的 Store。Store 是一个键值存储，每个文件是一条记录，用"namespace + 路径"定位。

- `namespace` 从 v0.5 起必填，它决定数据怎么隔离，一般按用户身份分。
- 身份从运行时取，不让模型传。本地 `invoke()` 拿不到用户身份，要写个兜底值，比如上面的 `"local-user"`。
- 因为按 namespace 分组，换一个对话照样读得到同一份文件。适合用户偏好、长期记忆这类下次还要用的东西。

**CompositeBackend**

按路径前缀，把读写分给不同的后端：

- 写 `/workspace/plan.md`，进默认的 StateBackend；
- 写 `/memories/preferences.md`，进 StoreBackend。

几个细节：

- 在根目录 `ls`，能看到挂载的目录；跨路由的 `glob`、`grep` 会合并结果。
- 交给路由后端时会去掉前缀，Store 里记录的键是 `/preferences.md`。
- 存到哪由写入路径决定，所以要在提示词里告诉模型：长期要记的东西写到 `/memories/` 下面。
- v0.7 起 `backend=` 只能传后端实例，不能再传"创建后端的函数"。`namespace=lambda rt: ...` 照常可用。

**两件事不要混**

原文说 StoreBackend 是"持久化"，我一开始理解成撑得过重启。其实这是两件事：

- 按什么分组，决定能不能跨对话；
- 存在什么介质里，决定能不能撑过重启。

```
                         存在内存里                 存在数据库里
按 thread_id 分          不能跨对话，撑不过重启      不能跨对话，能撑过重启
（StateBackend + Checkpointer）
按 namespace 分          能跨对话，撑不过重启        能跨对话，也能撑过重启
（StoreBackend + Store）
```

- StateBackend 的文件能不能撑过重启，看 Checkpointer。Checkpointer 是存档器：Agent 每走一步，就把完整状态（消息、`files` 等）按 `thread_id` 存一份。`InMemorySaver` 重启就没了，存到数据库的还在。
- StoreBackend 的文件看传进去的 Store。`InMemoryStore` 重启就没了，生产环境要换成持久化的 Store。

**实验**

我用实验验证了这一点。配置：

- 一个 Deep Agent，后端是上面那种 CompositeBackend，namespace 固定；
- Checkpointer 用 `InMemorySaver`，Store 用 `InMemoryStore`。

分四步做：

```
步骤                                    /memories/prefs.md（Store）   /workspace/draft.md（t1 状态）
1 线程 t1 写两个文件                     写入                          写入
2 新线程 t2 读                           读到                          读不到
3 回到 t1 读                             读到                          读到
4 Checkpointer 留下、Store 换成空的，    读不到                        读到
  再回到 t1 读
```

第 4 步模拟的是"Checkpointer 存在数据库、Store 存在内存"时的一次重启。进程并没有真的重启。

结果：

- 第 4 步 prefs 跟着 Store 一起没了，draft 跟着 Checkpointer 留了下来。能不能撑过重启，看存储介质，不看后端名字。
- 另一个现象：第 4 步模型读不到 prefs 后，自己又 `ls` 了两次，最后说这次读取结果和上次不同。原因是第 3 步读到的原文还在 t1 的消息历史里。文件没了，读过的内容不一定从上下文里消失。

实验的设计：

- 第 4 步回到 t1 读，而不是开新线程读。因为 draft 在新线程里本来就读不到，开新线程就分不出是重启弄丢的，还是换了线程。回到 t1，第 3、4 步就只差"重启"这一个条件。
- 清空 Store、留下 Checkpointer，而不是反过来。因为反过来的结果（prefs 在、draft 没了）和"State 临时、Store 持久"的直觉一样，"看后端名字"和"看存储介质"两种说法预测相同，分不出谁对。现在这样做，两种说法的预测正好相反。

## 6. 写规则拦：FilesystemPermission 取第一条命中 + 删除按目录范围算

模型不可信：它会犯错，也会被网页里的指令带偏。所以"不许碰"不能写在提示词里求它，要在工具真正读写之前由代码拦下来。最省事的办法是声明式权限：

```python
agent = create_deep_agent(
    model=model,
    backend=backend,
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/policies/drafts/**"], mode="allow"),
        FilesystemPermission(operations=["write"], paths=["/policies/**"], mode="deny"),
    ],
)
```

**一条规则由三样组成**

- 操作：只有两类。`read` 管 `ls`、`read_file`、`glob`、`grep`；`write` 管 `write_file`、`edit_file`、`delete`，删除也算写。
- 路径：模型看到的虚拟路径，以 `/` 开头，用 glob 匹配。注意 `/policies/**` 不包含 `/policies` 本身。
- 处理方式：
  - `allow`：放行；
  - `deny`：不执行，工具返回权限拒绝的错误；
  - `interrupt`：暂停，等人审批。要接 Checkpointer，第 9 章讲。

另外，`deny read` 还会把 `ls`、`glob`、`grep` 结果里被拒绝的路径过滤掉。

**求值顺序**

从上往下，第一条命中的规则说了算；一条都没命中，默认允许。所以：

- 具体的规则要放在前面。上面的例子里，`/policies/drafts/` 下的文件能改，`/policies/` 其余部分不能改。两条顺序反过来，`deny` 先命中，例外永远轮不到。
- 只配 `allow` 等于没配。要做白名单，最后得补一条 `deny /**`。

**删除不完全按顺序**

删除是递归的，删一个目录会连带删掉下面的所有东西。所以删除检查分两种情况：

- 删一个目录，或者后端确认不了它是不是普通文件：只要有一条 `deny write` 可能命中这个目录、它下面的任何路径，或者它的某个上级目录，就拒绝，不看顺序。
- 删一个确认过的普通文件：它没有下级要保护，就和写一样，取第一条命中的规则。

我实测过：规则是 `allow /workspace/drafts/**` 在前、`deny /workspace/**` 在后。删文件 `/workspace/drafts/a.md`，放行；删目录 `/workspace/drafts`，被拒绝。

**管不到 shell**

路径规则只管文件工具，管不到 shell，`execute("cat /policies/rules.md")` 就能绕过去。所以配了 `permissions`，而传给 Agent 的后端带 `execute`（LocalShellBackend、沙箱后端；如果是 CompositeBackend，看它的 `default`）时，创建 Agent 会直接报 `NotImplementedError`。例外是每条规则的路径都以 CompositeBackend 的某个路由前缀开头。这个检查只比路径前缀，不看路由到的后端有没有 shell。

## 7. 写代码拦：自定义后端 + 包装器比继承好换 + 删除要拦上级目录

规则只能表达"哪些路径、读还是写、放行还是拒绝还是审批"。要限速、留审计日志、检查写进去的内容，或者要把文件存进 S3、公司的 Postgres，就得在后端写代码。

**自定义后端**

实现 `BackendProtocol` 这个接口：

- 核心是 `ls`、`read`、`write`、`edit`、`grep`、`glob` 六个方法；
- 要给模型删除能力，再实现 `delete`。直接交给 Agent 的后端没实现 `delete`，模型就没有这个工具。包在 CompositeBackend 里时不一样：它自己实现了 `delete`，所以这个工具总会出现；路由到的后端不支持删除，调用时返回错误；
- 每个方法返回一个结果对象。出错时把说明放进 `error` 返回，不抛异常，和第 2 章"让错误回到上下文"是同一个做法；
- 异步版本（`awrite` 等）默认就是在后台线程里调同步版本，所以只写同步的也能用。

**两种加检查的写法**

- 继承现有后端，原文叫 GuardedBackend。比如继承 `FilesystemBackend`，覆盖 `write`、`edit`、`delete`：先检查，通过了再用 `super().write(...)` 调父类原来的方法。
- 包装器，原文叫 PolicyWrapper。自己实现一个后端，里面装着真正干活的 `inner`：读类方法原样转给 `inner`，写类方法先检查再转。

包装器什么后端都能包。明年要从本地磁盘迁到 S3，只换 `inner`；继承绑死了一种后端，换存储就要重写一个类。

**检查只管经过它的调用**

两种写法都要记住这一条：

- LocalShellBackend 的 `execute("rm -rf policies")` 不经过 `write`、`delete`，拦不到。
- 继承时，没覆盖的方法沿用父类的实现，不经过检查。比如 FilesystemBackend 的 `upload_files` 照样直接写盘，它是给应用代码用的，不是模型的工具。
- 父类如果自己实现了某个方法的异步版本，继承时只覆盖同步方法，异步运行时这个方法就绕过了检查。要逐个方法看：StoreBackend、CompositeBackend 的写、改、删都有自己的异步实现；沙箱后端只有写、改有，删除没有，异步删除会调到你覆盖的同步 `delete`。
- 包装器继承的是接口，没转发的方法走接口的默认实现，直接报 `NotImplementedError`。结果是这个方法用不了，不是没人守。其中 `delete` 没转发的话，模型直接看不到这个工具。

**删除要按目录层级拦**

v0.7 的删除是递归的。原文的旧示例用字符串前缀 `/workspace/policies/` 判断，能拦住写 `/workspace/policies/rules.txt`，但漏掉两种删除：

- 删 `/workspace/policies` 本身：路径结尾没有 `/`，前缀对不上；
- 删上级 `/workspace`：前缀也对不上。

这两种删除执行下去，受保护的内容都会被连带删掉。

改法是用 `PurePosixPath(...).is_relative_to(...)` 按目录层级比，不按字符串比：

- 写和改：目标是受保护目录本身，或者在它下面，就拒绝；
- 删除：除了上面两种，目标是受保护目录的上级，也拒绝。

按层级比还有一个好处：不会误拦 `/workspace/policies-copy` 这种名字开头一样的目录。按字符串比时，前缀结尾不带 `/`（像下面实验的第 2 组）就会误拦它。另外，每一个能改数据的入口都要守。只守 `write`、`edit`，模型用 `delete` 照样能删。

**实验**

我用真实模型跑了一次。Store 里有三个文件：制度文件 `/shared/policies/rules.md`，草稿 `/shared/tmp/old1.md`、`/shared/tmp/old2.md`。四组都让 Agent 调一次 `delete`，把 `/shared` 整个删掉；被拒绝就停下，报告拒绝信息。

```
1 不保护（StoreBackend）
  delete /shared → Deleted /shared                                     制度文件没了
2 前缀包装器（以 /shared/policies 开头就拒绝）
  delete /shared → Deleted /shared                                     制度文件没了
3 层级包装器
  delete /shared → 拒绝删除 /shared：它包含受保护目录 /shared/policies   三个文件都在
4 FilesystemPermission（write /shared/policies/** deny）
  delete /shared → Error: permission denied for write on /shared
                   (matches deny rule(s): /shared/policies/**)          三个文件都在
```

几点观察：

- 第 2 组：`/shared` 不以 `/shared/policies` 开头，前缀检查放行了，递归删除把制度目录一起删掉。
- 第 4 组：没写一行代码，内置的删除检查已经处理了上级目录。
- 第 4 组报错里的 matches 不是 glob 匹配。`/shared` 本身对不上 `/shared/policies/**`，这里的意思是 `/shared` 下面的范围和规则保护的路径有重叠。第 4 组的模型在回复里就把它误读成了"该路径匹配规则"。
- 第 1、2 组的模型都报告"删除成功"。它照请求做了，并不知道哪些该保护。制度文件保不保得住，只看代码拦不拦。

实验的设计：

- 第 1 组什么都不保护，是为了证明不拦的话，这个请求真会发出，删目录也真会递归删掉制度文件。没有这一组，第 3、4 组的"还在"就分不清是保护拦下的，还是模型根本没删、删目录本来就不递归。
- 四组 `delete` 的参数都是 `/shared`，所以结果的差别来自保护方式，不是模型做了不同的事。

**怎么选**

- 规则能写成"路径 + 读写 + 放行/拒绝/审批"的，用 `FilesystemPermission`。不用写代码，删除时的上级目录也处理好了。
- 要限速、审计、查内容，或者接自己的存储，才写包装器。这时目录层级、递归删除这些边界都要自己处理。
- 只想去掉 `delete` 工具，可以在 `FilesystemMiddleware(tools=...)` 里不列它。但这管不到自定义工具和 shell 里的删除。

## 8. 第 3 章串起来：上下文有限 + 模型不可信 + 文件活多久、谁看得见

第 3 章的细节都在应对两个约束：

- **上下文有限**：文件系统让信息放在上下文外面，用到哪部分取哪部分。大结果卸载、历史总结，是 Harness 自动替模型把东西挪出去。
- **模型不可信**：后端决定它最坏能碰到什么；权限和策略决定在能碰到的范围里，哪些路径能读、能写。两者都在执行层拦，不靠提示词。

后端还有一个工程上的作用，和这两个约束无关：文件能活多久、谁看得见。按 `thread_id` 分组还是按 `namespace` 分组，决定能不能跨对话；存储介质决定能不能撑过重启。

判断一个文件系统设计，我会问三件事：

1. 上下文放不放得下，该不该放进去；
2. 模型不按提示词做怎么办；
3. 文件该存多久，谁能看到。

举个例子，一个团队共用的代码审查 Agent：

- 审查意见要让组里每个人开新对话都读得到，就存进 StoreBackend，namespace 按仓库分，不按人分。
- 一次审查用的草稿，放默认的 StateBackend 就行。
- 仓库代码不要整份拼进系统提示词，告诉模型路径，让它按需读。
- 不许改的目录用 `deny write`，不靠提示词约束。
