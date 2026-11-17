# 《Deep Agents 实战》学习笔记｜第 1–2 章 认识 Deep Agents 与快速上手

实验用的模型是 DeepSeek（deepseek-flash，走 OpenAI 兼容接口），环境是 deepagents 0.7.8、langchain 1.4.3。


## 1. Deep Agents 的定位：Runtime + Framework + Harness

已经有 LangChain 和 LangGraph，为什么还要 Deep Agents？原文的例子是做一个编程助手，要读代码、定计划、分步执行并追踪进度、把子任务交给别的 Agent、记住用户偏好。用 LangChain 从头写，文件读写、任务追踪、子 Agent 调度都要自己实现，而几乎每个复杂一点的 Agent 都需要这些。

原文把技术栈分成三层，从下往上依次构建。Runtime 是 LangGraph，负责 Agent 可靠地运行：断点恢复、流式输出、人工审批时暂停、跨对话保存状态。Framework 是 LangChain，提供模型抽象、工具接口、Agent 循环和中间件，构建在 LangGraph 之上。Harness 是 Deep Agents，在前两层上预置一套工具和中间件，`create_deep_agent()` 内部调用的就是 `create_agent()`。

三层不能互相替代。选起点时看哪些工作已经做好、哪些要自己写：需要文件操作、上下文管理和子任务委派的多步骤任务用 `create_deep_agent()`；只要标准的模型和工具循环、想自己决定上下文策略用 `create_agent()`；流程有固定步骤和分支、要精确控制路径用 LangGraph 的 `StateGraph`。没有必须自定义的流程时先用 Deep Agents，不够用再往下走，三层也可以组合。

需要 Harness 这一层，是因为 Claude Code、Manus、Cursor 这些能完成复杂任务的产品核心能力很像：都能读写和搜索文件，能把大任务拆成步骤，能把工作交给子 Agent，有上下文管理。Harness 把这些验证过的做法固定下来。

书里的 Harness 和课程里的用法不同。书里 Agent = Model + Harness，Harness 是模型之外、Agent 之内的代码，任何 Agent 都有；课程里的 Agent Harness 是三层中最上面那一层，是一个把常用 Harness 能力预先装好的套件。

## 2. Context Engineering 与技术全景：按需取用 + 可插拔 Backend + Middleware 装配

把 20 个文件、5 万 token 一次放进 prompt 有三个问题：超出上下文窗口，信息越多模型越抓不住重点，项目大了没法处理。Deep Agents 给 Agent 一个虚拟文件系统：需要时 `read_file`，大文件用 `offset` 和 `limit` 只读一部分，中间结果 `write_file` 写下来，用 `grep`、`glob` 搜索，上下文里只留当前这一步需要的内容。背后的存储叫 Backend，可以是内存、本地磁盘、数据库、远程沙箱，也可以按路径混合。这对应书第 2 章的上下文工程，书讲原则，Deep Agents 把它做成了文件工具加可替换的存储层。

和 Claude Agent SDK、Codex SDK 比，Deep Agents 不绑定模型提供商，后端可以替换，用 LangGraph Store 和 Backend 做跨线程的长期记忆，可以用 LangSmith 部署和追踪。我这个项目能接 DeepSeek，靠的就是模型无关。

整体架构是在 LangChain 的 Agent 循环周围装配 Middleware，最后由 LangGraph 执行。能力分四组：执行环境（文件系统、沙箱、Interpreter、MCP）、上下文管理（摘要、大结果卸载、Skills、记忆）、任务委派（子 Agent、任务规划）、人机控制（人工审批、文件权限）。Tool 是模型可以请求的动作；Middleware 提供能力，并在模型调用和工具执行时介入；Backend 负责真正的存储和环境操作。例如 `FilesystemMiddleware` 提供 `read_file` 这个工具，工具通过 Backend 读到内容。

v0.7 起这些能力不再默认全开：文件系统、子 Agent 和摘要默认装上，任务规划要传入 `TodoListMiddleware`，Skills、记忆、人工审批要单独配置，基础提示词默认为空。

## 3. 最小的 Deep Agent：invoke 的输入输出 + 自带的 9 个工具

```python
agent = create_deep_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]})
print(result["messages"][-1].content)
```

`create_deep_agent()` 返回一个编译好的 LangGraph 图，`invoke()` 交进去一份状态，拿回更新后的状态。实际打印出来是这样的：

```
result 的键：['messages', 'files']

[0] HumanMessage   北京今天天气怎么样？
[1] AIMessage      → 请求调用工具 get_weather({'city': '北京'})
[2] ToolMessage    ← It's always sunny in 北京!
[3] AIMessage      北京今天天气晴朗，阳光明媚！☀️ …
```

`messages` 是完整的对话历史，最后一条是最终回复。第 1 条 AI 消息只有工具调用请求；Harness 执行工具，把结果作为 ToolMessage 追加，再把整个列表发给模型，模型第二次才回答，所以一次 `invoke()` 里模型被调用了 2 次。系统提示词不在 `messages` 状态里，每次调用模型时临时拼在最前面。`files` 是虚拟文件系统的状态。

在模型被调用前截获请求可以看到，只给一个 `get_weather`，模型实际看到 9 个工具：`ls`、`read_file`、`write_file`、`edit_file`、`delete`、`glob`、`grep`、`task` 和 `get_weather`，系统提示词就是我写的 28 个字符。加上 `TodoListMiddleware` 后多了 `write_todos`，系统提示词变成 1400 个字符。内置工具的用法写在各自的定义里，8 个加起来大约 1 万字符，每次请求都要带上。短任务不必为用不到的能力付这部分成本，所以 v0.7 把任务规划改成了可选。

`invoke()` 的第二个参数 `config` 可以挂回调、设轮数上限等，Deep Agents 把 `recursion_limit` 默认值从 LangChain 的 25 改成了 9999，因为长任务一次 `invoke()` 可能要转几十上百轮。

## 4. 工具的定义：函数即工具 + 三要素 + docstring 放在哪

一个普通的 Python 函数就是一个工具。以前要分两处写：一份 JSON 定义发给模型，一个执行函数负责调用。在 Deep Agents 里只写一个函数放进 `tools=[...]`，Harness 从函数生成定义；模型返回工具调用后，Harness 按名字找到函数、把参数文本解析成 Python 的值、调用函数、把返回值包成 ToolMessage 追加。普通函数在创建工具节点时经过 `@tool` 转换成工具对象。

函数里有三要素进到定义：类型标注变成参数的 `type`，docstring 变成 `description`，有默认值的参数不进必填列表 `required`。原文的总结是类型标注决定输入形态，docstring 决定调用时机，默认值减少必填项。描述写得太宽泛，模型会乱用；没写清覆盖范围，该用的时候又可能不用。

docstring 里 `Args:` 那几行参数说明，直接传普通函数时整段放在顶层的 `description` 里，参数上只有类型；写成 `@tool(parse_docstring=True)` 后，顶层只剩第一句，参数说明拆到各个参数自己的 `description` 里。去掉类型标注不报错，只是参数没有类型；去掉 docstring 会直接报错：

```
ValueError: Function must have a docstring if description not provided.
```

## 5. 工具的参数与安全：一个字符串参数 + 白名单 + 默认值和 InjectedToolArg

计算器工具 `calculate` 只有一个参数 `expression: str`，用一个字符串承接整个表达式，带括号的多步运算一次就能算完，docstring 里给了参数例子 `e.g. "1 + 2 * 3"`。表达式是模型生成的，函数里没有用 `eval`，而是先限制长度和复杂度，再用 `ast.parse` 解析成语法树，只放行数字、加减乘除和正负号。这条检查不能只写在系统提示词里，表达式可能来自用户或网页内容，被注入后模型自己也察觉不到。按书里的三层护栏，系统提示词是上下文层，写在函数里的白名单是执行层。

`convert_currency` 的 `to_currency` 默认是 `"CNY"`，只换成人民币时模型填两个参数就够。我这个项目的 `tavily_search` 用 `Annotated[int, InjectedToolArg]` 把 `max_results` 和 `topic` 从定义里去掉了，打印出来只剩 `query`。默认值是模型可以不填、想填也能填；`InjectedToolArg` 是模型看不到、填不了。员工 ID 这类身份和权限参数应该这样藏起来，由程序从会话里取，否则用户可以借模型查到别人的数据；`max_results` 这类决定往上下文塞多少内容的参数，也可以由程序统一定来控制成本。

## 6. 工具出错：抛异常会中断 + 让错误回到上下文

原文的 `convert_currency` 遇到汇率表里没有的币种会抛 `KeyError`。我问"把 1000 日元换成人民币"，三种写法各跑 3 次：

```
A 直接抛 KeyError        崩溃 3/3   invoke() 直接抛出 KeyError: 'JPY'
B 返回错误说明           崩溃 0/3   模型调用 2 次，3 次都告诉用户不支持日元
C ToolException          崩溃 0/3   模型调用 2 次，3 次都告诉用户不支持日元
```

Harness 默认只把参数校验的错误转成消息还给模型，其他异常原样抛出，所以 A 组的 `KeyError` 一路冒出 `invoke()`，没有 ToolMessage，模型不会被第二次调用，用户什么也收不到。B 组在工具里先检查，返回 `{"error": "不支持的币种：JPY。目前只支持 USD, CNY, EUR。"}`；C 组抛 `ToolException`，工具设了 `handle_tool_error=True`，工具对象自己接住异常，把错误信息作为结果返回，并把消息标成 `status="error"`。B、C 组每次运行里模型和工具来回一趟：模型请求 `convert_currency`，工具返回不支持 JPY，模型据此回复用户。

错误信息和 docstring 一样是写给模型看的，要写清错在哪、允许什么、下一步能怎么办。只返回 `{"error": "failed"}` 的话，模型可能只说出错了、自己编个原因，或者换着参数反复重试。

这个实验的设计有三点：用不支持的币种制造错误，是为了每次都出错，三组遇到同一个错误，差别只来自出错后的处理方式，而且这是真实会遇到的业务边界；加 C 组是控制变量，C 和 A 都抛异常，只差有没有人接住，对比后能确定 A 崩溃的原因是没人接住；每组跑 3 次，是因为 A 组的崩溃是确定的代码路径，B、C 组怎么回复用户是模型行为，有随机性，3/3 也只能说明不常犯错。


## 7. 研究助手与执行循环：搜索工具的参数 + 模型—工具循环 + Middleware 的两种挂载

原文的研究助手用 Tavily 做搜索工具 `internet_search`，`query`、`max_results`、`topic`、`include_raw_content` 四个参数全部开放给模型。模型可以自己把结果条数调大，也可以要求返回网页原文。从上下文有限这个约束看，这样做有几层代价：结果太大可能撑爆窗口，信息越多重点越容易被淹没，而且历史每轮都要重发，一次大结果之后的每次模型调用都要再带上它。我这个项目的 `tavily_search` 走的是另一条路，用 `InjectedToolArg` 把 `max_results` 和 `topic` 藏起来，模型只能填 `query`。实测原文的写法，模型多次把 `max_results` 调成了 6 或 8。

系统提示词只写研究目标、证据要求和输出边界。工具的参数和用法已经在工具定义里，每次请求都会带上，提示词里再写一遍既重复花 token，工具改了还要同时改提示词，两边容易对不上。原文的提示词只有三句，证据要求和输出边界其实都没写。

一次 `invoke()` 背后是一个模型—工具循环。用 `agent.get_graph()` 把开了 TodoList 的研究助手打印出来，结构是 `before_agent → model → after_model`，`after_model` 之后分两路：模型请求了工具就去 `tools`，执行完回到 `model`；没请求就结束。读书时年假循环里的 `if not tool_calls: break`，在这里变成了图上的一条条件边。

Middleware 挂在循环上有两种方式。`before_*`、`after_*` 在两步之间单独执行，图上是独立的节点。`wrap_model_call`、`wrap_tool_call` 包住模型调用或工具执行本身，和面向切面的环绕通知一样，进、出各有一个切点；它们在 `model`、`tools` 节点的函数里执行，所以图上看不到。wrap 的前后之所以不能拆成两个节点，是因为前后必须在同一次调用里，才能重试、短路、用 try/except 接住模型调用的异常、在后半段用前半段的局部变量。`before_model`、`after_model` 读写的是图的状态，还能决定下一步去哪，所以必须是节点。

本项目研究助手实际的包裹顺序，从外到内是 Filesystem、SubAgent、Summarization、TodoList、AnthropicPromptCaching，最里面才是模型调用。AnthropicPromptCaching 对 DeepSeek 什么都不做，但这一层照样包着。文件系统中间件在工具执行这一步还有一个兜底：单条工具结果超过约 2 万 token 时，自动写进虚拟文件系统的 `/large_tool_results/`，上下文里只留预览和路径。这个阈值按单次结果算，不是多次累计。

循环有三种结束方式：模型不再请求工具，正常给出回复；发生异常或者达到运行限制，提前结束；触发人工审批，暂停等待恢复。


## 8. 任务规划与可能路径：开启和使用是两件事 + 5 步路径不是工作流

开不开 `TodoListMiddleware` 是开发者在代码里决定的，开了模型才有 `write_todos`；开了以后用不用、搜几次、写不写文件、要不要委派，每一步都是模型运行时决定的。所以原文给的"规划、搜索、写文件、委派、写报告"5 步，是自主 Agent 可能走出的一条路径，不是工作流。

我用原文的研究助手跑了三组，每组 2 次，记录每次的工具调用顺序：

```
A 简单题 + TodoList      internet_search×10、internet_search×6          没有 write_todos
B 复杂题 + TodoList      write_todos → internet_search×27 → write_todos×2
                         ls → internet_search → write_todos → internet_search×19 → write_todos
C 复杂题 + 无 TodoList    internet_search×9 → read_file → internet_search×10、internet_search×36
```

简单题没有调用 `write_todos`，TodoList 注入的提示词本来就说简单任务不要用。但简单题也搜了 6 到 10 次，写了近万字，因为原文的系统提示词要求"深入研究、写完整报告"，路径是提示词、任务和模型共同决定的。B、C 两组只差 TodoList，搜索次数一组是 27、20，一组是 19、36，看不出谁更多；稳定的区别只有 B 组两次都写了 5 项计划并更新进度。6 次运行都没有 `write_file`、`task`，也没有触发自动卸载，每次搜索只返回几条摘要，离阈值很远，提示词里也没要求保存中间结果。

这个实验的设计：A、B 只换题目，看任务复杂度的影响；C 和 B 只差 TodoList，看 TodoList 本身带来了什么；记录工具调用顺序，是因为要比较的是过程，搜了几次、有没有先写计划，在最终报告里都不留痕迹。同一组内部的波动也很大，C 组两次差了将近一倍，每组 2 次只能看出方向。


## 9. 模型接入与 LangSmith：换平台只改配置 + 三个地址各管各的 + 看 Trace 排查

Deep Agents 能用的模型只有一个前提：支持工具调用，因为整个循环靠模型请求工具推动。行业里有两种事实标准的接口，OpenAI 兼容的 `/chat/completions` 和 Anthropic 兼容的 `/messages`。接法有三种：`ChatOpenAI` 加 `base_url`，`ChatAnthropic` 加 `base_url`，或者直接写 `"openai:gpt-4.1"` 这样的字符串。我这个项目用 `init_chat_model(model_provider="openai", base_url=...)`，等价于第一种，模型名、Key、地址都放在 `.env` 里。从 DeepSeek 官方换到别的平台，只改这三项，代码不用动。

LangSmith 在 `.env` 里配好 `LANGSMITH_TRACING`、Key、项目名、Endpoint 就自动上报，不写追踪代码。三个地址各管各的：`LANGSMITH_ENDPOINT` 是程序上报 Trace 的地址，浏览器控制台是人查看 Trace 的地方，模型的 `base_url` 是调用模型的地址。不写 Endpoint 时默认连美国区，Key、Endpoint、控制台要属于同一个区域。`.env` 也不会自动生效，要由代码或启动工具加载，改了要重启进程。所以 Agent 能正常回答、但看不到 Trace 时，不用先查 `base_url`，能回答说明模型那条路是通的，要先查区域、Key、进程有没有读到新变量，再查项目名和 Key 的权限。我这个项目 `.env` 里的 Key 只能上报、不能用 API 读回，看 Trace 只能去控制台。

Trace 里能看到每次模型调用的输入和输出、工具调用的参数和返回值、子 Agent 的过程、每一步的 token 和耗时。结论错了时，找到产生错误结论的那次模型调用，看它的输入里有没有需要的正确信息：有，是模型判断错了；没有，是信息没进输入，再往前追是没调工具、参数不对、返回里本来就没有，还是结果太大被卸载只剩预览。

我在控制台里看了 L03 实验 B 组的一次运行。第一次模型调用的输入是 3.15K token，最后一次是 57.27K，开头那部分主要是系统提示词和工具定义，后面涨的是每轮追加的搜索结果和回复。图上看不到的 wrap 层在 Trace 里是有的，默认折叠在 `model` 下面，展开后能看到 `FilesystemMiddleware.wrap_model_call` 一层套一层，包在真正的模型调用外面。图上的节点是注册的步骤，Trace 里的记录是运行时被追踪包住的调用，这是两回事。
