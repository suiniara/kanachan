# 原始牌谱到训练数据

这条流水线只读取原始 Mahjong Soul WebSocket 响应，产出当前代码使用的
八列 BC、连续 RL 和 EOR 数据。Python 阶段仅需标准库和 Protobuf，不依赖
Torch，也不使用 GPU。

## 筛选与清点

先从 `src/common/mahjongsoul.proto` 生成 Python 协议：

```powershell
python -m grpc_tools.protoc --proto_path src/common --python_out generated src/common/mahjongsoul.proto
```

独立环境可使用 Python 3.12、`grpcio-tools==1.76.0`、`grpcio==1.76.0`、
`protobuf==6.33.0`。无需安装整个训练项目的依赖。

```powershell
python -m kanachan.data.catalog `
  --raw-root D:\majsoul\record-data `
  --raw-root D:\majsoul\record-data-iter `
  --schema-dir C:\data-work\generated `
  --output C:\data-work\catalog `
  --workers 2 --log-file C:\data-work\logs\catalog.log
```

- 房间以每局 `head.config.meta.mode_id` 为准：12、16，即四麻半庄的玉之间、
  王座之间。历史目录后缀可能与真实房间不符，因此检查所有日期目录。
- 校验四名玩家及结果的座位、时间、终局记录和协议版本。
- 同一 UUID 按训练相关的头部字段及牌谱正文去重。传输消息编号、昵称变化
  不会误判为内容冲突；真实内容冲突会排除该 UUID 的全部副本。
- 保存源文件 SHA256、大小、修改时间，固定按 UUID 哈希划分
  90%/5%/5% 的训练、验证、测试集合。同一整局始终属于同一个集合。
- SQLite 索引和 `progress.json` 支持恢复。USB/I/O 失败会中断任务，
  不会将设备故障记成坏牌谱。`rejected` 包含房间不匹配，具体原因见摘要。
- 在输出目录创建 `pause.request` 可在批次边界暂停；移除后用原命令恢复。
  输入、协议或实现改变时，需要使用新的输出目录。

完成后生成 `manifest.{train,validation,test}.jsonl` 和 `summary.json`。
`--limit 128` 只用于小规模试跑，进度为 `limited`，不代表全量完成。
首末时间、模式数量和冲突数量都以最终摘要为准。

针对大量 Windows 小文件，枚举时的元数据会直接传给检查器，避免每个文件
重复打开句柄解析路径。读取后的大小、时间和完整 SHA256 校验保持有效。
`migrate_cache` 仅支持已明确验证兼容的旧实现：先暂停旧索引，导入到全新的
目录，并逐表比较全部记录的哈希。旧目录保留用于回滚，不在原目录强行
修改实现指纹，也不重新读取已经验证过的牌谱正文。

## 原生标注器

原项目构建继续可用。独立的 `tools/data/CMakeLists.txt` 可仅构建 CPU
标注器，不需要训练模拟器的 Python/Boost.Python 依赖。需要 Boost 头文件和
与仓库生成代码兼容的 Protobuf C++；已验证 3.20.3。

```powershell
cmake -S tools/data -B build/data -DCMAKE_BUILD_TYPE=Release
cmake --build build/data --config Release --parallel 2
```

使用 MSVC 时选择 x64 开发环境、静态 Protobuf 库和一致的 MSVC runtime，
通过 `Protobuf_INCLUDE_DIR`、`Protobuf_LIBRARY`、
`KANACHAN_BOOST_INCLUDE_DIR` 指定隔离依赖。生成的旧版 Protobuf C++
文件使用 C++17 编译，标注逻辑仍使用 C++20。

新的批量接口：

```powershell
annotate.exe --files-from files.txt --status-file new-status.tsv --continue-on-error
```

文件列表每行一个原始文件路径。状态文件必须不存在，其列为
`ok/error、path、rows、bytes、error_message`。每局先在内存中完成转换，
成功才输出；末轮发生错误也不会泄漏此前轮次的半局样本。
旧的 `annotate DIRECTORY` 用法仍有效。

## 分片准备与验收

```powershell
python -m kanachan.data.prepare `
  --catalog C:\data-work\catalog `
  --annotator C:\data-work\annotate.exe `
  --repository E:\Workspace\kanachan `
  --output C:\data-work\prepared `
  --staging-root C:\data-work\staging `
  --workers 2 --shard-games 64 --reserve-gib 20 `
  --log-file C:\data-work\logs\prepare.log
```

`--repository` 需包含当前 `bin/annotation/{rl,eor}` 转换脚本。
正式处理要求索引状态为 `complete`；`--wait-for-catalog` 可以等待清点完成。
`--allow-limited-catalog` 和 `--limit-games` 仅用于试跑。

1. 源文件按索引的大小、时间和 SHA256 校验，再复制到工作目录供标注器读取。
2. 整局按语义哈希混排，分片保持完整对局；BC 在转换结束后打乱片内样本。
   RL 保留每位玩家的完整转移顺序。划分不因混排而改变。
3. 核对房间、玩法、每个座位的段位、合法动作、最终分数；RL 额外核对每个
   下一状态和实际下一次决策一致，每位有决策的玩家恰有一个终局转移。
   极短对局中没有决策的座位不会被误判为缺失。
4. 每片记录输入指纹、文件 SHA256、行数、接纳/排除数量。坏局整局排除，
   原因保存在 `shards/.../rejections.jsonl`。默认排除率超过 2% 就停止检查。
5. 已完成分片在恢复时先校验哈希，再复用。最终按片号合并 gzip 成员，
   生成训练器可直接读取的单个文件，并保留分片便于恢复。

正式输出为 `{train,validation,test}.{bc,rl,eor}.tsv.gz`。
`dataset.json` 包含文件哈希、样本数量、来源和实现指纹；
只有 `progress.json` 为 `complete` 才代表整条准备流程结束。
需要给分片和最终合并结果保留足够空间，程序另留默认 20 GiB 的安全余量。
低空间或设备掉线时保留已完成分片，不删除原始数据或旧模型。

## 小型验证

```powershell
$env:KANACHAN_SCHEMA_DIR = 'C:\data-work\generated'
python -m unittest discover -s test/training -p test_record_catalog.py -v
python -m unittest discover -s test/training -p test_data_preparation.py -v
```

设置 `KANACHAN_ANNOTATOR` 和 `KANACHAN_SAMPLE_MANIFEST` 后，还会验证真实
对局在账户顺序改变时输出一致，以及坏局在末轮失败时不产生半局输出。
这些检查验证数据完整性，不代表模型训练效果或对局强度已得到验证。

## 边清点边转换

不必等待全部清点结束。`stream` 从索引的 SQLite WAL 快照追加读取合格局，
沿用相同的整局划分和标注校验，及时生成有独立凭据的分片：

```powershell
python -m kanachan.data.stream `
  --catalog C:\data-work\catalog `
  --annotator C:\data-work\annotate.exe `
  --repository E:\Workspace\kanachan `
  --output D:\kanachan-prepared\jade-hanchan-stream-v1 `
  --state-dir C:\data-work\stream-state `
  --staging-root C:\data-work\staging `
  --workers 1 --shard-games 64 --reserve-gib 20 `
  --log-file C:\data-work\logs\stream.log
```

状态目录记录已消费的索引游标、固定的分片成员、执行状态和实现指纹。
恢复时使用原命令、原状态目录和原输出目录。创建状态目录中的
`pause.request` 可在当前分片结束后暂停；移除后按原命令继续。
代码、输入或配置发生变化时必须使用新的状态及输出目录。

清点期间的分片是暂存产物。索引全部完成后，程序核对每个 UUID 的最终
冲突状态，重建受影响分片、检查全部源局的覆盖和集合互斥，再组装
`final/{train,validation,test}.{bc,rl,eor}.tsv.gz` 和 `final/dataset.json`。
只有最终对账与完整性检查通过才标记 `complete`。

2026-10-30 的远端真实牌谱验收覆盖了清点未结束时转换、断点复用，以及
事后发现整片两局冲突时的重建；九个最终输出与串行转换逐行多重集合一致。

## 独立全量验收

生成完成后还可用独立命令检查全部最终文件；添加 `--wait` 可提前启动，
待 `final/dataset.json` 发布后自动开始：

```powershell
python -m kanachan.data.audit `
  --dataset D:\kanachan-prepared\jade-hanchan-stream-v1\final `
  --catalog C:\data-work\catalog `
  --state-dir C:\data-work\audit-state `
  --wait --log-file C:\data-work\logs\audit.log
```

验收器逐片解析，逐局对照 BC/RL 决策，并检查合法动作、转移连续性、终局
数量、EOR 特征与原始终局分数。通过独立 UUID 成员表和完整来源的双向
关联核对覆盖、互斥、去重及排除统计。来源摘要的模式和时间范围也需匹配。
EOR 还需与 BC 的小局开始决策逐项对账，核对遗漏、重复和局末分数错位。
整场最后一个小局按原转换规则不生成 EOR 目标；只有一个小局的整场对局
可以没有 EOR 行，不会被误判为缺失。
九个最终 gzip 会完整读取以核对全部成员的 CRC、SHA256、字节和行数，
并要求与已审计分片的有序串联完全一致。

成员表和断点留在 SSD，不需要将整个数据集载入内存。恢复仍重新校验已审计
分片的压缩内容哈希，并重新读取最终文件；代码或输入指纹改变必须使用新的
验收状态目录。通过后在该目录生成 `acceptance.json`，否则保留错误与断点。
验收器只读取产物和索引，不会重新遍历原始牌谱或加载 Torch。
