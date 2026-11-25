# 最终Prod对当前Final3b：新鲜成对公平复核

4K/32K初始KV、1024输出的受影响正式N5均正向，完整输出与MTP工作量逐位一致。这里比较当前Final3b的已验证最快FLAGS1与最终Prod默认路径；不把旧Safe397结论外推为最终Prod。

| 初始KV | 顺序 | Final Decode tok/s | Prod Decode tok/s | Decode改善 | Final含TTFT tok/s | Prod含TTFT tok/s | 改善 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode-4k | Prod→currentFinal3b | 109.53350276 | 110.95069936 | 1.293848% | 103.98448248 | 106.54316565 | 2.460639% |
| decode-32k | currentFinal3b→Prod | 96.13240087 | 96.32994567 | 0.205492% | 65.34537199 | 69.15444050 | 5.829133% |

两臂各2轮预热/5轮测量；同检查点、LCG输入、capacity、greedy/fixed MTP3、NVFP4、动态FP8 KV及1150份有效recipe，禁止在线调优/fallback。Decode使用首token后的1023间隔；含TTFT吞吐使用全部1024输出和总时延。GPU固定1530MHz（原最高1575），CPU2601MHz、EMC4266MHz。

Pure3b实际a012构建与9c7177收据闭合：native ae7ad71a、MTP069e4c7f；actual CUDA Rust/model库与examples均重建，恰gdn_ops.o/runner.o变化，其他native成员及调优身份不变，原11成员AOT和四份临时移开文档/源码元数据恢复。Prod保留15a0/native774/MTP588a不可变产物及其新12成员AOT；当前mutable cache单列为pure3b实际构建后库存。

原parser已重解析strict/screen/formal共12进程52轮，所有raw SHA/字节、配置、完整token/FNV、版内工作量、频率及恢复守卫通过；本次跨臂work也恰好一致。大ELF/native仍在远端不可变快照；本地核的是实际构建stdout和收据/运行身份，没有虚称重新下载并rehash大文件。

**范围限制：**只覆盖当前Final的4K/32K两个Decode1024 case，不代表六档/15项当前Final全部无回退或Decode提高20%。旧冻结主表32K的−1.81%点保留，本次没有定位其原因。TTFT改善不替代追加4K Prefill benchmark。两个20%目标仍按原完整主表单独验收，推送状态由root实际操作另记。

原包SHA：`a9d897c24054280a49abe28ce3e048fb2dce8835924b25ff87d6bf84a174960a`；各轮样本/波动、实际source/build/ELF/recipe身份和频率恢复见同目录[逐轮证据](evidence/current-final-fair-thor3.json)。
