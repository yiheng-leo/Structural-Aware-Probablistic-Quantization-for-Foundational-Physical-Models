# 历史材料

本目录保留目录整理前的实现、入口和实验产物。生产代码不导入这里，pytest 默认也不收集这里。

- legacy/PPQ、SAPQ、DynamicQ、StaticQ、BRECQ 等：各方法的旧入口、旧 trainer 和研究原型。
- legacy/unified_quantization：本次目录整理前统一模块的完整源码快照。
- legacy/root、launchers：原根目录实验脚本及重复启动脚本。
- results：历史图表、缓存与结果，按原来源分组。

归档保留来源和相对结构，但不保证旧脚本在新目录直接运行；它们可能依赖旧的绝对集群路径、
旧模块命名以及未附带的 checkpoint/校准缓存。正式使用方法见 ../README.md。
原仓库初始提交仍保留更早历史；本次迁移没有删除实验数据内容。
