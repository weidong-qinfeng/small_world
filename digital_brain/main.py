"""数字大脑 MVP 入口文件

设计哲学（核心原则）：
    大脑初始没有任何能力，只会「学习单词及含义」。
    启动后是空白脑，需要人工执行 learn 命令加载知识包后才能答题。

运行示例：
    python3 digital_brain/main.py          # 交互模式
    python3 digital_brain/main.py "1+1=?"  # 非交互模式（需先 learn）
"""
from __future__ import annotations

import sys
import os

# 保证从项目根目录运行时可以 import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_brain.src.interfaces.symbolic_interface import SymbolicInterface


def build_brain() -> SymbolicInterface:
    """构建一个空白大脑（不做任何自动学习）"""
    return SymbolicInterface(
        auto_build=False,
        auto_learn_tokenizer=False,
    )


def demo() -> None:
    """交互式演示 —— 空白脑启动，人工触发学习"""
    print("=" * 60)
    print("     数字大脑 MVP")
    print("     初始空白，通过学习知识包获得能力")
    print("=" * 60)
    print()

    brain = build_brain()
    print(f"[空白脑已启动] 实体={brain.declarative.entity_count}, "
          f"程序={brain.procedural.procedure_count}, "
          f"词素={len(brain.tokenizer.known_morphemes)}")
    print()

    # 列出可用知识包
    packages = SymbolicInterface.list_packages()
    if packages:
        print("可用知识包:")
        for p in packages:
            print(f"  - {p}")
        print()
    else:
        print("(暂无知识包)")
        print()

    print("命令:")
    print("  learn <包名>     学习知识包")
    print("  packages         列出可用知识包")
    print("  status           查看大脑当前知识状态")
    print("  <问题>           直接输入问题求解")
    print("  q                退出")
    print()

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue
        if text.lower() in ("q", "quit", "exit"):
            print("再见。")
            break

        # ---- learn 命令 ----
        if text.lower().startswith("learn "):
            pkg_name = text[6:].strip()
            if not pkg_name:
                print("用法: learn <知识包名>")
                continue
            try:
                stats = brain.learn_from_package(pkg_name)
                s = stats.get("learned_from_package", {})
                print(f"[学习完成] 知识包 '{s.get('package')}' 已加载")
                print(f"  数字={s.get('numbers', 0)}, 操作符={s.get('operators', 0)}, "
                      f"标记={s.get('markers', 0)}, 程序={s.get('procedures', 0)}, "
                      f"分词样本={s.get('tokenizer_samples', 0)}")
                print(f"  当前: 实体={brain.declarative.entity_count}, "
                      f"程序={brain.procedural.procedure_count}, "
                      f"词素={len(brain.tokenizer.known_morphemes)}")
            except FileNotFoundError as e:
                print(f"[错误] {e}")
            print()
            continue

        # ---- packages 命令 ----
        if text.lower() == "packages":
            pkgs = SymbolicInterface.list_packages()
            if pkgs:
                print("可用知识包:")
                for p in pkgs:
                    print(f"  - {p}")
            else:
                print("(暂无知识包)")
            print()
            continue

        # ---- status 命令 ----
        if text.lower() == "status":
            print(f"实体={brain.declarative.entity_count}, "
                  f"关系={brain.declarative.relation_count}, "
                  f"程序={brain.procedural.procedure_count}, "
                  f"词素={len(brain.tokenizer.known_morphemes)}")
            print()
            continue

        # ---- 当作问题求解 ----
        result = brain.solve(text)
        if result.answer is None:
            print(f"[无法回答] 大脑尚不具备相关知识。")
            print(f"  当前: 实体={brain.declarative.entity_count}, "
                  f"程序={brain.procedural.procedure_count}")
            print(f"  提示: 使用 'learn <知识包名>' 学习后再试")
        else:
            print()
            print(result.format())
        print()


def main(argv: list) -> int:
    if len(argv) <= 1:
        demo()
        return 0
    # 非交互模式：python3 digital_brain/main.py "1+1=?"
    # 需要先 learn，否则空白脑答不出
    brain = build_brain()
    for q in argv[1:]:
        result = brain.solve(q)
        if result.answer is None:
            print(f"[无法回答 '{q}'] 大脑尚不具备相关知识。")
            print(f"  提示: 交互模式下执行 'learn 1plus1' 后再试")
        else:
            print(result.format())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
