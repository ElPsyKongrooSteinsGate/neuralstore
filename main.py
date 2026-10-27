"""Command-line interface for NeuralStore."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from data.import_json import import_json_file
from neuralstore import NeuralStore


def load_store(path: Path) -> NeuralStore:
    if path.exists():
        return NeuralStore.load(str(path))
    return NeuralStore(seed=7)


def parse_record(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg} (column {exc.colno})") from exc


def print_matches(matches: list[dict[str, Any]]) -> None:
    if not matches:
        print("No matching records.")
        return
    for match in matches:
        print(
            f"{match['id']}  score={match['score']:.4f}  "
            f"importance={match['importance']:.4f}  "
            f"record={json.dumps(match['record'], ensure_ascii=False)}"
        )


def run_interactive(store: NeuralStore, db_path: Path) -> None:
    print(f"NeuralStore CLI — database: {db_path}")
    print("Commands: add, import-json, read, update, delete, search, list, quit")
    print('Enter records and search queries as JSON, e.g. {"name": "Ada"}.')
    while True:
        try:
            command = input("\nneuralstore> ").strip().lower()
            if command in {"quit", "exit", "q"}:
                break
            if command in {"list", "ls"}:
                for item in store.all_records():
                    print(f"{item['id']}: {json.dumps(item['record'], ensure_ascii=False)}")
                if not store.records:
                    print("No records stored.")
            elif command in {"add", "insert"}:
                store.insert(input("Record ID: ").strip(), parse_record(input("Record JSON: ")))
                store.save(str(db_path))
                print("Record added and saved.")
            elif command == "import-json":
                path = input("JSON file path: ").strip().strip('"')
                imported = import_json_file(store, path)
                store.save(str(db_path))
                print(f"Imported {imported} record(s) and saved.")
            elif command == "update":
                record_id = input("Record ID: ").strip()
                store.update(record_id, parse_record(input("Replacement record JSON: ")))
                store.save(str(db_path))
                print("Record updated and saved.")
            elif command == "read":
                print(json.dumps(store.read(input("Record ID: ").strip()), ensure_ascii=False, indent=2))
            elif command == "delete":
                store.delete(input("Record ID: ").strip())
                store.save(str(db_path))
                print("Record deleted and saved.")
            elif command == "search":
                query = parse_record(input("Query JSON: "))
                raw_top_k = input("Number of results [5]: ").strip()
                print_matches(store.search(query, top_k=int(raw_top_k) if raw_top_k else 5))
            elif command in {"help", "?"}:
                print("Commands: add, import-json, read, update, delete, search, list, quit")
            elif command:
                print(f"Unknown command: {command}. Type 'help' for commands.")
        except (ValueError, KeyError, OSError) as exc:
            print(f"Error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Store and search JSON records using NeuralStore.")
    parser.add_argument("--db", type=Path, default=Path("neuralstore.pt"),
                        help="checkpoint file to read/write (default: neuralstore.pt)")
    subparsers = parser.add_subparsers(dest="command")
    add_parser = subparsers.add_parser("add", help="insert a JSON record")
    add_parser.add_argument("record_id")
    add_parser.add_argument("json_record", help="record as a JSON string")
    import_parser = subparsers.add_parser("import-json", help="insert records from a JSON file")
    import_parser.add_argument("json_file", type=Path, help="path to the JSON file")
    read_parser = subparsers.add_parser("read", help="read a record by ID")
    read_parser.add_argument("record_id")
    update_parser = subparsers.add_parser("update", help="replace a JSON record")
    update_parser.add_argument("record_id")
    update_parser.add_argument("json_record", help="replacement record as JSON")
    delete_parser = subparsers.add_parser("delete", help="delete a record by ID")
    delete_parser.add_argument("record_id")
    search_parser = subparsers.add_parser("search", help="search by JSON query")
    search_parser.add_argument("json_query", help="query as a JSON string")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--latent-weight", type=float, default=0.8)
    subparsers.add_parser("list", help="list all records")
    subparsers.add_parser("repl", help="start the interactive prompt")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    db_path: Path = args.db
    try:
        store = load_store(db_path)
        if args.command in {None, "repl"}:
            run_interactive(store, db_path)
        elif args.command == "add":
            store.insert(args.record_id, parse_record(args.json_record))
            store.save(str(db_path))
            print(f"Added {args.record_id!r} to {db_path}.")
        elif args.command == "import-json":
            imported = import_json_file(store, args.json_file)
            store.save(str(db_path))
            print(f"Imported {imported} record(s) from {args.json_file} into {db_path}.")
        elif args.command == "read":
            print(json.dumps(store.read(args.record_id), ensure_ascii=False, indent=2))
        elif args.command == "update":
            store.update(args.record_id, parse_record(args.json_record))
            store.save(str(db_path))
            print(f"Updated {args.record_id!r} in {db_path}.")
        elif args.command == "delete":
            store.delete(args.record_id)
            store.save(str(db_path))
            print(f"Deleted {args.record_id!r} from {db_path}.")
        elif args.command == "search":
            print_matches(store.search(parse_record(args.json_query), top_k=args.top_k,
                                       latent_weight=args.latent_weight))
        elif args.command == "list":
            for item in store.all_records():
                print(f"{item['id']}: {json.dumps(item['record'], ensure_ascii=False)}")
            if not store.records:
                print("No records stored.")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
