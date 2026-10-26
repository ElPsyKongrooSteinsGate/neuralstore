from neuralstore import NeuralStore


def main() -> None:
    store = NeuralStore(seed=7)
    store.insert("user-001", {"name": "Ada", "role": "engineer", "active": True})
    store.insert("user-002", {"name": "Linus", "role": "maintainer", "active": True})

    print(store.read("user-001"))

    matches = store.search({"role": "engineer"}, top_k=3)
    for match in matches:
        print(match["id"], match["score"], match["record"])

    store.save("neuralstore.pt")
    restored = NeuralStore.load("neuralstore.pt")
    print("Restored records:", restored.all_records())


if __name__ == "__main__":
    main()
