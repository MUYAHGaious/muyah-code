import argparse


def main():
    parser = argparse.ArgumentParser(description="Greet someone")
    parser.add_argument("name")
    args = parser.parse_args()
    print(f"Hello, {args.name}!")


if __name__ == "__main__":
    main()
