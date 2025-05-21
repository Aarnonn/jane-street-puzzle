import itertools

def generate_permutations(input_path, output_path, buffer_size=1000000):
    # Load words
    with open(input_path, "r") as f:
        words = [line.strip() for line in f if line.strip()]

    total = len(words) * (len(words) - 1)
    print(f"Generating {total:,} two-word permutations...")

    # Generator for 2-word permutations
    perm_gen = itertools.permutations(words, 2)

    with open(output_path, "w") as out:
        buffer = []
        for i, (a, b) in enumerate(perm_gen, 1):
            buffer.append(f"{a} {b}\n")
            if len(buffer) >= buffer_size:
                out.writelines(buffer)
                buffer.clear()
                print(f"Written {i:,}/{total:,} permutations...", end="\r")

        # Final flush
        if buffer:
            out.writelines(buffer)

    print(f"\nDone. All permutations written to: {output_path}")

if __name__ == "__main__":
    generate_permutations("Google 10000 English USA.txt", "two_word_permutations.txt")