# python ./visualize_throughput.py ./sparse.log ./normal.log --label1 "SPARSE" --label2 "FULL" --output throughput_compare.png

import re
import matplotlib.pyplot as plt
import argparse

# 正则表达式：只匹配 throughput 值
pattern = re.compile(r"Avg generation throughput: ([\d.]+) tokens/s")

def parse_log_file(log_path, label):
    throughputs = []
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                try:
                    throughput = float(match.group(1))
                    throughputs.append(throughput)
                except Exception as e:
                    print(f"[{label}] Warning: Failed to parse throughput on line {line_num}: {e}")
    return throughputs

def plot_comparison(data1, label1, data2, label2, output_image=None):
    if not data1 and not data2:
        print("No data found in both files.")
        return

    plt.figure(figsize=(10, 6))

    x1 = list(range(len(data1)))
    x2 = list(range(len(data2)))

    if data1:
        plt.plot(x1, data1, marker='o', linestyle='-', label=label1, alpha=0.8)
    if data2:
        plt.plot(x2, data2, marker='s', linestyle='--', label=label2, alpha=0.8)

    plt.title("Comparison of Avg Generation Throughput")
    plt.xlabel("Timeline")
    plt.ylabel("Throughput (tokens/s)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_image}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Compare Avg generation throughput from two log files (indexed by order)."
    )
    parser.add_argument("file1", help="First log file")
    parser.add_argument("file2", help="Second log file")
    parser.add_argument("--label1", default="File 1", help="Label for first file (default: File 1)")
    parser.add_argument("--label2", default="File 2", help="Label for second file (default: File 2)")
    parser.add_argument("--output", "-o", help="Optional output image file (e.g., comparison.png)")

    args = parser.parse_args()

    data1 = parse_log_file(args.file1, args.label1)
    data2 = parse_log_file(args.file2, args.label2)

    print(f"Parsed {len(data1)} throughput values from {args.file1}")
    print(f"Parsed {len(data2)} throughput values from {args.file2}")

    plot_comparison(data1, args.label1, data2, args.label2, args.output)

if __name__ == "__main__":
    main()
