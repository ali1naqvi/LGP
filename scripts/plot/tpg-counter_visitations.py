import sys

path = "experiments/ant_neuro_stateless/logs/misc/tpg.16.42.replay.std" 
def count_false_and_total(filename):
    false_count = 0
    total_count = 0
    try:
        with open(filename, "r") as file:
            for line in file:
                value = line.strip()
                if not value:
                    continue
                total_count += 1
                if value == "False":
                    false_count += 1
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)
    return false_count, total_count

if __name__ == "__main__":
    filename = path
    false_count, total_count = count_false_and_total(filename)
    print("Count of False:", false_count)
    print("Total lines counted:", total_count)
    print(total_count-false_count, "/", total_count, " = ", (total_count-false_count)/ total_count)