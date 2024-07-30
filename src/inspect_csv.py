import pandas as pd

def inspect_csv(file_path):
    data = pd.read_csv(file_path)
    print(data.head())

if __name__ == "__main__":
    csv_path = "/Users/nithinrajulapati/Downloads/PROJECT 1/data/classes_in_imagenet.csv"
    inspect_csv(csv_path)
