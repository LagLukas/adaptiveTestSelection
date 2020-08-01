import json
import os


def load(file_path):
    with open(file_path) as json_file:
        return json.load(json_file)


def load_and_clean(file_path="runtimes" + os.sep + "TestHist_51.json"):
    data = load(file_path)
    data = data["test_results"]
    data = list(map(lambda x: float(x[2]), data))
    data = list(filter(lambda x: x > 0, data))
    return data

if "__main__" == __name__:
    data = load_and_clean("runtimes" + os.sep + "TestHist_51.json")
    print(data)
    print(len(data))
