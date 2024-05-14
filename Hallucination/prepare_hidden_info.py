import pickle






if __name__ == "__main__":
    file_path = "datasets/answer_truthful_qa.bin"
    with open(file_path, "rb") as f:
        responses = pickle.load(f)

    # for idx, response in responses.items():
