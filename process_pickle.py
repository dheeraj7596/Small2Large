import pickle
import csv
import utils
import argparse

parser = argparse.ArgumentParser(
    description="Process dumped perplexities pickle files into csv format."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")
required.add_argument(
    "--dataset",
    help="The dataset for which the perplexities were dumped.",
    required=True,
)
required.add_argument(
    "--epoch0", help="Pickle containing perplexities before training.", required=True
)
required.add_argument(
    "--epoch1", help="Pickle containing perplexities after epoch 1.", required=True
)
required.add_argument(
    "--epoch2", help="Pickle containing perplexities after epoch 2.", required=True
)
required.add_argument(
    "--epoch3", help="Pickle containing perplexities after epoch 3.", required=True
)
required.add_argument(
    "--clustering",
    help="Pickle containing k-means clustering for dataset.",
    required=True,
)
optional.add_argument(
    "--model_name",
    help="Optional model name which is added to filename of generated csv.",
    default="model",
)
args = parser.parse_args()

list_data_dict = utils.jload(args.dataset)

MODEL_NAME = args.model_name
HEADER = [
    "index",
    "sample",
    "len_response",
    "labels_1000",
    None,
    f"{MODEL_NAME}_P0",
    f"{MODEL_NAME}_P1",
    f"{MODEL_NAME}_P2",
    f"{MODEL_NAME}_P3",
    None,
    f"{MODEL_NAME}_L1",
]

full_pre = pickle.load(open(args.epoch0, "rb"))
full_1 = pickle.load(open(args.epoch1, "rb"))
full_2 = pickle.load(open(args.epoch2, "rb"))
full_3 = pickle.load(open(args.epoch3, "rb"))

cluster = pickle.load(open(args.clustering, "rb"))

with open(f"./l1-rankings/{MODEL_NAME}-l1_ranking.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    for i, p in enumerate(list_data_dict):
        if len(p["input"]) != 0:
            sample = f"Instruction: {p['instruction']}\nInput: {p['input']}\nResponse: {p['output']}"
        else:
            sample = f"Instruction: {p['instruction']}\nResponse: {p['output']}"
        l1 = (
            (full_pre[i] - full_1[i]) / (full_pre[i] - full_3[i])
            if full_pre[i] - full_3[i] != 0
            else 0
        )
        row = [
            i,
            sample,
            len(p["output"]),
            cluster[i]["labels_300"],
            None,
            full_pre[i],
            full_1[i],
            full_2[i],
            full_3[i],
            None,
            l1,
        ]
        writer.writerow(row)
