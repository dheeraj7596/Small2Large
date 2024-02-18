import utils
import json
import pandas as pd
import numpy as np
import random
import argparse
import pickle

parser = argparse.ArgumentParser(
    description="Process perplexities csv files into LP(1) subsets."
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
    "--clustering",
    help="Pickle containing k-means clustering for dataset.",
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
optional.add_argument(
    "--lp1",
    help="If set, generates the subsets using the LP(1) metric.",
    action=argparse.BooleanOptionalAction,
    default=False,
)
optional.add_argument(
    "--lp1_approx",
    help="If set, generates the subsets using the LP(1) approx metric.",
    action=argparse.BooleanOptionalAction,
    default=False,
)
optional.add_argument(
    "--clust_rand",
    help="If set, generates the subsets using clust_rand baseline as described in paper.",
    action=argparse.BooleanOptionalAction,
    default=False,
)
optional.add_argument(
    "--model_name",
    help="Optional model name which is added to filenames of generated subsets.",
    default="model",
)
args = parser.parse_args()

if not args.lp1 and not args.lp1_approx and not args.clust_rand:
    print("Please choose at least 1 metric to generate subsets for!")
    exit()

random.seed(42)
list_data_dict = utils.jload(args.dataset)
cluster = pickle.load(open(args.clustering, "rb"))

MODEL_NAME = args.model_name

full_pre = pickle.load(open(args.epoch0, "rb"))
full_1 = pickle.load(open(args.epoch1, "rb"))
full_2 = pickle.load(open(args.epoch2, "rb"))
full_3 = pickle.load(open(args.epoch3, "rb"))


print("Calculating LP(1) and LP(1) Approx...\n")
data = {
    "index": list(range(len(list_data_dict))),
    "sample": [
        (
            f"Instruction: {s['instruction']}\nInput: {s['input']}\nResponse: {s['output']}"
            if len(s["input"]) != 0
            else f"Instruction: {s['instruction']}\nResponse: {s['output']}"
        )
        for s in list_data_dict
    ],
    "len_response": [len(s["output"]) for s in list_data_dict],
    "cluster_num": [cluster[i]["cluster_num"] for i in range(len(list_data_dict))],
    "P0": full_pre,
    "P1": full_1,
    "P2": full_2,
    "P3": full_3,
    "LP1": [
        (
            (full_pre[i] - full_1[i]) / (full_pre[i] - full_3[i])
            if full_pre[i] - full_3[i] != 0
            else 0
        )
        for i in range(len(list_data_dict))
    ],
}
df = pd.DataFrame.from_dict(data)
# Calculate proxy
df["LP1_Approx"] = (df["P0"] - df["P1"]) / (df["P0"])
print(df)
print()


def indexes_to_dataset(index_list):
    dataset = []
    for index in index_list:
        dataset.append(list_data_dict[index])
    return json.dumps(dataset, indent=4)


low_l1_33 = []
mid_l1_33 = []
high_l1_33 = []

low_l1_25 = []
low_l1_10 = []
low_l1_5 = []
low_l1_3 = []
low_l1_1 = []

low_proxy_33 = []
mid_proxy_33 = []
high_proxy_33 = []

low_proxy_25 = []
low_proxy_10 = []
low_proxy_5 = []
low_proxy_3 = []
low_proxy_1 = []

clust_rand_33 = []
clust_rand_25 = []
clust_rand_10 = []
clust_rand_5 = []
clust_rand_3 = []
clust_rand_1 = []

for cluster in range(df["cluster_num"].max() + 1):
    filter_clust = df[df["cluster_num"] == cluster]
    sort_L1 = filter_clust.sort_values(by=["LP1"]).reset_index(drop=True)
    sort_proxy = filter_clust.sort_values(by=["LP1_Approx"]).reset_index(drop=True)
    indexes_L1 = sort_L1["index"].tolist()
    indexes_proxy = sort_proxy["index"].tolist()

    low_l1_33.extend(np.array_split(indexes_L1, 3)[0])
    mid_l1_33.extend(np.array_split(indexes_L1, 3)[1])
    high_l1_33.extend(np.array_split(indexes_L1, 3)[2])

    low_l1_25.extend(np.array_split(indexes_L1, 4)[0])
    low_l1_10.extend(np.array_split(indexes_L1, 10)[0])
    low_l1_5.extend(np.array_split(indexes_L1, 20)[0])
    low_l1_3.extend(np.array_split(indexes_L1, 33)[0])
    low_l1_1.extend(np.array_split(indexes_L1, 100)[0])

    low_proxy_33.extend(np.array_split(indexes_proxy, 3)[0])
    mid_proxy_33.extend(np.array_split(indexes_proxy, 3)[1])
    high_proxy_33.extend(np.array_split(indexes_proxy, 3)[2])

    low_proxy_25.extend(np.array_split(indexes_proxy, 4)[0])
    low_proxy_10.extend(np.array_split(indexes_proxy, 10)[0])
    low_proxy_5.extend(np.array_split(indexes_proxy, 20)[0])
    low_proxy_3.extend(np.array_split(indexes_proxy, 33)[0])
    low_proxy_1.extend(np.array_split(indexes_proxy, 100)[0])

    clust_rand_33.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 3)[0]))
    )
    clust_rand_25.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 4)[0]))
    )
    clust_rand_10.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 10)[0]))
    )
    clust_rand_5.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 20)[0]))
    )
    clust_rand_3.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 33)[0]))
    )
    clust_rand_1.extend(
        random.sample(indexes_L1, len(np.array_split(indexes_L1, 100)[0]))
    )


if args.lp1:
    with open(f"./data/33_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_33))
    with open(f"./data/33_mid_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_mid_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(mid_l1_33))
    with open(f"./data/33_high_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_high_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(high_l1_33))
    with open(f"./data/25_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/25_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_25))
    with open(f"./data/10_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/10_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_10))
    with open(f"./data/5_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/5_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_5))
    with open(f"./data/3_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/3_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_3))
    with open(f"./data/1_low_lp1-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/1_low_lp1-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_l1_1))

if args.lp1_approx:
    with open(f"./data/33_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_33))
    with open(f"./data/33_mid_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_mid_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(mid_proxy_33))
    with open(f"./data/33_high_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_high_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(high_proxy_33))
    with open(f"./data/25_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/25_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_25))
    with open(f"./data/10_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/10_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_10))
    with open(f"./data/5_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/5_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_5))
    with open(f"./data/3_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/3_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_3))
    with open(f"./data/1_low_lp1_approx-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/1_low_lp1_approx-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(low_proxy_1))

if args.clust_rand:
    with open(f"./data/33_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/33_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_33))
    with open(f"./data/25_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/25_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_25))
    with open(f"./data/10_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/10_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_10))
    with open(f"./data/5_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/5_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_5))
    with open(f"./data/3_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/3_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_3))
    with open(f"./data/1_clust_rand-{MODEL_NAME}.json", "w") as outfile:
        print(f"Generated ./data/1_clust_rand-{MODEL_NAME}.json")
        outfile.write(indexes_to_dataset(clust_rand_1))
