import pickle
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import utils
import argparse

parser = argparse.ArgumentParser(
    description="Cluster dataset for diversity using k-means on sentence embeddings."
)
parser._action_groups.pop()
required = parser.add_argument_group("Required Arguments")
optional = parser.add_argument_group("Optional Arguments")
required.add_argument(
    "--dataset",
    help="The dataset for which to cluster.",
    required=True,
)
required.add_argument(
    "--num_clusters",
    help="The number of clusters to split the dataset into.",
    required=True,
    type=int,
)
optional.add_argument(
    "--output_name",
    help="Optional filename for the clustering pickle.",
    default="dataset",
)
args = parser.parse_args()

dataset = utils.jload(args.dataset)
embed_sentences = [
    f'Instruction: {(sample["instruction"].strip() + " " + sample["input"].strip()).strip()} Response: {sample["output"].strip()}'
    for sample in dataset
]

print("Generating embeddings, this step might take a while, please wait...")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256
embeddings = model.encode(embed_sentences)

print("Embeddings generated, starting clustering...")

kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init="auto", verbose=1).fit(embeddings)
output = [{'embed_sentence': s, 'cluster_num': l} for s,l in zip(embed_sentences, kmeans.labels_)]

print(output[9])
print("Clustering finished!")

with open(f'clustering/{args.output_name}-clustering.pkl', "wb") as fOut:
    pickle.dump(output, fOut, protocol=pickle.HIGHEST_PROTOCOL)
