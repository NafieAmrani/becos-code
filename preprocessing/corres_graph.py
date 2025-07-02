import os
import re
import csv
import yaml

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List

###################
# Module-level cache for dataset metadata
_DATASET_METADATA_CACHE = {}

def get_dataset_metadata(dataset_name):
    global _DATASET_METADATA_CACHE
    if dataset_name in _DATASET_METADATA_CACHE:
        return _DATASET_METADATA_CACHE[dataset_name]
    config_dir = f"{os.path.dirname(os.path.realpath(__file__))}/../config/"
    yaml_path = os.path.join(config_dir, f"{dataset_name}.yaml")
    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)
    general_yaml_path = os.path.join(config_dir, f"general_config.yaml")
    with open(general_yaml_path, "r") as f:
        general_meta = yaml.safe_load(f)
    # Only keep relevant fields
    metadata = {
        "data_dir": general_meta.get("data_dir", None),  # fallback to None if not present
        "folder_name": meta["folder_name"],
        "file_type": meta["file_type"],
        "category": meta["category"],
        "scale": meta.get("scale", 1),
        "rot": meta.get("rot", [0, 0, 0]),
    }
    _DATASET_METADATA_CACHE[dataset_name] = metadata
    return metadata

def build_adjacency_graph(plot_graph=False) -> nx.Graph:
    csv_file = Path(
        f"{os.path.dirname(os.path.realpath(__file__))}/../config/adjacency_matrix.csv"
    )

    # Create a new graph
    G = nx.Graph()

    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        nodes = next(reader)[1:]

        # add nodes
        for node in nodes:
            G.add_node(node)

        # add edges
        for row in reader:
            source = row[0]
            for idx, val in enumerate(row[1:]):
                if val != "0":
                    target = nodes[idx]
                    G.add_edge(source, target)

    if plot_graph:
        fig = plt.figure(1, figsize=(30, 30))
        nx.draw_kamada_kawai(G, with_labels=True, font_size=12)
        plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/../config/adjacency_graph.png")

    return G


def get_template_name(templates, query):
    try:
        return templates[templates[:, 0] == query, 1][0]
    except:
        raise Exception(f"{query} was not found in templates")


def get_node(shape: dict, is_target: bool, templates: np.array) -> List[dict]:
    path = [shape]
    node = None
    template_name = None

    shape_name = shape["name"]
    shape_dataset = shape["org_dataset"]

    if shape_dataset in ["faust", "scape", "kids"]:
        template_name = get_template_name(templates, shape_dataset)
        node = f"{shape_dataset}_{template_name}"

    elif shape_dataset in ["tosca"]:
        matches = re.search(r"^(.*?)(\d+)$", shape_name)
        shape_org_name = matches.group(1)
        template_name = get_template_name(templates, f"tosca_{shape_org_name}")
        node = f"tosca_{template_name}"

    elif shape_dataset == "shrec20":
        template_name = shape_name
        node = f"{shape_dataset}_{shape_name}"

    elif shape_dataset == "smal":
        split_name = shape_name.split("/")
        template_name = get_template_name(templates, f"smal_{split_name[0]}")
        node = f"smal_{template_name.replace('/', '_')}"

    elif shape_dataset == "dt4d_human" or shape_dataset == "dt4d_animal":
        split_name = shape_name.split("/")
        dt4d_type = shape_dataset.split("_")[1]
        template_name = templates[templates[:, 0] == f"dt4d_{dt4d_type}_{split_name[0]}", 1][0]
        node = f"dt4d_{dt4d_type}_{split_name[0]}"

    if template_name != shape_name:
        intermediate_shape = {
            "name": template_name,
            "org_dataset": shape['org_dataset'],
            "category": shape['category']}
        path += [intermediate_shape]

    if is_target:
        path.reverse()

    assert path and node, f"Either Path:{path}, or Node:{node} is None"
    return path, node

def template_to_shape(target: str, templates):
    result = target.split("_")
    if result[0] == "dt4d":
        dt4d_type = result[1]
        template_name = templates[
            templates[:, 0] == f"dt4d_{dt4d_type}_{result[2]}", 1
        ][0]
        target = {
            "name": f"{template_name}",
            "org_dataset": f"dt4d_{dt4d_type}",
            "category": "four-legged" if dt4d_type == "animal" else "human",
        }
    elif result[0] == "tosca":
        if result[1] in ["victoria0", "david0", "michael0"]:
            target = {
                "name": f"{result[1]}",
                "org_dataset": "tosca",
                "category": "human",
            }
        elif result[1] == "centaur":
            target = {
                "name": f"{result[1]}",
                "org_dataset": "tosca",
                "category": "centaur",
            }
        else:
            target = {
                "name": f"{result[1]}",
                "org_dataset": "tosca",
                "category": "four-legged",
            }
    elif result[0] == "shrec20":
        target = {
            "name": f"{result[1]}_{result[2]}" if len(result) == 3 else f"{result[1]}",
            "org_dataset": "shrec20",
            "category": "four-legged",
        }
    elif result[0] == "faust":
        target = {
            "name": f"{result[1]}_{result[2]}_{result[3]}",
            "org_dataset": "faust",
            "category": "human",
        }
    return target

def get_path_to_closest_annotated_template(config_folder: str, G: nx.Graph, source: str, template_file: str,
                                 annotated_template_file: str):
    templates = np.genfromtxt(
        os.path.join(config_folder, template_file),
        delimiter=",",
        dtype=str,
    )

    annotated_templates = np.genfromtxt(
        os.path.join(config_folder, annotated_template_file),
        delimiter=",",
        dtype=str,
    )

    _, source_node = get_node(source, is_target=False, templates=templates)

    annotated_nodes = [
        get_node({"org_dataset": "_".join(x[0].split("_")[:-1]) if len(x[0].split("_")) > 1 else x[0], "name": x[1], "category": None}, is_target=True, templates=annotated_templates)[1] for x in
        annotated_templates[1:]]

    path_dict = nx.shortest_path(G, source=source_node)
    dist = {k: len(path_dict[k]) for k in annotated_nodes}
    closest_annotated_template = min(dist, key=dist.get)

    target = template_to_shape(closest_annotated_template, annotated_templates)
    full_shape_dataset_object = get_dataset_metadata(target["org_dataset"])
    target["file_path"] = f"{full_shape_dataset_object['data_dir']}/{full_shape_dataset_object['folder_name']}/{target['name']}.{full_shape_dataset_object['file_type']}"
    return target, get_shortest_path(config_folder, G, source, target, template_file)


def get_shortest_path(config_folder: str, G: nx.Graph, source: str, target: str, template_file: str):
    templates = np.genfromtxt(
        os.path.join(config_folder, template_file),
        delimiter=",",
        dtype=str,
    )

    if (
            source["org_dataset"] == target["org_dataset"]
    ):
        if source["org_dataset"] == "dt4d_human" or source["org_dataset"] == "dt4d_animal":
            if source["name"].split("/")[0] == target["name"].split("/")[0]:
                dt4d_type = source["org_dataset"].split("_")[1]
                class_name = source["name"].split("/")[0]
                template_name = templates[templates[:, 0] == f"dt4d_{dt4d_type}_{class_name}", 1][0]
                if source["name"].split("/")[1] == template_name or target["name"].split("/")[1] == template_name:
                    return [source, target]
                return [source, {"name": f"{template_name}",
                                 "org_dataset": f"{source['org_dataset']}",
                                 "category": "four-legged" if dt4d_type == "animal" else "human"},
                        target]
        elif source["category"] == target["category"] == "human":
            return [source, target]

    path_start, source_node = get_node(source, is_target=False, templates=templates)
    path_end, target_node = get_node(target, is_target=True, templates=templates)

    shortest_path = []

    nodes_list = nx.shortest_path(G, source=source_node, target=target_node)

    for node in nodes_list[1:-1]:
        shortest_path += [template_to_shape(node, templates)]

    assert len(nodes_list[1:-1]) == len(
        shortest_path
    ), f"length on graph is {len(nodes_list)} and length of shortest path is {len(shortest_path)}"
    return path_start + shortest_path + path_end
