from genericpath import isfile
import os
import re
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

import yaml

from utils.shape_util import read_file


def get_partial_corres_yx(
    vert_x,
    face_x,
    bary_coords_yx,
    vert_x_partial,
    face_x_partial,
    partial2full_x,
    partial2full_y,
):
    corres_yx_partial = bary_coords_yx[partial2full_y]
    used_faces_shape_x_yx = corres_yx_partial[:, 0].astype(int)
    used_verts_shape_x_yx = np.array(face_x[used_faces_shape_x_yx])
    found_faces_mask_yx = np.any(np.isin(used_verts_shape_x_yx, partial2full_x), axis=1)
    corres_yx_partial[~found_faces_mask_yx, :] = -1
    overlapping_corres_yx = corres_yx_partial[found_faces_mask_yx]
    overlapping_corres_yx_faces = overlapping_corres_yx[:, 0].astype(int)
    overlapping_corres_yx_verts = np.array(vert_x[face_x[overlapping_corres_yx_faces]])
    verts_faces_yx_partial = np.array(vert_x_partial[face_x_partial])
    corresponding_faces = np.zeros((overlapping_corres_yx.shape[0]))
    for i, face in enumerate(overlapping_corres_yx):
        corresponding_faces[i] = np.sum(
            np.linalg.norm(
                overlapping_corres_yx_verts[i, :] - verts_faces_yx_partial, axis=1
            ),
            axis=1,
        ).argmin()
    corres_yx_partial[found_faces_mask_yx, 0] = corresponding_faces
    return corres_yx_partial, found_faces_mask_yx


def get_filename_bary_corres(filename: str, dataset: str = ""):
    if filename.find("/") != -1 and dataset == "dt4d_animal" or dataset == "dt4d_human":
        first_folder = filename.split("/")[0]
        pose = filename.split("/")[1]
        filename = f"{first_folder}_{pose}"
    elif filename.find("/") != -1:
        filename = filename.split("/")[1]
    if filename.find("_human") != -1 or filename.find("_animal") != -1:
        filename = filename.split("_")[0]
    return filename


def read_corres_different_dataset(
    source_name, target_name, source_dataset, target_dataset
):
    corres_folder = f"{os.path.dirname(os.path.realpath(__file__))}/../precomputed_corr_templates/"
    # split source and target name
    source_name = get_filename_bary_corres(source_name, source_dataset)
    target_name = get_filename_bary_corres(target_name, target_dataset)
    source_dataset = get_filename_bary_corres(source_dataset)
    target_dataset = get_filename_bary_corres(target_dataset)
    regex_filename = (
        f"{source_dataset}_{source_name}*_{target_dataset}_{target_name}*_corres.txt"
    )
    # get all files that match the regex
    generator_regex = Path(corres_folder).glob(regex_filename)
    # to list
    list_files = [f for f in generator_regex]
    if len(list_files) == 0:
        raise Exception(
            f"Correspondence file not found for {source_name} and {target_name}"
        )
    filename = list_files[0].name
    bary_corres = np.loadtxt(corres_folder + filename)
    return bary_corres


def read_corres_same_dataset(general_config: dict, source: dict, target: dict, data_dir: str):
    conf = Path(
        f"{os.path.dirname(os.path.realpath(__file__))}/../config/{source['org_dataset']}.yaml"
    )
    with open(conf, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    save = general_config["save_shape_precomputation"]
    use_precomputed = general_config["use_shape_precomputation"]
    same_tr = config["same_triangulation"]
    gt_corr_type = config["gt_corr_type"]
    if same_tr and gt_corr_type == "p2p":
        try:
            nr_verts = config["nr_verts"]
        except KeyError:
            source_name = source["name"]
            precomputed_shape = f"{source_name}_V.npy"
            precomputed_shape = precomputed_shape.replace("/", "_")
            precomputed_folder = f"{os.path.dirname(os.path.realpath(__file__))}/precomputed_shapes"
            if isfile(f"{precomputed_folder}/{precomputed_shape}") and use_precomputed:
                V = np.load(f"{precomputed_folder}/{precomputed_shape}", allow_pickle=True)
            else:
                foldername = config["folder_name"]
                shape_filetype = config["file_type"]
                shape_file = f"{data_dir}/{foldername}/{source_name}.{shape_filetype}"
                V, _ = read_file(shape_file)
                if save:
                    np.save(f"{precomputed_folder}/{precomputed_shape}", V)
            nr_verts = V.shape[0]
        corres = np.arange(nr_verts)
    return corres


def get_corr_for_path(config: dict, path: List[dict], data_dir: str):
    use_precomputed = config["use_corres_precomputation"]
    save_precomputed = config["save_corres_precomputation"]
    corr_path = []
    for i in range(len(path) - 1):
        source = path[i]
        target = path[i + 1]
        precomputed_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "precomputed_corres")
        precomputed_source_name = source['name'].replace("/", "_")
        precomputed_target_name = target['name'].replace("/", "_")
        precomputed_name = f"{precomputed_folder}/{precomputed_source_name}_{precomputed_target_name}_corres.npz"
        if isfile(precomputed_name) and use_precomputed:
            corres_dict = np.load(precomputed_name)
            corres_src_target = {"corres": corres_dict["corres"], "same_tr": corres_dict["same_tr"].item(), "barycentric": corres_dict["barycentric"].item()}
        else:
            corres_src_target = get_corres_between_src_target(config, data_dir, corr_path, source, target)
            if save_precomputed: 
                np.savez(precomputed_name, corres=corres_src_target["corres"], same_tr=corres_src_target["same_tr"], barycentric=corres_src_target["barycentric"])
        corr_path.append(corres_src_target)
    return corr_path

def get_corres_between_src_target(general_config, data_dir, corr_path, source, target):
    corres_between_src_target = None
    source_name = source["name"]
    target_name = target["name"]
    source_dataset = source["org_dataset"]
    target_dataset = target["org_dataset"]
    conf_file = Path(
            f"{os.path.dirname(os.path.realpath(__file__))}/../config/{source_dataset}.yaml"
        )
    with open(conf_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    corres_given = config["corr_given"]
    if source_dataset == target_dataset and corres_given:
        if source_dataset == "dt4d_animal" or source_dataset == "dt4d_human":
            cat_source = source_name.split("/")[0]
            cat_target = target_name.split("/")[0]
            if cat_source == cat_target:
                corres = read_corres_same_dataset(general_config, source, target, data_dir)
                corres_between_src_target = {"corres": corres, "same_tr": True, "barycentric": False}
            else:
                found_same_tr = False
                cat_source = source_name.split("/")[0]
                cat_target = target_name.split("/")[0]
                    # load same_tr file
                same_tr = f"{os.path.dirname(os.path.realpath(__file__))}/../precomputed_corr_templates/same_triangulation_dt4d_animals.txt"
                with open(same_tr, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    (cat_same_tr_1, cat_same_tr_2) = line.split(",")
                    cat_same_tr_1 = cat_same_tr_1.split("_")[1]
                    cat_same_tr_2 = cat_same_tr_2.split("_")[1]
                    if (
                            cat_source == cat_same_tr_1
                            and cat_target == cat_same_tr_2
                            or cat_source == cat_same_tr_2
                            and cat_target == cat_same_tr_1
                        ):
                        corres = read_corres_same_dataset(general_config, source, target, data_dir)
                            # here only p2p
                        if (
                                "catBG" in cat_source and "leopardSLM" in cat_target
                            ) or ("catBG" in cat_target and "leopardSLM" in cat_source):
                            same_tr = False
                        else:
                            same_tr = True
                        corres_between_src_target = {
                                    "corres": corres,
                                    "same_tr": same_tr,
                                    "barycentric": False,
                                }
                            
                        found_same_tr = True
                        break
                if not found_same_tr:
                    corres = read_corres_different_dataset(
                            source_name, target_name, source_dataset, target_dataset
                        )
                    corres_between_src_target = {"corres": corres, "same_tr": False, "barycentric": True}
        elif source_dataset == "tosca":
                # remove number from string
            cat_source = re.sub(r"\d", "", source_name)
            cat_target = re.sub(r"\d", "", target_name)
            if cat_source == cat_target:
                corres = read_corres_same_dataset(general_config, source, target, data_dir)
                corres_between_src_target = {"corres": corres, "same_tr": True, "barycentric": False}
            else:
                corres = read_corres_different_dataset(
                        source_name, target_name, source_dataset, target_dataset
                    )
                corres_between_src_target = {"corres": corres, "same_tr": False, "barycentric": True}
        else:
            corres = read_corres_same_dataset(general_config, source, target, data_dir)
            corres_between_src_target = {
                        "corres": corres,
                        "same_tr": config["same_triangulation"],
                        "barycentric": False,
                    }
    else:
            # read
        bary_corres = read_corres_different_dataset(
                source_name, target_name, source_dataset, target_dataset
            )
        corres_between_src_target = {"corres": bary_corres, "same_tr": False, "barycentric": True}
    return corres_between_src_target


def check_partial(corres: np.array):
    if np.any(corres == -1):
        return True
    return False


def barycentric_corres_to_p2p(config_general, corres, i, path, data_dir):
    shape_name = path[i]["name"]
    precomputed_shape = f"{shape_name}_F.npy"
    precomputed_shape = precomputed_shape.replace("/", "_")
    precomputed_folder = f"{os.path.dirname(os.path.realpath(__file__))}/precomputed_shapes"
    if isfile(f"{precomputed_folder}/{precomputed_shape}"):
        F = np.load(f"{precomputed_folder}/{precomputed_shape}", allow_pickle=True)
        corres = barycentric_to_p2p(corres, F)
    else:
        conf_file = Path(
        f"{os.path.dirname(os.path.realpath(__file__))}/../config/{path[i]['org_dataset']}.yaml"
        )
        with open(conf_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        save = config_general["save_shape_precomputation"]
        shape_dataset = config["folder_name"]
        shape_filetype = config["file_type"]
        shape_file = f"{data_dir}/{shape_dataset}/{shape_name}.{shape_filetype}"
        try:
            _ , F = read_file(shape_file)
            if save:
                np.save(f"{precomputed_folder}/{precomputed_shape}", F)
            corres = barycentric_to_p2p(corres, F)
        except:
            saved_filename = f"{os.path.dirname(os.path.realpath(__file__))}/default_vertices/{shape_name}.npy"
            if isfile(saved_filename):
                v_default = np.load(saved_filename, allow_pickle=True)
                corres = v_default[corres[:,0].astype(int)]
    return corres


def combine_corres_paths(config_general: dict, corres_path: List[dict], path: List[dict], data_dir: str):
    start_corres = corres_path[0]["corres"]
    corres = start_corres
    last_barycentric = corres_path[0]["barycentric"]
    for i in range(1, len(corres_path)):
        new_corres = corres_path[i]["corres"]
        same_tr = corres_path[i]["same_tr"]
        barycentric = corres_path[i]["barycentric"]
        if same_tr:
            continue
        elif barycentric and not last_barycentric:
            old_corres = corres
            corres = new_corres[corres.astype(int), :]
            corres[old_corres == -1, :] = -1
            last_barycentric = True
        elif last_barycentric and barycentric:
            # load shape to get faces
            corres = barycentric_corres_to_p2p(config_general, corres, i, path, data_dir)
            old_corres = corres
            corres = new_corres[old_corres, :]
            corres[old_corres == -1] = -1
        elif last_barycentric and not barycentric and not same_tr:
            corres = barycentric_corres_to_p2p(config_general, corres, i, path, data_dir)
            old_corres = corres
            corres = new_corres[corres.astype(int)]
            corres[old_corres == -1] = -1
            last_barycentric = False
        elif not last_barycentric and not barycentric:
            old_corres = corres
            corres = new_corres[corres.astype(int)]
            corres[old_corres == -1] = -1
        else:
            print("Error in combining correspondences")
    return corres

def barycentric_to_p2p(corres, faces2):
    # get barycentric coordinates
    bary_coords = corres[:, 1:]
    # get closest face
    closest_faces = corres[:, 0].astype(int)
    p2p = np.zeros_like(closest_faces)
    p2p = np.where(closest_faces != -1, faces2[closest_faces, np.argmax(bary_coords, axis=1)], -1)

    return p2p


def p2p_to_barycentric_batch(p2p_corres, faces):
    bary_coords = np.zeros((p2p_corres.shape[0], 4))
    p2p_corres = p2p_corres.astype(np.int32)
    faces = faces.astype(np.int32)

    p2p_faces = get_p2p_faces_for_corres(p2p_corres, faces)
    # find for every row in p2p_corres the first column that is true
    faces_idx = np.argmax(p2p_faces, axis=1)
    bary_coords[:, 0] = faces_idx
    # get idx in every face that belongs to p2p_corres
    idx = np.where(faces[faces_idx] == p2p_corres[:, None])[1]
    rows = np.arange(p2p_corres.shape[0])
    bary_coords[rows, idx + 1] = 1
    return bary_coords


def get_p2p_faces_for_corres(p2p_corres, faces):
    p2p_faces = (
        ((faces[None, :] == p2p_corres[:, None, None])[:, :, 0])
        | ((faces[None, :] == p2p_corres[:, None, None])[:, :, 1])
        | ((faces[None, :] == p2p_corres[:, None, None])[:, :, 2])
    )

    return p2p_faces


def p2p_to_barycentric(p2p_corres, faces):

    batch_size = len(p2p_corres) + 1
    num_samples = p2p_corres.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    bary_coords = np.zeros((p2p_corres.shape[0], 4))

    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_samples)
        not_partial = p2p_corres[start:end] != -1
        p2p_corres_batch = p2p_corres[start:end][not_partial]
        bary_coords_batch = p2p_to_barycentric_batch(p2p_corres_batch, faces)
        bary_coords[start:end][not_partial] = bary_coords_batch
        bary_coords[start:end][~not_partial] = [-1, 0, 0, 0]

    # assert that there is no row with all zeros
    assert np.any(bary_coords != 0, axis=1).all()
    return bary_coords


def get_bary_and_p2p_corres_from_path(corres, face_y):
    if len(corres.shape) == 2:
        bary_coord = corres
        p2p_corres = barycentric_to_p2p(corres, face_y)
    else:
        p2p_corres = corres
        bary_coord = p2p_to_barycentric(corres.astype(int), face_y)
    return bary_coord, p2p_corres
