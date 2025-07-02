import os
import time
import pickle
from pathlib import Path

import numpy as np
import yaml
import open3d as o3d
from joblib import Parallel, delayed

from corres_comp.corres import get_partial_corres_yx
from corres_comp.corres import get_corr_for_path
from corres_comp.corres import combine_corres_paths
from corres_comp.corres import barycentric_to_p2p
from corres_comp.corres import get_bary_and_p2p_corres_from_path
from datasets.shape_dataset import build_datasets, get_splits
from preprocessing.corres_graph import build_adjacency_graph, get_shortest_path, get_path_to_closest_annotated_template
from preprocessing.remesh import adjust_path_remeshing, remeshing_wrapper
from ray_casting.ray_caster import RayCaster
from utils.misc import write_pickle
from utils.shape_util import (
    align_shapes,
    create_colormap,
    plot_shapes,
    read_file,
    write_off,
)


def save_corres_for_shape_pair(shape_x, shape_y, config, annotation_graph, data_dir, split="train", pair_id=0, split_folder=""):
    # Get path through the annotation graph
    path = get_shortest_path(
        config["config_dir"], annotation_graph, shape_x, shape_y, config["template_file"]
    )

    # get the correspondences between source and target shape
    corres_path = get_corr_for_path(config, path, data_dir)
    corres_path_yx = get_corr_for_path(config, path[::-1], data_dir)

    if config["propagate_annotations"]:
        template_annotated_0, path_annotated_0 = get_path_to_closest_annotated_template(
            config["config_dir"], annotation_graph, shape_x, config["template_file"], config["annotated_template_file"]
        )

        template_annotated_1, path_annotated_1 = get_path_to_closest_annotated_template(
            config["config_dir"], annotation_graph, shape_y, config["template_file"], config["annotated_template_file"]
        )

        corres_path_annotated_0 = get_corr_for_path(config, path_annotated_0, data_dir)
        corres_path_annotated_1 = get_corr_for_path(config, path_annotated_1, data_dir)

        annotated_folder = f"{os.path.dirname(os.path.realpath(__file__))}/../{config['annotated_templates_folder']}/"#precomputed_annotated_templates/"
        annotated_info_0 = np.loadtxt(annotated_folder + template_annotated_0["org_dataset"] + "_" + template_annotated_0["name"].replace("/", "_") + ".txt")
        annotated_info_1 = np.loadtxt(annotated_folder + template_annotated_1["org_dataset"] + "_" + template_annotated_1["name"].replace("/", "_") + ".txt")

        vert_0, face_0 = read_file(template_annotated_0["file_path"])
        vert_1, face_1 = read_file(template_annotated_1["file_path"])

    # if using precomputed remeshed shapes, don't load shapes
    if config['remesh']: 
        if config["use_precompute_remeshing"]:
            remeshed_shape_file_x = os.path.join(
                config["precomputed_remeshed"], f"remeshed_{shape_x['name'].replace('/', '_')}.npz"
            )
            remeshed_shape_file_y = os.path.join(
                config["precomputed_remeshed"], f"remeshed_{shape_y['name'].replace('/', '_')}.npz"
            )
            file_exists_x = os.path.exists(remeshed_shape_file_x)
            file_exists_y = os.path.exists(remeshed_shape_file_y)
        else:
            remeshed_shape_file_x = None
            remeshed_shape_file_y = None
            file_exists_x = False
            file_exists_y = False
    else:
        file_exists_x = False
        file_exists_y = False

    # if shapes is already cached -> don't load the original shape
    if file_exists_x:
        vert_x, face_x = None, None
    else:
        # read shapes
        vert_x, face_x = read_file(shape_x["file_path"])

    if file_exists_y:
        vert_y, face_y = None, None
    else:
        # read shapes
        vert_y, face_y = read_file(shape_y["file_path"])

    # remesh shape
    if config["remesh"]:
        (
            vert_x,
            face_x,
            bary_array_source_to_remeshed_x,
            bary_array_remeshed_to_source_x,
        ) = remeshing_wrapper(
            config,
            shape_x["name"],
            vert_x,
            face_x,
            face_count=shape_x["desired_n_face_remeshed"],
            remeshed_shape_file=remeshed_shape_file_x,
            file_exists=file_exists_x,
        )
        (
            vert_y,
            face_y,
            bary_array_source_to_remeshed_y,
            bary_array_remeshed_to_source_y,
        ) = remeshing_wrapper(
            config,
            shape_y["name"],
            vert_y,
            face_y,
            face_count=shape_y["desired_n_face_remeshed"],
            remeshed_shape_file=remeshed_shape_file_y,
            file_exists=file_exists_y,
        )

        # add bary_array_remeshed_to_source_x to corres_path
        org_path = path.copy()
        path, corres_path = adjust_path_remeshing(
            shape_x,
            shape_y,
            path,
            corres_path,
            bary_array_source_to_remeshed_x,
            bary_array_remeshed_to_source_y,
        )
        _, corres_path_yx = adjust_path_remeshing(
            shape_y,
            shape_x,
            org_path[::-1],
            corres_path_yx,
            bary_array_source_to_remeshed_y,
            bary_array_remeshed_to_source_x,
        )

        if config["propagate_annotations"]:
            identity_meshing = {"corres": np.arange(len(annotated_info_0)), "same_tr": False, "barycentric": False}
            path_annotated_0, corres_path_annotated_0 = adjust_path_remeshing(
                shape_x,
                template_annotated_0,
                path_annotated_0,
                corres_path_annotated_0,
                bary_array_source_to_remeshed_x,
                identity_meshing,
            )
            identity_meshing = {"corres": np.arange(len(annotated_info_1)), "same_tr": False, "barycentric": False}
            path_annotated_1, corres_path_annotated_1 = adjust_path_remeshing(
                shape_y,
                template_annotated_1,
                path_annotated_1,
                corres_path_annotated_1,
                bary_array_source_to_remeshed_y,
                identity_meshing,
            )

    # scale meshes to a reasonable size
    vert_x = vert_x * shape_x["scale"]
    vert_y = vert_y * shape_y["scale"]

    vert_x = vert_x - np.mean(vert_x, axis=0)
    vert_y = vert_y - np.mean(vert_y, axis=0)


    # rotate mesh to align with rest of the dataset
    # check if shape_x['rot'] is a dictionary
    if isinstance(shape_x["rot"], dict):
        shape_x_name = shape_x["name"].replace("/", "_")
        try:
            shape_x["rot"] = shape_x['rot'][shape_x_name]
        except KeyError:
            shape_x["rot"] = [0, 0, 0]

    if isinstance(shape_y["rot"], dict):
        shape_y_name = shape_y["name"].replace("/", "_")
        try:
            shape_y["rot"] = shape_y['rot'][shape_y_name]
        except KeyError:
            shape_y["rot"] = [0, 0, 0]

    R_x = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(shape_x["rot"]))
    R_y = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(shape_y["rot"]))
    vert_x = np.dot(vert_x, R_x)
    vert_y = np.dot(vert_y, R_y)

    corres = combine_corres_paths(config, corres_path, path, data_dir)
    corres_yx = combine_corres_paths(config, corres_path_yx, path[::-1], data_dir)


    if config["remesh"]:
        p2p_corres = corres
        p2p_corres_yx = corres_yx
    else:
        bary_coords_xy, p2p_corres = get_bary_and_p2p_corres_from_path(
            corres, face_y
        )
        bary_coords_yx, p2p_corres_yx = get_bary_and_p2p_corres_from_path(
            corres_yx, face_x
        )


    vert_x, vert_y, area_x, area_y, align_R, align_t = align_shapes(
        vert_x,
        face_x,
        vert_y,
        face_y,
        p2p_corres,
    )

    # make surface area equal to 1 for both shapes
    vert_x /= area_x
    vert_y /= area_y

    if config["propagate_annotations"]:
        corres_annotated_0 = combine_corres_paths(config, corres_path_annotated_0, path_annotated_0, data_dir)
        corres_annotated_1 = combine_corres_paths(config, corres_path_annotated_1, path_annotated_1, data_dir)

        # We're not doing interpolation for discrete labels, but if you want, you can do it here.
        if len(corres_annotated_0.shape) > 1:
            corres_annotated_0 = barycentric_to_p2p(corres_annotated_0, face_0)
        annotated_data_0 = annotated_info_0[corres_annotated_0]
        if len(corres_annotated_1.shape) > 1:
            corres_annotated_1 = barycentric_to_p2p(corres_annotated_1, face_1)
        annotated_data_1 = annotated_info_1[corres_annotated_1]

    if config["setting"] == "partial_partial":
        best_cam_pos_id = 0
        min_diff = 1
        for cam_pos_id in range(config['n_cam_pos']):
            # perform ray casting
            ray_caster = RayCaster()

            if config["use_precomputed_partial_raycasting"]:
                cam_pose = shape_x['cam_pos'][cam_pos_id]
                cam_pose_str = f"{cam_pose[0]:.3f}_{cam_pose[1]:.3f}_{cam_pose[2]:.3f}"
                curr_filedir = os.path.dirname(os.path.realpath(__file__))
                filename = (
                    f"{curr_filedir}/../precomputed_triangle_ids/partial_partial/{shape_x['name'].replace('/', '_')}_cam_{cam_pose_str}.npy"
                )

                assert os.path.exists(filename), f"File does not exist: {filename}"
                face_ids_x = np.load(filename)
                vert_x_partial, face_x_partial, partial2full_x, face_ids_x = (
                    ray_caster.get_partial_shape(vert_x, face_x, shape_x['cam_pos'][cam_pos_id], face_ids_x)
                )
            else:
                vert_x_partial, face_x_partial, partial2full_x, _ = ray_caster.get_partial_shape(
                    vert_x, face_x, shape_x['cam_pos'][cam_pos_id]
                )

            if config["use_precomputed_partial_raycasting"]:
                filename = (
                    f"{curr_filedir}/../precomputed_triangle_ids/partial_partial/{shape_y['name'].replace('/', '_')}_cam_{cam_pose_str}.npy"
                )
                assert os.path.exists(filename), f"File does not exist: {filename}"

                face_ids_y = np.load(filename)
                vert_y_partial, face_y_partial, partial2full_y, face_ids_y = (
                    ray_caster.get_partial_shape(vert_y, face_y, shape_y['cam_pos'][cam_pos_id], face_ids_y)
                )
            else:
                vert_y_partial, face_y_partial, partial2full_y, _ = ray_caster.get_partial_shape(
                    vert_y, face_y, shape_y['cam_pos'][cam_pos_id]
                )
            if not config["remesh"]:
                corres_yx_partial, found_faces_mask_yx = get_partial_corres_yx(
                    vert_x,
                    face_x,
                    bary_coords_yx,
                    vert_x_partial,
                    face_x_partial,
                    partial2full_x,
                    partial2full_y,
                )
                corres_xy_partial, found_faces_mask = get_partial_corres_yx(
                    vert_y,
                    face_y,
                    bary_coords_xy,
                    vert_y_partial,
                    face_y_partial,
                    partial2full_y,
                    partial2full_x,
                )
            else:
                corres_yx_partial_to_full = p2p_corres_yx[partial2full_y]
                corres_xy_partial_to_full = p2p_corres[partial2full_x]

                corres_yx_partial = -1 * np.ones(len(corres_yx_partial_to_full))
                corres_xy_partial = -1 * np.ones(len(corres_xy_partial_to_full))
                for i in range(len(corres_yx_partial_to_full)):
                    found_idx = np.where(corres_yx_partial_to_full[i]==partial2full_x)[0]
                    if len(found_idx) > 0:
                        corres_yx_partial[i] = found_idx[0]
                for i in range(len(corres_xy_partial_to_full)):
                    found_idx = np.where(corres_xy_partial_to_full[i]==partial2full_y)[0]
                    if len(found_idx) > 0:
                        corres_xy_partial[i] = found_idx[0]
                corres_xy_partial = corres_xy_partial.astype(int)
                corres_yx_partial = corres_yx_partial.astype(int)
                found_faces_mask = corres_xy_partial!=-1
                found_faces_mask_yx = corres_yx_partial!=-1

            overlap_xy = np.sum(found_faces_mask) / len(found_faces_mask)
            overlap_yx = np.sum(found_faces_mask_yx) / len(found_faces_mask_yx)

            # make sure that the overlap is bigger than the minimum overlap
            if (config["min_overlap"] <= overlap_xy <= config["max_overlap"]
                    and config["min_overlap"] <= overlap_yx <= config["max_overlap"]):
                # store the cam pos used
                shape_x_cam_pos = shape_x['cam_pos'][cam_pos_id]
                shape_y_cam_pos = shape_y['cam_pos'][cam_pos_id]
                break

            diff_xy = config["min_overlap"] - overlap_xy if overlap_xy < config["min_overlap"] else 0
            diff_yx = config["min_overlap"] - overlap_yx if overlap_yx < config["min_overlap"] else 0
            diff = diff_xy + diff_yx

            if diff < min_diff:
                best_cam_pos_id = cam_pos_id
                min_diff = diff

            if cam_pos_id == config['n_cam_pos'] - 1:
                # use the camer apose that is as close to the threshold as possible
                shape_x_cam_pos = shape_x['cam_pos'][best_cam_pos_id]
                shape_y_cam_pos = shape_y['cam_pos'][best_cam_pos_id]

        if config["store_vis"] or config["show_output"]:
            if not config["remesh"]:
                p2p_corres_yx = barycentric_to_p2p(
                    corres_yx_partial, face_x_partial
                )
            else:
                p2p_corres_yx = corres_yx_partial
            color_partial_x = create_colormap(vert_x_partial)
            unmatched_color = [0, 0, 0]
            full_shape_color = [0.9, 0.9, 0.9]
            color_partial_x[found_faces_mask == 0] = unmatched_color
            color_partial_y = color_partial_x[p2p_corres_yx]
            color_partial_y[p2p_corres_yx == -1] = unmatched_color
            color_x = full_shape_color * np.ones_like(vert_x)
            color_y = full_shape_color * np.ones_like(vert_y)

        corres_xy_2_save = corres_xy_partial
        corres_yx_2_save = corres_yx_partial

    if config["setting"] == "partial_full":
        # take the first cam pos (no overlap in this setting)
        shape_x_cam_pos = None
        shape_y_cam_pos = shape_y['cam_pos'][0]
        ray_caster = RayCaster()

        if config["use_precomputed_partial_raycasting"]:
            curr_filedir = os.path.dirname(os.path.realpath(__file__))
            cam_pose = shape_y['cam_pos'][0]
            cam_pose_str = f"{cam_pose[0]:.3f}_{cam_pose[1]:.3f}_{cam_pose[2]:.3f}"
            filename = (
                f"{curr_filedir}/../precomputed_triangle_ids/partial_full/{shape_y['name'].replace('/', '_')}_cam_{cam_pose_str}.npy")
            assert os.path.exists(filename), f"File does not exist: {filename}"
            face_ids_y = np.load(filename)
            vert_y_partial, face_y_partial, partial2full_y, face_ids_y = ray_caster.get_partial_shape(vert_y, face_y, shape_y['cam_pos'][0], face_ids_y)
        else:
            vert_y_partial, face_y_partial, partial2full_y, _ = ray_caster.get_partial_shape(
                vert_y, face_y, shape_y['cam_pos'][0]
            )

        if not config["remesh"]:
            bary_coords_yx_partial = bary_coords_yx[partial2full_y]
        else:
            p2p_corres_yx_partial = p2p_corres_yx[partial2full_y]
        corres_xy = p2p_corres
        corres_xy_partial = -1 * np.ones(len(corres_xy))
        for i in range(len(corres_xy_partial)):
            found_idx = np.where(corres_xy[i]==partial2full_y)[0]
            if len(found_idx) > 0:
                corres_xy_partial[i] = found_idx[0]
        found_faces_mask = corres_xy_partial != -1
        vert_y_partial = vert_y[partial2full_y]

        # dummy variables
        vert_x_partial, face_x_partial, partial2full_x = None, None, None

        # final correspondences to save
        corres_xy_2_save = None
        if not config["remesh"]:
            corres_yx_2_save = bary_coords_yx_partial
        else:
            corres_yx_2_save = p2p_corres_yx_partial

    if config["setting"] == "full_full":
        shape_x_cam_pos, shape_y_cam_pos = None, None
        vert_x_partial, face_x_partial, partial2full_x = None, None, None
        vert_y_partial, face_y_partial, partial2full_y = None, None, None

        if config["remesh"]:
            corres_xy_2_save = p2p_corres
            corres_yx_2_save = p2p_corres_yx
        else:
            corres_xy_2_save = bary_coords_xy
            corres_yx_2_save = bary_coords_yx

    if config["store_vis"]:
        name_x = shape_x["name"].replace("/", "_")
        name_y = shape_y["name"].replace("/", "_")
        file_name = os.path.join(
            config["vis_dir"], f"{split}_{pair_id}_{name_x}_{name_y}.png"
        )
    else:
        file_name = None

    # create colormaps
    if config["setting"] != "partial_partial":
        color_x = create_colormap(vert_x)
        color_y = color_x[p2p_corres_yx]
        color_y[p2p_corres_yx == -1] = [1, 0, 0]

    if config["setting"] == "full_full":
        color_partial_x = None
        color_partial_y = None

    if config["setting"] == "partial_full":
        color_partial_x = None
        color_partial_y = color_x[p2p_corres_yx_partial]
        color_partial_y[p2p_corres_yx_partial == -1] = [1, 0, 0]

    if config["store_vis"] or config["show_output"]:
        plot_shapes(
            vert_x,
            face_x,
            color_x,
            vert_y,
            face_y,
            color_y,
            vert_x_partial,
            face_x_partial,
            color_partial_x,
            vert_y_partial,
            face_y_partial,
            color_partial_y,
            store_output=config["store_vis"],
            show_output=config["show_output"],
            file_name=file_name,
            rot_x=shape_x["rot"],
            area_x=None,
            area_y=None,
            cam_pos_x=shape_x_cam_pos,
            cam_pos_y=shape_y_cam_pos
        )

    # return to original size
    if config["setting"] == "partial_partial":
        vert_x_partial *= area_x
        vert_y_partial *= area_y

    if config["setting"] == "partial_full":
        vert_y_partial *= area_y

    # all full shapes need to go back to original size
    vert_x *= area_x
    vert_y *= area_y

    if config["setting"] == "partial_partial":
        # randomly rotate the partial shapes
        vert_x_2_save = np.dot(vert_x_partial, shape_x['random_rot'])
        vert_y_2_save = np.dot(vert_y_partial, shape_y['random_rot'])

        face_x_2_save = face_x_partial
        face_y_2_save = face_y_partial

    if config["setting"] == "partial_full":
        # randomly rotate the partial shape y and original shape x
        vert_x_2_save = np.dot(vert_x, shape_x['random_rot'])
        vert_y_2_save = np.dot(vert_y_partial, shape_y['random_rot'])

        face_x_2_save = face_x
        face_y_2_save = face_y_partial

    if config["setting"] == "full_full":
        # center and randomly rotate the partial shape y and original shape x
        vert_x_2_save = np.dot(vert_x, shape_x['random_rot'])
        vert_y_2_save = np.dot(vert_y, shape_y['random_rot'])

        face_x_2_save = face_x
        face_y_2_save = face_y

        assert all(
            var is not None
            for var in (
                vert_x_2_save,
                vert_y_2_save,
                face_x_2_save,
                face_y_2_save,
            )
        )

    # # save shapes
    pair_folder = os.path.join(split_folder, f"{pair_id}")
    os.makedirs(pair_folder, exist_ok=True)
    write_off(
        os.path.join(
            pair_folder, f"0_{shape_x['name'].replace('/', '_')}.off"
        ),
        vert_x_2_save,
        face_x_2_save,
    )
    write_off(
        os.path.join(
            pair_folder, f"1_{shape_y['name'].replace('/', '_')}.off"
        ),
        vert_y_2_save,
        face_y_2_save,
    )

    # save correspondences
    if corres_xy_2_save is not None:
        np.save(
            os.path.join(pair_folder, f"corres_01.npy"), corres_xy_2_save
        )
    if corres_yx_2_save is not None:
        np.save(
            os.path.join(pair_folder, f"corres_10.npy"), corres_yx_2_save
        )

    if config["propagate_annotations"]:
        if config["setting"] == "partial_full":
            annotated_data_1 = annotated_data_1[partial2full_y]

        if config["setting"] == "partial_partial":
            annotated_data_0 = annotated_data_0[partial2full_x]
            annotated_data_1 = annotated_data_1[partial2full_y]


        np.save(
            os.path.join(pair_folder, f"0_annotation.npy"), annotated_data_0
            )
        np.save(
            os.path.join(pair_folder, f"1_annotation.npy"), annotated_data_1
            )

    # save all parameters to be able to go back to original size
    dict_x_2_save = {
        "name": shape_x["name"],
        "org_dataset": shape_x["org_dataset"],
        "scale_per_dataset": shape_x["scale"],
        "surface_area": area_x,
        "random_rotation": shape_x["random_rot"],
        "align_rotation": None,
        "align_translation": None,
    }

    dict_y_2_save = {
        "name": shape_y["name"],
        "org_dataset": shape_y["org_dataset"],
        "scale_per_dataset": shape_y["scale"],
        "surface_area": area_y,
        "random_rotation": shape_y["random_rot"],
        "align_rotation": align_R,
        "align_translation": align_t,
    }

    if config["setting"] == "partial_partial":
        dict_x_2_save["partial2full"] = partial2full_x
        dict_y_2_save["partial2full"] = partial2full_y

        dict_x_2_save["cam_pos"] = shape_x_cam_pos
        dict_y_2_save["cam_pos"] = shape_y_cam_pos

        dict_x_2_save["overlap_mask"] = found_faces_mask_yx
        dict_y_2_save["overlap_mask"] = found_faces_mask

        dict_x_2_save["overlap_percentage"] = overlap_yx
        dict_y_2_save["overlap_percentage"] = overlap_xy

    if config["setting"] == "partial_full":
        dict_y_2_save["partial2full"] = partial2full_y
        dict_y_2_save["cam_pos"] = shape_y_cam_pos

        dict_x_2_save["overlap_mask"] = found_faces_mask

    write_pickle(
        os.path.join(
            pair_folder, f"0_{shape_x['name'].replace('/', '_')}_info.pkl"
        ),
        dict_x_2_save,
    )
    write_pickle(
        os.path.join(
            pair_folder, f"1_{shape_y['name'].replace('/', '_')}_info.pkl"
        ),
        dict_y_2_save,
    )
    return (
        vert_x_2_save,
        vert_y_2_save,
        face_x_2_save,
        face_y_2_save,
        dict_x_2_save,
        dict_y_2_save,
        corres_xy_2_save, 
        corres_yx_2_save,
    )

def update_data_path_in_dict(config, all_pairs):
    if config['data_dir'] == "data":
        return all_pairs
    new_all_pairs = []
    for set_pairs in all_pairs:
        new_set_pairs = []
        for pair in set_pairs:
            new_pair = []
            for shape in pair:
                new_path = f"{config['data_dir']}/{shape['file_path'][4:]}"
                new_shape = shape.copy()
                new_shape['file_path'] = new_path
                new_pair += [new_shape]
            new_pair = tuple(new_pair)
            new_set_pairs += [new_pair]
        new_all_pairs += [new_set_pairs]
    return new_all_pairs

def process_pair(pair_id, pair, split, start_time, config, annotation_graph, data_dir, split_folder, nr_split_pairs):
    shape_x = pair[0]
    shape_y = pair[1]
    print(
        f"{split} - {pair_id:03}/{nr_split_pairs} - Time elapsed: {(time.time() - start_time) / 60:.2f} min "
        f"- Working on: {shape_x['name']} and {shape_y['name']}... "
    )
    save_corres_for_shape_pair(shape_x, shape_y, config, annotation_graph, data_dir, split=split, pair_id=pair_id, split_folder=split_folder)

def process_all_pairs_in_parallel(split_pairs, split, config, annotation_graph, data_dir, split_folder, nr_split_pairs, n_jobs=-1):
    start_time = time.time()
    # 
    Parallel(n_jobs=n_jobs)(delayed(process_pair)(pair_id, pair, split, start_time, config, annotation_graph, data_dir, split_folder, nr_split_pairs)
                        for pair_id, pair in enumerate(split_pairs))

def main(config, data_dir):

    if config["original_settings"]:
        original_pairing_file = Path(f"{os.path.dirname(os.path.realpath(__file__))}/../config/pairings_original_{config['setting']}.pkl")
        with open(original_pairing_file, "rb") as f:
            all_pairs = pickle.load(f)
        all_pairs = update_data_path_in_dict(config, all_pairs)
    else:
        # create datasets
        datasets = build_datasets(config)
        train_pairs, test_pairs, val_pairs = get_splits(config, datasets)
        all_pairs = [train_pairs, test_pairs, val_pairs]

        with open(os.path.join(config["output_dir"], "pairings.pkl"), "wb") as f:
            pickle.dump(all_pairs, f)

    # build annotation graph
    annotation_graph = build_adjacency_graph(plot_graph=False)

    splits = ["train", "test", "val"]
    split_folders = [config["train_dir"], config["test_dir"], config["val_dir"]]

    # log time
    start_time = time.time()

    for split_id, split_pairs in enumerate(all_pairs):
        split_folder = split_folders[split_id]
        split = splits[split_id]
        process_all_pairs_in_parallel(split_pairs, split, config, annotation_graph, data_dir, split_folder, len(split_pairs), n_jobs=config["n_jobs"])

    end_time = time.time()
    print(
        f"Dataset was generated successfully in {(end_time - start_time) / 60} minutes. "
    )
