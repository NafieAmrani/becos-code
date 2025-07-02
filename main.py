import os
import yaml
import argparse

from utils.misc import mkdir_and_rename, set_random_seed
from preprocessing.load_shape_pairs import main

def prepare_config(conf):
    with open(conf, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # save config path
    config["config_dir"] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "config"
    )

    # seed everything
    set_random_seed(config["SEED"])
    data_dir = config["data_dir"]

    # make output folder
    output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    config["output_dir"] = os.path.join(output_folder, config["setting"])
    config["train_dir"] = os.path.join(config["output_dir"], "train")
    config["val_dir"] = os.path.join(config["output_dir"], "val")
    config["test_dir"] = os.path.join(config["output_dir"], "test")
    config["logs_dir"] = os.path.join(config["output_dir"], "logs")
    mkdir_and_rename(config["output_dir"])
    mkdir_and_rename(config["train_dir"])
    mkdir_and_rename(config["val_dir"])
    mkdir_and_rename(config["test_dir"])
    mkdir_and_rename(config["logs_dir"])

    # hardcode these options to avoid confusion in the option files
    config['use_precompute_remeshing'] = False
    config['update_precomputed_remeshed'] = False
    config['use_precomputed_partial_raycasting'] = config['original_settings']
    config['update_precomputed_raycasting'] = False
    config['use_shape_precomputation'] = False
    config['save_shape_precomputation'] = False
    config['use_corres_precomputation'] = False
    config['save_corres_precomputation'] = False
    config['store_vis'] = False
    config['show_output'] = False

    # save the remeshed templates to avoid remeshing it every iteration
    if config['use_precompute_remeshing']:
        config["precomputed_remeshed"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      "precomputed_remeshed")
        os.makedirs(config["precomputed_remeshed"], exist_ok=True)
    
    if config['use_precomputed_partial_raycasting']:
        conf_setting = config["setting"]
        config["precomputed_partial_raycasting"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                      f"precomputed_triangle_ids/{conf_setting}")
        os.makedirs(config["precomputed_partial_raycasting"], exist_ok=True)

    # precomputerd correspondences
    config["precomputed_shapes_dir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    "corres_comp", "precomputed_shapes")
    os.makedirs(config["precomputed_shapes_dir"], exist_ok=True)
    config["precomputed_corres_dir"] = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                    "corres_comp", "precomputed_corres")
    os.makedirs(config["precomputed_corres_dir"], exist_ok=True)


    # vis output
    if config["store_vis"]:
        config["vis_dir"] = os.path.join(config["output_dir"], "visualisations")
        mkdir_and_rename(config["vis_dir"])

    with open(f"{config['logs_dir']}/config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return config, data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="general_config")

    args = parser.parse_args()
    conf = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "config", f"{args.config}.yaml"
    )
    config, data_dir = prepare_config(conf)
    
    main(config, data_dir)
