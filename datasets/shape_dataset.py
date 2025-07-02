import os
import re
import yaml
import random
import numpy as np

from pathlib import Path
from utils.misc import sort_list
from ray_casting.camera_position_generator import cam_generator
from utils.shape_util import sample_random_rotation


def generate_pairs(lst, config):
    if isinstance(lst, dict):
        pairs_human = create_pairs(lst['human'], config)
        pairs_four_legged = create_pairs(lst['four-legged'], config)
        pairs = pairs_human + pairs_four_legged
    else:
        pairs = create_pairs(lst, config)
    return pairs


def add_camera_poses(pairs, config):
    cam_pos = cam_generator(similarity=config["cam_pos_regime"], n_cam_pos=config["n_cam_pos"])
    for pair in pairs:
        pair[0]["cam_pos"] = cam_pos[0]
        pair[1]["cam_pos"] = cam_pos[1]
    return pairs


def create_pairs(lst, config):
    pairs = []
    while len(lst) > 2:
        current_element = lst.pop(0)
        paired_element = None
        for index, elem in enumerate(lst):
            if current_element['name'] != elem['name']:
                paired_element = lst.pop(index)
                break
        if paired_element is None:
            lst += [current_element]
        else:
            pairs += [(current_element, paired_element)]

    if len(lst) == 2:
        if lst[0]['name'] != lst[1]['name']:
            pairs += [(lst.pop(0), lst.pop(0))]
        else:
            lst.pop(0)
            lst.pop(0)
    assert len(lst) == 0 or len(lst) == 1
    assert all(dict_x["name"] != dict_y["name"] for dict_x, dict_y in pairs), (f"At least one shape got used in the "
                                                                               f"same pair twice.")

    if config['setting'] == 'partial_partial':
        pairs = add_camera_poses(pairs, config)

    return pairs


def create_pairs_partial_full(config, datasets, lst):
    pairs = []
    if isinstance(lst, dict):
        lst = lst['human'] + lst['four-legged']
    for shape in lst:
        full_shape_name = shape['partial_full_template']
        full_shape_dataset_name = shape['partial_full_dataset']
        if full_shape_name == "undefined" or full_shape_name == shape['name']:
            continue

        assert config['datasets'][full_shape_dataset_name], (f"{full_shape_dataset_name} is dataset of the template "
                                                             f"but it is not active. Activate this dataset first.")
        # sample camera pose
        cam_pos = cam_generator(similarity=config["cam_pos_regime"], n_cam_pos=config["n_cam_pos"])
        shape['cam_pos'] = cam_pos[0]

        full_shape_dataset_object = datasets[full_shape_dataset_name]
        full_shape = {
            "name": full_shape_name,
            "org_dataset": full_shape_dataset_name,
            "category": full_shape_dataset_object.category,
            "file_path": f"{full_shape_dataset_object.data_dir}/{full_shape_dataset_object.folder_name}/"
                         f"{full_shape_name}.{full_shape_dataset_object.file_type}",
            "scale": full_shape_dataset_object.scale,
            "rot": full_shape_dataset_object.rot,
            "partial_full_template": full_shape_name,
            "partial_full_dataset": full_shape_dataset_name,
            "desired_n_face_remeshed": full_shape_dataset_object.partial_full_templates[full_shape_name]
        }
        pairs.append((full_shape, shape))
    return pairs

def get_template_as_x(datasets, lst, config):
    pairs = []
    if isinstance(lst, dict):
        lst = lst['human'] + lst['four-legged']

    if config['setting'] != 'partial_full':
        shape_x_human = [shape_x for shape_x in datasets["faust"].all_shapes if shape_x['name'] == "tr_reg_000"][0]
        shape_x_animal = [shape_x for shape_x in datasets["tosca"].all_shapes if shape_x['name'] == "horse0"][0]
        for shape in lst:
            if shape["category"] == "human":
                pairs.append((shape_x_human, shape))
            elif shape["category"] == "four-legged":
                pairs.append((shape_x_animal, shape))
            else:
                pairs.append((shape_x_animal, shape))
                pairs.append((shape_x_human, shape))
        if config['setting'] == 'partial_partial':
            pairs = add_camera_poses(pairs, config)
    else:
        for shape in lst:
            full_shape_name = shape['partial_full_template']
            if full_shape_name != config["template_partial_full"]:
                continue

            full_shape_dataset_name = shape['partial_full_dataset']
            if full_shape_name == "undefined" or full_shape_name == shape['name']:
                continue

            assert config['datasets'][full_shape_dataset_name], (
                f"{full_shape_dataset_name} is dataset of the template "
                f"but it is not active. Activate this dataset first.")
            # sample camera pose
            cam_pos = cam_generator(similarity=config["cam_pos_regime"], n_cam_pos=config["n_cam_pos"])
            shape['cam_pos'] = cam_pos[0]

            full_shape_dataset_object = datasets[full_shape_dataset_name]
            full_shape = {
                "name": full_shape_name,
                "org_dataset": full_shape_dataset_name,
                "category": full_shape_dataset_object.category,
                "file_path": f"{full_shape_dataset_object.data_dir}/{full_shape_dataset_object.folder_name}/"
                             f"{full_shape_name}.{full_shape_dataset_object.file_type}",
                "scale": full_shape_dataset_object.scale,
                "rot": full_shape_dataset_object.rot,
                "partial_full_template": full_shape_name,
                "partial_full_dataset": full_shape_dataset_name,
                "desired_n_face_remeshed": full_shape_dataset_object.partial_full_templates[full_shape_name]
            }
            pairs.append((full_shape, shape))

    return pairs


def build_datasets(config):
    datasets = dict()

    if config["combinations"] in ["human_centaur", "four-legged_centaur", "all"]:
        assert config["datasets"][
            "tosca"
        ], f"Tosca dataset should be active as it has the centaur."

    if config["combinations"] in ["human", "human_centaur", "all"]:
        if config["datasets"]["faust"]:
            datasets["faust"] = FaustDataset(config)

        if config["datasets"]["scape"]:
            datasets["scape"] = ScapeDataset(config)

        if config["datasets"]["kids"]:
            datasets["kids"] = KidsDataset(config)

        if config["datasets"]["dt4d_human"]:
            datasets["dt4d_human"] = Dt4dHumanDataset(config)

    if config["combinations"] in ["four-legged", "four-legged_centaur", "all"]:
        if config["datasets"]["shrec20"]:
            datasets["shrec20"] = Shrec20Dataset(config)

        if config["datasets"]["smal"]:
            datasets["smal"] = SmalDataset(config)

        if config["datasets"]["dt4d_animal"]:
            datasets["dt4d_animal"] = Dt4dAnimalDataset(config)

    if config["datasets"]["tosca"]:
        datasets["tosca"] = ToscaDataset(config)

    return datasets


def get_splits(config, datasets):
    if config["combinations"] != "all":
        train_splits = []
        test_splits = []
        val_splits = []

        for _, dataset in datasets.items():
            train_splits += dataset.train_split
            test_splits += dataset.test_split
            val_splits += dataset.val_split

        random.shuffle(train_splits)
        random.shuffle(test_splits)
        random.shuffle(val_splits)
    else:
        train_splits = {"human": [], "four-legged": []}
        test_splits = {"human": [], "four-legged": []}
        val_splits = {"human": [], "four-legged": []}

        for _, dataset in datasets.items():
            if dataset.data_name != "tosca":
                train_splits[dataset.category] += dataset.train_split
                test_splits[dataset.category] += dataset.test_split
                val_splits[dataset.category] += dataset.val_split
            else:
                for category in dataset.category:
                    if category != "centaur":
                        train_splits[category] += dataset.train_split[category]
                        test_splits[category] += dataset.test_split[category]
                        val_splits[category] += dataset.val_split[category]

        for category in train_splits.keys():
            random.shuffle(train_splits[category])
            random.shuffle(val_splits[category])
            random.shuffle(test_splits[category])

    if config['setting'] == "partial_full":
        train_pairs = create_pairs_partial_full(config, datasets, train_splits)
        test_pairs = create_pairs_partial_full(config, datasets, test_splits)
        val_pairs = create_pairs_partial_full(config, datasets, val_splits)
        random.shuffle(train_pairs)
        random.shuffle(test_pairs)
        random.shuffle(val_pairs)
    else:
        train_pairs = generate_pairs(train_splits, config)
        test_pairs = generate_pairs(test_splits,  config)
        val_pairs = generate_pairs(val_splits,  config)
        random.shuffle(train_pairs)
        random.shuffle(test_pairs)
        random.shuffle(val_pairs)

    # add random rotations to all pairs
    one_axis_rotation = config["one_axis_rotation"]
    train_pairs = add_random_rotations(train_pairs, one_axis_rotation)
    test_pairs = add_random_rotations(test_pairs, one_axis_rotation)
    val_pairs = add_random_rotations(val_pairs, one_axis_rotation)

    return train_pairs, test_pairs, val_pairs


def add_random_rotations(pairs, one_axis_rotation=False):
    for pair in pairs:
        pair[0]["random_rot"] = sample_random_rotation(one_axis_rotation)
        pair[1]["random_rot"] = sample_random_rotation(one_axis_rotation)
    return pairs


def get_partial_full_template(setting, templates, dataset_name, shape_name, columns=[3, 5]):
    if setting != "partial_full":
        return ['undefined', 'undefined']
    if dataset_name in ["faust", "kids", "scape"]:
        result = templates[templates[:, 0] == f"{dataset_name}", columns[0]:columns[1]][0]
    elif dataset_name == "shrec20":
        result = templates[templates[:, 0] == f"shrec20_{shape_name}", columns[0]:columns[1]][0]
    elif dataset_name == "tosca":
        matches = re.search(r"^(.*?)(\d+)$", shape_name)
        shape_org_name = matches.group(1)
        result = templates[templates[:, 0] == f"tosca_{shape_org_name}", columns[0]:columns[1]][0]
    elif dataset_name == "smal":
        split_name = shape_name.split("/")
        result = templates[templates[:, 0] == f"smal_{split_name[0]}", columns[0]:columns[1]][0]
    elif dataset_name == "dt4d_human" or dataset_name == "dt4d_animal":
        split_name = shape_name.split("/")
        dt4d_type = dataset_name.split("_")[1]
        result = templates[templates[:, 0] == f"dt4d_{dt4d_type}_{split_name[0]}", columns[0]:columns[1]][0]
    return result


class CustomDataset:
    def __init__(self, general_config, data_name):
        self.general_config = general_config
        conf_file = Path(
            f"{os.path.dirname(os.path.realpath(__file__))}/../config/{data_name}.yaml"
        )
        assert (
            conf_file.is_file()
        ), f"Dataset {data_name} not found in config directory."

        self.templates = np.genfromtxt(
            os.path.join(general_config["config_dir"], general_config["template_file"]),
            delimiter=",", dtype=str, skip_header=True)

        with open(conf_file, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.data_dir = self.general_config['data_dir']
        self.setting = self.general_config['setting']
        self.data_name = data_name
        self.folder_name = self.config["folder_name"]
        self.file_type = self.config["file_type"]
        self.category = self.config["category"]
        self.scale = self.config["scale"]
        self.rot = self.config["rot"] if "rot" in self.config.keys() else None

        self.all_shapes = []
        self.train_split = []
        self.test_split = []
        self.val_split = []
        self._init_data()

        if self.setting == "partial_full":
            self._init_template_shapes()

        if self.data_name != "tosca":
            assert len(self.all_shapes) == len(self.train_split) + len(
                self.test_split
            ) + len(
                self.val_split
            ), f"Sum is {len(self.train_split) + len(self.test_split) + len(self.val_split)} while all_shapes has {len(self.all_shapes)}"

        # Use every shape twice
        if self.config['shape_usage_counts']:
            if data_name == "tosca": # and self.general_config["combinations"] == "all":
                self.train_split = {key: self.train_split[key] * self.config['shape_usage_counts'] for key in self.train_split}
                self.all_shapes = (self.train_split['human'] + self.train_split['four-legged']
                                   + self.test_split['human'] + self.test_split['four-legged']
                                   + self.val_split['human'] + self.val_split['four-legged'])
            else:
                self.train_split *= self.config['shape_usage_counts']
                self.all_shapes = self.train_split + self.test_split + self.val_split

        self._size = len(self.all_shapes)
        assert self._size != 0


    def _init_data(self):
        all_files = sort_list(os.listdir(f"{self.data_dir}/{self.folder_name}/"))
        a = 1
        for f in all_files:
            category = self.config["category"]
            # check if category is a list
            if isinstance(self.category, list):
                regex_category = self.config["regex_category"]
                for i, cat in enumerate(self.category):
                    curr_regex = regex_category[i]
                    # check if f fulfills the regex
                    if re.search(curr_regex, f, re.IGNORECASE):
                        category = cat
                        break
            if isinstance(category, list):
                print(f"Category for {f} not found")
                continue
            elif os.path.isdir(f"{self.data_dir}/{self.folder_name}/{f}"):
                self.all_shapes += [
                    {
                        "name": f"{f}/{f_sub[:-4]}",
                        "org_dataset": self.data_name,
                        "category": category,
                        "file_path": f"{self.data_dir}/{self.folder_name}/{f}/{f_sub}",
                        "scale": self.scale,
                        "rot": self.rot,
                        "partial_full_template": get_partial_full_template(self.setting, self.templates, self.data_name,
                                                                           f"{f}/{f_sub[:-4]}")[0],
                        "partial_full_dataset": get_partial_full_template(self.setting, self.templates, self.data_name,
                                                                          f"{f}/{f_sub[:-4]}")[1],
                        "desired_n_face_remeshed": random.randint(18000, 20000),
                    }
                    for f_sub in sorted(os.listdir(f"{self.data_dir}/{self.folder_name}/{f}"))
                    if f_sub.endswith(self.file_type)
                ]
            else:
                if f.endswith(self.file_type):
                    self.all_shapes += [
                        {
                            "name": f[:-4],
                            "org_dataset": self.data_name,
                            "category": category,
                            "file_path": f"{self.data_dir}/{self.folder_name}/{f}",
                            "scale": self.scale,
                            "rot": self.rot,
                            "partial_full_template":
                                get_partial_full_template(self.setting, self.templates, self.data_name, f[:-4])[0],
                            "partial_full_dataset":
                                get_partial_full_template(self.setting, self.templates, self.data_name, f[:-4])[1],
                            "desired_n_face_remeshed": random.randint(18000, 20000),
                        }
                    ]

    def _init_template_shapes(self):
        self.partial_full_templates = {}
        for row in self.templates:
            dataset_name = row[4].split("_")
            if len(dataset_name) == 2:
                if f"{dataset_name[0]}_{dataset_name[1]}" != self.data_name:
                    continue
            if len(dataset_name) == 1 and dataset_name[0] != self.data_name:
                continue
            key = row[3]
            if key not in self.partial_full_templates:
                desired_face_count = [shape['desired_n_face_remeshed'] for shape in self.all_shapes if shape['name'] == key]
                assert len(desired_face_count) > 0
                self.partial_full_templates[key] = desired_face_count[0]


    def __getitem__(self, index):
        item = self.all_shapes[index]
        return item

    def __len__(self):
        return self._size


class FaustDataset(CustomDataset):
    def __init__(self, general_config):
        super(FaustDataset, self).__init__(general_config, data_name="faust")
        original_n_shapes = 100
        assert (
                self.n_shapes == original_n_shapes
        ), f"FAUST dataset should contain {original_n_shapes} human body shapes, but get {len(self)}."

    def _init_data(self):
        super(FaustDataset, self)._init_data()
        n_train_shapes = self.config["splits"]["train"]
        n_val_shapes = self.config["splits"]["val"]
        n_test_shapes = self.config["splits"]["test"]
        self.train_split = self.all_shapes[:n_train_shapes]
        self.val_split = self.all_shapes[n_train_shapes: n_train_shapes + n_val_shapes]
        self.test_split = self.all_shapes[n_train_shapes + n_val_shapes:]
        self.n_shapes = len(self.all_shapes)

        assert (
                len(self.train_split) == n_train_shapes
        ), f"Number of train shapes should be {n_train_shapes}, but got {len(self.train_split)}"
        assert (
                len(self.val_split) == n_val_shapes
        ), f"Number of val shapes should be {n_val_shapes}, but got {len(self.val_split)}"
        assert (
                len(self.test_split) == n_test_shapes
        ), f"Number of test shapes should be {n_test_shapes}, but got {len(self.test_split)}"


class ScapeDataset(CustomDataset):
    def __init__(self, general_config):
        super(ScapeDataset, self).__init__(general_config, data_name="scape")
        original_n_shapes = 71
        assert (
                self.n_shapes == original_n_shapes
        ), f"SCAPE dataset should contain {original_n_shapes} human body shapes, but get {len(self)}."

    def _init_data(self):
        super(ScapeDataset, self)._init_data()
        n_train_shapes = self.config["splits"]["train"]
        n_val_shapes = self.config["splits"]["val"]
        n_test_shapes = self.config["splits"]["test"]
        self.train_split = self.all_shapes[:n_train_shapes]
        self.val_split = self.all_shapes[n_train_shapes: n_train_shapes + n_val_shapes]
        self.test_split = self.all_shapes[n_train_shapes + n_val_shapes:]
        self.n_shapes = len(self.all_shapes)

        assert (
                len(self.train_split) == n_train_shapes
        ), f"Number of train shapes should be {n_train_shapes}, but got {len(self.train_split)}"
        assert (
                len(self.val_split) == n_val_shapes
        ), f"Number of val shapes should be {n_val_shapes}, but got {len(self.val_split)}"
        assert (
                len(self.test_split) == n_test_shapes
        ), f"Number of test shapes should be {n_test_shapes}, but got {len(self.test_split)}"


class KidsDataset(CustomDataset):
    def __init__(self, general_config):
        super(KidsDataset, self).__init__(general_config, data_name="kids")
        original_n_shapes = 32
        assert (
                self.n_shapes == original_n_shapes
        ), f"kids dataset should contain {original_n_shapes} human body shapes, but get {len(self)}."

    def _init_data(self):
        super(KidsDataset, self)._init_data()
        n_train_shapes = self.config["splits"]["train"]
        n_val_shapes = self.config["splits"]["val"]
        n_test_shapes = self.config["splits"]["test"]
        self.train_split = self.all_shapes[:n_train_shapes]
        self.val_split = self.all_shapes[n_train_shapes: n_train_shapes + n_val_shapes]
        self.test_split = self.all_shapes[n_train_shapes + n_val_shapes:]
        self.n_shapes = len(self.all_shapes)

        assert (
                len(self.train_split) == n_train_shapes
        ), f"Number of train shapes should be {n_train_shapes}, but got {len(self.train_split)}"
        assert (
                len(self.val_split) == n_val_shapes
        ), f"Number of val shapes should be {n_val_shapes}, but got {len(self.val_split)}"
        assert (
                len(self.test_split) == n_test_shapes
        ), f"Number of test shapes should be {n_test_shapes}, but got {len(self.test_split)}"


class Shrec20Dataset(CustomDataset):
    def __init__(self, general_config):
        super(Shrec20Dataset, self).__init__(general_config, data_name="shrec20")
        original_n_shapes = 11
        assert (
                self.n_shapes == original_n_shapes
        ), f"Shrec20 dataset should contain {original_n_shapes} shapes, but get {len(self)}."

    def _init_data(self):
        super(Shrec20Dataset, self)._init_data()

        self.all_shapes = [
            d
            for d in self.all_shapes
            if d["name"] not in self.config["splits"]["excluded_shapes"]
        ]
        self.train_split = [
            d for d in self.all_shapes if d["name"] in self.config["splits"]["train"]
        ]
        self.val_split = [
            d for d in self.all_shapes if d["name"] in self.config["splits"]["val"]
        ]
        self.test_split = [
            d for d in self.all_shapes if d["name"] in self.config["splits"]["test"]
        ]
        self.n_shapes = len(self.all_shapes)


class SmalDataset(CustomDataset):
    def __init__(self, general_config):
        super(SmalDataset, self).__init__(general_config, data_name="smal")
        original_n_shapes = 49
        assert (
                self.n_shapes == original_n_shapes
        ), f"Shrec20 dataset should contain {original_n_shapes} shapes, but get {len(self)}."

    def _init_data(self):
        super(SmalDataset, self)._init_data()

        for shape in self.all_shapes:
            shape_class = shape["name"].split("/")[0]

            if shape_class in self.config["splits"]["train"]:
                self.train_split += [shape]

            if shape_class in self.config["splits"]["test"]:
                self.test_split += [shape]

            if shape_class in self.config["splits"]["val"]:
                self.val_split += [shape]

        self.n_shapes = len(self.all_shapes)


class ToscaDataset(CustomDataset):
    def __init__(self, general_config):
        self.combinations = general_config['combinations']
        super(ToscaDataset, self).__init__(general_config, data_name="tosca")

    def _init_data(self):
        super(ToscaDataset, self)._init_data()

        def remove_digits_at_end(shape_name):
            return re.sub(r"\d+$", "", shape_name)

        classes = dict()

        self.all_shapes = self.cat = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) != "gorilla"
        ]
        assert (
                len(self.all_shapes) == 76
        ), f"After removing gorilla shapes, Tosca should have 76 shapes {len(self.all_shapes)}."

        classes["cat"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "cat"
        ]
        assert (
                len(classes["cat"]) == 11
        ), f"Tosca dataset should contain 11 cat shapes, but get {len(classes['cat'])}."

        classes["centaur"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "centaur"
        ]
        assert (
                len(classes["centaur"]) == 6
        ), f"Tosca dataset should contain 6 centaur shapes, but get {len(classes['centaur'])}."

        classes["david"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "david"
        ]
        assert (
                len(classes["david"]) == 7
        ), f"Tosca dataset should contain 7 david shapes, but get {len(classes['david'])}."

        classes["dog"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "dog"
        ]
        assert (
                len(classes["dog"]) == 9
        ), f"Tosca dataset should contain 9 dog shapes, but get {len(classes['dog'])}."

        classes["horse"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "horse"
        ]
        assert (
                len(classes["horse"]) == 8
        ), f"Tosca dataset should contain 8 horse shapes, but get {len(classes['horse'])}."

        classes["michael"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "michael"
        ]
        assert (
                len(classes["michael"]) == 20
        ), f"Tosca dataset should contain 20 michael shapes, but get {len(classes['michael'])}."

        classes["victoria"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "victoria"
        ]
        assert (
                len(classes["victoria"]) == 12
        ), f"Tosca dataset should contain 12 victoria shapes, but get {len(classes['victoria'])}."

        classes["wolf"] = [
            d for d in self.all_shapes if remove_digits_at_end(d["name"]) == "wolf"
        ]
        assert (
                len(classes["wolf"]) == 3
        ), f"Tosca dataset should contain 3 wolf shapes, but get {len(classes['wolf'])}."

        if self.combinations in ["human", "human_centaur"]:
            self.train_split = [
                item
                for class_name in self.config["splits"]["human"]["train"]
                for item in classes[class_name]
            ]
            self.val_split = [
                item
                for class_name in self.config["splits"]["human"]["val"]
                for item in classes[class_name]
            ]
            self.test_split = [
                item
                for class_name in self.config["splits"]["human"]["test"]
                for item in classes[class_name]
            ]

        if self.combinations in ["four-legged", "four-legged_centaur"]:
            self.train_split += [
                item
                for class_name in self.config["splits"]["four-legged"]["train"]
                for item in classes[class_name]
            ]
            self.val_split += [
                item
                for class_name in self.config["splits"]["four-legged"]["val"]
                for item in classes[class_name]
            ]
            self.test_split += [
                item
                for class_name in self.config["splits"]["four-legged"]["test"]
                for item in classes[class_name]
            ]

        if self.combinations in ["human_centaur", "four-legged_centaur"]:
            self.train_split += [item for item in classes["centaur"]]

        if self.combinations == "all":
            self.train_split = dict()
            self.test_split = dict()
            self.val_split = dict()

            for type in ["human", "four-legged"]:
                self.train_split[type] = [
                                             item
                                             for class_name in self.config["splits"][type]["train"]
                                             for item in classes[class_name]
                                         ] + [item for item in classes["centaur"]]

                self.test_split[type] = [
                    item
                    for class_name in self.config["splits"][type]["test"]
                    for item in classes[class_name]
                ]

                self.val_split[type] = [
                    item
                    for class_name in self.config["splits"][type]["val"]
                    for item in classes[class_name]
                ]


class Dt4dHumanDataset(CustomDataset):
    def __init__(self, general_config):
        super(Dt4dHumanDataset, self).__init__(general_config, data_name="dt4d_human")
        original_n_shapes = 248
        assert (
                self.n_shapes == original_n_shapes
        ), f"Dt4d_human dataset should contain {original_n_shapes} shapes, but get {len(self)}."

    def _init_data(self):
        super(Dt4dHumanDataset, self)._init_data()

        train_classes = []
        test_classes = []
        val_classes = []
        for row in self.templates:
            split = row[2]
            if len(row[0].split("_")) != 3:
                continue
            if row[0].split("_")[1] != "human":
                continue
            class_name = row[0].split("_")[2]
            if split == "train":
                self.train_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                train_classes += [class_name]

            if split == "val":
                self.val_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                test_classes += [class_name]

            if split == "test":
                self.test_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                val_classes += [class_name]

        assert len(self.train_split) == (
                31 * len(train_classes)
        ), f"Training split should contain 31 * {len(train_classes)}, but get {len(self.train_split)}"

        assert len(self.test_split) == (
                31 * len(test_classes)
        ), f"Testing split should contain 31 * {len(test_classes)}, but get {len(self.test_split)}"

        assert len(self.val_split) == (
                31 * len(val_classes)
        ), f"Validation split should contain 31 * {len(val_classes)}, but get {len(self.val_split)}"

        self.n_shapes = len(self.all_shapes)


class Dt4dAnimalDataset(CustomDataset):
    def __init__(self, general_config):
        super(Dt4dAnimalDataset, self).__init__(general_config, data_name="dt4d_animal")
        original_n_shapes = 1950
        assert (
                self.n_shapes == original_n_shapes
        ), f"Dt4d_animal dataset should contain {original_n_shapes} human body shapes, but get {len(self)}."

    def _init_data(self):
        super(Dt4dAnimalDataset, self)._init_data()

        train_classes = []
        test_classes = []
        val_classes = []
        for row in self.templates:
            split = row[2]
            if len(row[0].split("_")) != 3:
                continue
            if row[0].split("_")[1] != "animal":
                continue
            class_name = row[0].split("_")[2]
            if split == "train":
                self.train_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                train_classes += [class_name]

            if split == "val":
                self.val_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                test_classes += [class_name]

            if split == "test":
                self.test_split += [
                    d for d in self.all_shapes if d["name"].split("/")[0] == class_name
                ]
                val_classes += [class_name]

        assert len(self.train_split) == (
                50 * len(train_classes)
        ), f"Training split should contain 50 * {len(train_classes)}, but get {len(self.train_split)}"

        assert len(self.test_split) == (
                50 * len(test_classes)
        ), f"Testing split should contain 50 * {len(test_classes)}, but get {len(self.test_split)}"

        assert len(self.val_split) == (
                50 * len(val_classes)
        ), f"Validation split should contain 50 * {len(val_classes)}, but get {len(self.val_split)}"

        self.n_shapes = len(self.all_shapes)
