"""Class representing the datasets for training // testing MIR samples.

Each sample is one situation in which the initial association of the tracker
failed, described by the three mask embeddings (detection, previous and current
frame), their bounding boxes, the frame indices, and a label stating whether the
track should have been recovered. Samples are extracted offline and stored as one
pickle file per sequence.
"""

import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from sgar_mot.utils.config import get_ram_usage, get_total_ram


class MIRDatasetTrain(torch.utils.data.Dataset):
    """Dataset for training MIR."""

    def __init__(self, data_path, memory_limit=None, seq_split=False, test_set=False, seq_split_halves=2,
                 data_type=None):
        """Initialization function. We load the data from the path provided.

        Parameters
        ----------
        data_path : str
            Path to the data to be loaded.
        memory_limit : float
            Memory limit to be used for loading the data, in GB. If None, no limit is established.
        seq_split : bool
            If True, the dataset will be split into halves and loaded alternatively.
        test_set : bool
            If True, the dataset will be loaded as a test set.
        seq_split_halves : int
            Number of halves to split the dataset into. Default is 2.
        data_type : int
            If not None, type of annotations that will be loaded (0: Normal, 1: With IDSW, 2: With bbox displacement).
        """
        self.seq_split = seq_split
        self.data_path = data_path
        self.memory_limit = memory_limit
        self.test_set = test_set
        self.seq_split_halves = seq_split_halves
        self.data_type = data_type

        # data_path can represent a directory or a file. If it is a file, we load it directly. If not, we will load all the files inside:
        print("Total memory available:", get_total_ram() / (1024 * 1024 * 1024))
        if os.path.isdir(self.data_path):
            # If we split the dataset, we shuffle the directory and split it into parts:
            if self.seq_split:
                self.full_seqs = os.listdir(self.data_path)
                np.random.shuffle(self.full_seqs)
                # We create the number of halves specified:
                self.seqs = []
                for i in range(self.seq_split_halves):
                    self.seqs.append(self.full_seqs[int(i * len(self.full_seqs) / self.seq_split_halves):int((i + 1) * len(self.full_seqs) / self.seq_split_halves)])
                # We will begin with the first group of sequences:
                self.seq_loaded = 1
                seqs = self.seqs[self.seq_loaded - 1]
            else:
                # If we don't split the dataset, we will simply load it full:
                seqs = os.listdir(self.data_path)
                self.seq_loaded = 1

            # We will load the data in a loop, checking the memory consumption:
            self.__load_data(seqs)

        else:
            self.num_seq = 1
            f = open(data_path, "rb")
            self.data_loaded = pickle.load(f)
            f.close()

        # Define transformations. We will convert all items to torch tensors:
        self.transforms = torch.Tensor

    def __load_data(self, seqs):
        """Function for loading the data from the files in the directory. We will load the data in a loop, checking the memory consumption.

        Parameters
        ----------
        seqs : list
            List of sequences to be loaded.
        """
        # For controlling memory consumption (if we establish a limit):
        memory_cons_by_dataset = 0
        self.num_seq = len(seqs)
        self.data_loaded = []

        print("Loading files from directory", self.data_path, "with memory limit", self.memory_limit,
              "GB. (Using the half number", self.seq_loaded, "of sequences.)")

        for file in tqdm(seqs):
            # Every time we load a file, we check if we have reached the memory limit:
            mem_usg_before = get_ram_usage() / (1024 * 1024 * 1024)

            # Open the file and load the data:
            f = open(self.data_path + "/" + file, "rb")
            data = pickle.load(f)
            final_data = []

            if self.data_type is not None:
                for d in data:
                    if d["type"] != int(self.data_type):
                        continue
                    final_data.append(d)
            else:
                final_data = data

            if self.test_set:
                for d in data:
                    d["video_name"] = file.split(".")[0]

            self.data_loaded += final_data
            mem_usg_after = get_ram_usage() / (1024 * 1024 * 1024)
            # Compute total consumption:
            memory_cons_by_dataset += mem_usg_after - mem_usg_before
            f.close()

            # Check if we have reached the memory limit (if it is established):
            if self.memory_limit is not None and memory_cons_by_dataset > self.memory_limit:
                print("Reached memory limit. Stopping loading files.")
                exit()

    def alternate_load(self):
        """Function that changes the data loaded to the other half of the sequences."""
        if self.seq_split:
            if self.seq_loaded < self.seq_split_halves:
                self.seq_loaded += 1
                seqs = self.seqs[self.seq_loaded - 1]
            else:
                self.seqs = []
                for i in range(self.seq_split_halves):
                    self.seqs.append(self.full_seqs[int(i * len(self.full_seqs) / self.seq_split_halves):int((i + 1) * len(self.full_seqs) / self.seq_split_halves)])
                # We will begin with the first group of sequences:
                self.seq_loaded = 1
                seqs = self.seqs[self.seq_loaded - 1]

            self.__load_data(seqs)
        else:
            print("The dataset is not split. No changes will be made.")

    def __getitem__(self, index):
        """Extraction of an item from dataset"""

        # Get the item from the list:
        frame_id = self.data_loaded[index]["frame_idx"]
        object_id = self.data_loaded[index]["object_id"]
        prev_frame_bbox = self.data_loaded[index]["prev_frame_bbox"]
        prev_frame_mask_emb = self.data_loaded[index]["prev_frame_mask_emb"]
        det_frame_bbox = self.data_loaded[index]["det_frame_bbox"]
        det_frame_mask_emb = self.data_loaded[index]["det_frame_mask_emb"]
        curr_frame_bbox = self.data_loaded[index]["curr_frame_bbox"]
        curr_frame_mask_emb = self.data_loaded[index]["curr_frame_mask_emb"]
        output = self.data_loaded[index]["output"]
        det_frame_id = self.data_loaded[index]["det_frame_idx"]
        t_frame_id = self.data_loaded[index]["frame_idx"]
        output = np.array([1]) if output > 0 else np.array([0])

        # Apply transforms over embeddings:
        frame_id = self.transforms([frame_id])
        object_id = self.transforms([object_id])
        prev_frame_bbox = self.transforms(prev_frame_bbox)
        prev_frame_mask_emb = self.transforms(prev_frame_mask_emb)
        det_frame_bbox = self.transforms(det_frame_bbox)
        det_frame_mask_emb = self.transforms(det_frame_mask_emb)
        curr_frame_bbox = self.transforms(curr_frame_bbox)
        curr_frame_mask_emb = self.transforms(curr_frame_mask_emb)

        label = self.transforms(output)

        # Concatenate mask embeddings:
        mask_embeddings = torch.cat([det_frame_mask_emb, prev_frame_mask_emb, curr_frame_mask_emb], dim=0)
        # Concatenate bboxes on a new dimension (each bbox in a separated dimension):
        msk_bboxes = torch.stack([det_frame_bbox, prev_frame_bbox, curr_frame_bbox], dim=0)

        return frame_id, object_id, mask_embeddings, msk_bboxes, det_frame_id, t_frame_id, label

    def __len__(self):
        """Extraction of the dataset length"""
        return len(self.data_loaded)

    @property
    def num_training_targets(self):
        """Extraction of number of training targets"""
        return len(self.data_loaded)


class MIRDatasetTest(torch.utils.data.Dataset):
    """Dataset for validating and testing MIR."""

    def __init__(self, data_path, filter_dist_0=False, data_type=None):
        """Initialization function. We load the data from the path provided.

        Parameters
        ----------
        data_path : str
            Path to the data to be loaded.
        filter_dist_0 : bool
            If True, we will filter the data where det_frame_idx != frame_idx - 1.
        data_type : int
            If not None, type of annotations that will be loaded (0: Normal, 1: With IDSW, 2: With bbox displacement).
        """
        self.data_path = data_path
        self.filter_dist_0 = filter_dist_0
        self.data_type = data_type

        # data_path can represent a directory or a file. If it is a file, we load it directly. If not, we will load all the files inside:
        print("Total memory available:", get_total_ram() / (1024 * 1024 * 1024))
        if os.path.isdir(self.data_path):
            # We will simply load the full dataset:
            seqs = os.listdir(self.data_path)
            self.seq_loaded = 1
            # We will load the data in a loop, checking the memory consumption:
            self.__load_data(seqs)

        else:
            self.num_seq = 1
            f = open(data_path, "rb")
            self.data_loaded = pickle.load(f)
            f.close()

        # Define transformations. We will convert all items to torch tensors:
        self.transforms = torch.Tensor

    def __load_data(self, seqs):
        """Function for loading the data from the files in the directory.

        Parameters
        ----------
        seqs : list
            List of sequences to be loaded.
        """
        self.num_seq = len(seqs)
        self.data_loaded = []

        print("Loading files from directory", self.data_path, "(Using the half number", self.seq_loaded, "of sequences.)")
        removed_data = 0

        for file in tqdm(seqs):
            # Open the file and load the data:
            f = open(self.data_path + "/" + file, "rb")
            data = pickle.load(f)
            new_data = []

            for d in data:
                d["video_name"] = file.split(".")[0]
                # Check if det_frame_idx coincides with frame_idx - 1 -- remove those examples:
                if self.filter_dist_0:
                    if d["det_frame_idx"] != d["frame_idx"] - 1:
                        if self.data_type is not None:
                            if "type" in d:
                                if d["type"] == int(self.data_type):
                                    new_data.append(d)
                                else:
                                    removed_data += 1
                            else:
                                new_data.append(d)
                        else:
                            new_data.append(d)
                    else:
                        removed_data += 1
                elif self.data_type is not None:
                    if "type" in d:
                        if d["type"] == int(self.data_type):
                            new_data.append(d)
                        else:
                            removed_data += 1
                    else:
                        new_data.append(d)
                else:
                    new_data.append(d)

            self.data_loaded += new_data
            f.close()

        if self.filter_dist_0:
            print("Removed data:", removed_data, "objects.")

    def __getitem__(self, index):
        """Extraction of an item from dataset"""

        # Get the item from the list:
        frame_id = self.data_loaded[index]["frame_idx"]
        object_id = self.data_loaded[index]["object_id"]
        prev_frame_bbox = self.data_loaded[index]["prev_frame_bbox"]
        prev_frame_mask_emb = self.data_loaded[index]["prev_frame_mask_emb"]
        det_frame_bbox = self.data_loaded[index]["det_frame_bbox"]
        det_frame_mask_emb = self.data_loaded[index]["det_frame_mask_emb"]
        curr_frame_bbox = self.data_loaded[index]["curr_frame_bbox"]
        curr_frame_mask_emb = self.data_loaded[index]["curr_frame_mask_emb"]
        output = self.data_loaded[index]["output"]
        det_frame_id = self.data_loaded[index]["det_frame_idx"]
        t_frame_id = self.data_loaded[index]["frame_idx"]
        output = np.array([1]) if output > 0 else np.array([0])

        # Apply transforms over embeddings:
        frame_id = self.transforms([frame_id])
        object_id = self.transforms([object_id])
        prev_frame_bbox = self.transforms(prev_frame_bbox)
        prev_frame_mask_emb = self.transforms(prev_frame_mask_emb)
        det_frame_bbox = self.transforms(det_frame_bbox)
        det_frame_mask_emb = self.transforms(det_frame_mask_emb)
        curr_frame_bbox = self.transforms(curr_frame_bbox)
        curr_frame_mask_emb = self.transforms(curr_frame_mask_emb)

        label = self.transforms(output)

        # Concatenate mask embeddings:
        mask_embeddings = torch.cat([det_frame_mask_emb, prev_frame_mask_emb, curr_frame_mask_emb], dim=0)
        # Concatenate bboxes on a new dimension (each bbox in a separated dimension):
        msk_bboxes = torch.stack([det_frame_bbox, prev_frame_bbox, curr_frame_bbox], dim=0)

        return frame_id, object_id, mask_embeddings, msk_bboxes, det_frame_id, t_frame_id, label

    def __len__(self):
        """Extraction of the dataset length"""
        return len(self.data_loaded)

    @property
    def num_training_targets(self):
        """Extraction of number of training targets"""
        return len(self.data_loaded)
