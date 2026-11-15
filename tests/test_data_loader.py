"""
Unit tests for DEAP dataset loading, channel slicing, and validation using standard unittest.
"""

import os
import unittest
import numpy as np
from pathlib import Path

from src.config import CONFIG
from src.data_loader import DEAPDataLoader
from sample_data.generate_sample_eeg import create_sample_dataset


class TestDEAPDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./sample_data"
        create_sample_dataset(
            output_dir=cls.test_dir,
            participant_ids=[1, 28],
            trials_per_participant=3
        )
        cls.loader = DEAPDataLoader()

    def test_channel_slicing_and_shapes(self):
        """Verify that only the 32 EEG channels are extracted and peripheral channels are dropped."""
        p1_file = Path(self.test_dir) / "s01.dat"
        eeg_data, labels, meta = self.loader.load_single_participant(str(p1_file), validate=True)
        
        self.assertTrue(meta["is_valid"])
        self.assertEqual(meta["original_shape"], (3, 40, 8064))
        self.assertEqual(eeg_data.shape, (3, 32, 8064), "Must strictly extract 32 EEG channels!")
        self.assertEqual(labels.shape, (3, 4), "Must extract all 4 targets (V, A, D, L)!")
        self.assertFalse(np.isnan(eeg_data).any(), "EEG data must not contain NaNs")
        self.assertFalse(np.isinf(eeg_data).any(), "EEG data must not contain Infs")

    def test_label_ranges(self):
        """Verify that labels are continuous in the 1-9 range."""
        p28_file = Path(self.test_dir) / "s28.dat"
        _, labels, _ = self.loader.load_single_participant(str(p28_file))
        
        self.assertTrue(np.all(labels >= 1.0) and np.all(labels <= 9.0))

    def test_participant_split_config(self):
        """Verify that participant splits are strictly disjoint and cover all 32 subjects."""
        cfg = CONFIG.data_split
        train_set = set(cfg.train_participants)
        val_set = set(cfg.val_participants)
        test_set = set(cfg.test_participants)
        
        # Check disjointness
        self.assertTrue(train_set.isdisjoint(val_set), "Train and Val must be mutually disjoint!")
        self.assertTrue(train_set.isdisjoint(test_set), "Train and Test must be mutually disjoint!")
        self.assertTrue(val_set.isdisjoint(test_set), "Val and Test must be mutually disjoint!")
        
        # Check totals
        total = train_set.union(val_set).union(test_set)
        self.assertEqual(len(total), 32)
        self.assertEqual(min(total), 1)
        self.assertEqual(max(total), 32)


if __name__ == "__main__":
    unittest.main()
