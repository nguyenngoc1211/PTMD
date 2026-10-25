"""Checks for the external-dataset boundary and isolation of experiment outputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import pandas as pd
from run_dataset_experiments import ROOT, normalize_dataset


class DatasetExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'source.csv'
        self.frame = pd.DataFrame({'target': ['0', '1'] * 20,
                                   'memory.a': range(40), 'memory.b': [i*i for i in range(40)],
                                   'sample_id': range(100, 140), 'extra_label': ['secret'] * 40})
        self.frame.to_csv(self.path, index=False)
        self.spec = dict(dataset_id='fixture', source='synthetic test',
                         collection_context='unit test only', label_column='target',
                         label_map={'0': 'Benign', '1': 'Malware'},
                         feature_columns=['memory.a', 'memory.b'])

    def test_only_allowlisted_features_and_explicit_labels(self):
        frame, cases, ignored = normalize_dataset(self.path, self.spec)
        self.assertEqual(frame.columns.tolist(), ['memory.a', 'memory.b', 'Category', 'Class'])
        self.assertEqual(frame.Class.tolist(), ['Benign', 'Malware'] * 20)
        self.assertEqual(ignored, ['extra_label', 'sample_id'])
        self.assertEqual(cases, ['memory.a', 'memory.b'])

    def test_unknown_labels_fail_instead_of_becoming_benign(self):
        self.frame.loc[0, 'target'] = 'unknown'
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            normalize_dataset(self.path, self.spec)

    def test_label_cannot_be_selected_as_feature(self):
        self.spec['feature_columns'].append('target')
        with self.assertRaisesRegex(ValueError, 'reserved'):
            normalize_dataset(self.path, self.spec)

    def test_nonnumeric_values_fail_instead_of_becoming_missing(self):
        self.frame['memory.a'] = self.frame['memory.a'].astype(str)
        self.frame.loc[0, 'memory.a'] = 'broken'
        self.frame.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            normalize_dataset(self.path, self.spec)

    def test_duplicate_headers_rejected(self):
        self.path.write_text('target,memory.a,memory.a\n0,1,2\n1,3,4\n')
        with self.assertRaisesRegex(ValueError, 'duplicate column'):
            normalize_dataset(self.path, self.spec)

    def test_prepare_cli_isolates_outputs_and_refuses_overwrite(self):
        manifest = Path(self.tmp.name) / 'manifest.json'
        manifest.write_text(json.dumps(self.spec))
        output = Path(self.tmp.name) / 'runs'
        original_config = (ROOT / 'config.json').read_bytes()
        command = [sys.executable, str(ROOT / 'python/run_dataset_experiments.py'),
                   '--data', str(self.path), '--manifest', str(manifest),
                   '--output', str(output), '--seeds', '42,43', '--prepare-only']
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual((ROOT / 'config.json').read_bytes(), original_config)
        for seed in (42, 43):
            config = json.loads((output / f'seed_{seed}/config.json').read_text())
            self.assertEqual(config['seed'], seed)
            self.assertTrue((output / f'seed_{seed}/data/input.csv').exists())
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)


if __name__ == '__main__':
    unittest.main()
