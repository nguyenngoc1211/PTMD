"""Regression checks for data separation, numerical stability and cross-language IDs."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from common import logloss_rows,inference,adjust_bh
from preprocess import prepare
from compute_cfi import aligned_values
from run_family_experiments import select_family


class PipelineTests(unittest.TestCase):
    def test_family_selection_excludes_other_malware_and_preserves_source_ids(self):
        df = pd.DataFrame({'Class': ['Benign', 'Malware', 'Malware', 'Malware'],
                           'Category': ['Benign', 'Ransomware-Conti-x', 'Spyware-Gator-x', 'Trojan-Zeus-x'],
                           'a': [1, 2, 3, 4]}, index=[10, 20, 30, 40])
        for family, row in [('Ransomware', 20), ('Spyware', 30), ('Trojan', 40)]:
            result = select_family(df, family)
            self.assertEqual(result.index.tolist(), [10, row])
        with self.assertRaises(ValueError):
            select_family(df, 'Unknown')
        with self.assertRaises(ValueError):
            select_family(df.iloc[[0]], 'Trojan')

    def test_float32_probabilities_are_clipped_in_float64(self):
        loss=logloss_rows([0,1],np.array([1,0],dtype=np.float32))
        self.assertTrue(np.isfinite(loss).all())
        self.assertTrue((loss>30).all())

    def test_zero_mean_does_not_imply_zero_uncertainty(self):
        result=inference([-1,1])
        self.assertAlmostEqual(result['p_value'],.5)
        self.assertLess(result['ci_lo'],0)
        self.assertGreater(result['se'],0)

    def test_t_inference_matches_scipy(self):
        values=[-.1,.2,.3,.5,.8]
        self.assertAlmostEqual(inference(values)['p_value'],ttest_1samp(values,0,alternative='greater').pvalue)
        self.assertEqual(inference([0,0])['p_value'],1)

    def test_bh_adjustment(self):
        np.testing.assert_allclose(adjust_bh([.01,.04,.03,.8]),[.04,.053333333333,.053333333333,.8])

    def test_deduplication_and_train_only_imputation(self):
        df=pd.DataFrame({'Class':['Benign','Malware']*20,'Category':['x']*40,
                         'a':np.arange(40,dtype=float),'b':np.arange(40,dtype=float)**2})
        df.loc[3,'b']=np.nan
        df=pd.concat([df,df.iloc[[0,1]]],ignore_index=True)
        split,profile,full=prepare(df)
        self.assertEqual(profile['duplicates_removed'],2)
        self.assertEqual(profile['train_test_duplicate_overlap'],0)
        expected=df.iloc[:40].iloc[split['train_indices']].b.median()
        self.assertEqual(split['imputation_medians']['b'],expected)
        self.assertEqual(full.loc[3,'b'],expected)

    def test_conflicting_duplicate_labels_rejected(self):
        df=pd.DataFrame({'Class':['Benign','Malware'],'Category':['x','y'],'a':[1,1]})
        with self.assertRaises(ValueError): prepare(df)

    def test_resample_alignment_and_duplicate_rejection(self):
        frame=pd.DataFrame({'test_position':[9,3,9,3],'replicate':[2,1,1,2],'a':[92,31,91,32]})
        np.testing.assert_array_equal(aligned_values(frame,'a',[3,9],2),[[31,32],[91,92]])
        frame.loc[0,['test_position','replicate']]=[3,1]
        with self.assertRaises(ValueError): aligned_values(frame,'a',[3,9],2)


if __name__ == '__main__': unittest.main()
