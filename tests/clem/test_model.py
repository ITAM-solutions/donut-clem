"""
File name: test_model
Author: Fran Moreno
Last Updated: 10/30/2025
Version: 1.0
Description: TOFILL
"""
import unittest
from unittest.mock import patch, MagicMock
from parameterized import parameterized
from pydantic import BaseModel
from clem.model import DonutCLEM
from pathlib import Path


class DummyModel(BaseModel):
    a: int


class TestDonutCLEM(unittest.TestCase):

    @patch('clem.model.DonutCLEM._inference')
    @patch('clem.model.DonutCLEM._split_image_in_half')
    @patch('clem.model.DonutCLEM._load_model_from_local')
    @patch('clem.combination_logic.CandidateCollector')
    def test_predict_sectionMetadata_expectedValues(self,
            mock_candidate_collector,
            mock_load_model_from_local,
            mock_split_image_in_half,
            mock_inference,
    ):
        # Mock dependencies
        def inference_side_effect(*args, **kwargs):
            if inference_side_effect.call_count < 2:
                inference_side_effect.call_count += 1
                DummyModel(a="not_int")
            return MagicMock()

        inference_side_effect.call_count = 0
        mock_inference.side_effect = inference_side_effect
        mock_split_image_in_half.side_effect = lambda x: [x, x]
        mock_load_model_from_local.side_effect = lambda x: ("dummy_model", "dummy_task")
        mock_candidate_collector.add.side_effect = lambda *args, **kwargs: None

        donut_clem = DonutCLEM(Path('dummy/model/path'))
        donut_clem.predict(Path('dummy/im/path'), output=mock_candidate_collector)


if __name__ == '__main__':
    unittest.main()