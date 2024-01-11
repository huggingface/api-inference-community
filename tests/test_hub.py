import logging
import sys
from unittest import TestCase

from api_inference_community import hub
from huggingface_hub import constants, hf_api, snapshot_download


logger = logging.getLogger(__name__)
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class HubTestCase(TestCase):
    def test_offline_model_info1(self):
        repo_id = "google/t5-efficient-tiny"
        revision = "3441d7e8bf3f89841f366d39452b95200416e4a9"
        bak_value = constants.HF_HUB_OFFLINE
        try:
            # with tempfile.TemporaryDirectory() as cache_dir:
            # logger.info("Cache directory %s", cache_dir)
            dirpath = snapshot_download(repo_id=repo_id, revision=revision)
            logger.info("Snapshot downloaded at %s", dirpath)
            constants.HF_HUB_OFFLINE = True
            model_info = hub.hub_model_info(repo_id=repo_id, revision=revision)
        finally:
            constants.HF_HUB_OFFLINE = bak_value

        logger.info("Model info %s", model_info)
        self.assertIsInstance(model_info, hf_api.ModelInfo)
        self.assertEqual(model_info.id, repo_id)
        self.assertEqual(model_info.downloads, 0)
        self.assertEqual(model_info.likes, 0)
        self.assertEqual(len(model_info.siblings), 12)
        self.assertIn("pytorch_model.bin", [s.rfilename for s in model_info.siblings])
        self.assertFalse(model_info.private)
        self.assertEqual(model_info.license, "apache-2.0")  # noqa
        self.assertEqual(model_info.tags, ["deep-narrow"])
        self.assertIsNone(model_info.library_name)

        logger.info("Model card data %s", model_info.card_data)
        self.assertEqual(model_info.card_data, model_info.cardData)
        self.assertEqual(model_info.card_data.license, "apache-2.0")
        self.assertEqual(model_info.card_data.tags, ["deep-narrow"])

    def test_offline_model_info2(self):
        repo_id = "dfurman/Mixtral-8x7B-peft-v0.1"
        revision = "8908d586219993ec79949acaef566363a7c7864c"
        bak_value = constants.HF_HUB_OFFLINE
        try:
            # with tempfile.TemporaryDirectory() as cache_dir:
            # logger.info("Cache directory %s", cache_dir)
            dirpath = snapshot_download(repo_id=repo_id, revision=revision)
            logger.info("Snapshot downloaded at %s", dirpath)
            constants.HF_HUB_OFFLINE = True
            model_info = hub.hub_model_info(repo_id=repo_id, revision=revision)
        finally:
            constants.HF_HUB_OFFLINE = bak_value

        logger.info("Model info %s", model_info)
        self.assertIsInstance(model_info, hf_api.ModelInfo)
        self.assertEqual(model_info.id, repo_id)
        self.assertEqual(model_info.downloads, 0)
        self.assertEqual(model_info.likes, 0)
        self.assertEqual(len(model_info.siblings), 9)
        self.assertFalse(model_info.private)
        self.assertEqual(model_info.license, "apache-2.0")  # noqa
        self.assertEqual(model_info.tags, ["mistral"])
        self.assertEqual(model_info.library_name, "peft")
        self.assertEqual(model_info.pipeline_tag, "text-generation")
        self.assertIn(".gitattributes", [s.rfilename for s in model_info.siblings])
        logger.info("Model card data %s", model_info.card_data)
        self.assertEqual(model_info.card_data, model_info.cardData)
        self.assertEqual(model_info.card_data.license, "apache-2.0")
        self.assertEqual(model_info.card_data.tags, ["mistral"])

    def test_online_model_info(self):
        repo_id = "dfurman/Mixtral-8x7B-Instruct-v0.1"
        revision = "8908d586219993ec79949acaef566363a7c7864c"
        bak_value = constants.HF_HUB_OFFLINE
        try:
            constants.HF_HUB_OFFLINE = False
            model_info = hub.hub_model_info(repo_id=repo_id, revision=revision)
        finally:
            constants.HF_HUB_OFFLINE = bak_value

        logger.info("Model info %s", model_info)
        self.assertIsInstance(model_info, hf_api.ModelInfo)
        self.assertEqual(model_info.id, repo_id)
        self.assertGreater(model_info.downloads, 0)
        self.assertGreater(model_info.likes, 0)
        self.assertEqual(len(model_info.siblings), 9)
        self.assertFalse(model_info.private)
        self.assertGreater(model_info.tags, ["peft", "safetensors", "mistral"])
        self.assertEqual(model_info.library_name, "peft")
        self.assertEqual(model_info.pipeline_tag, "text-generation")
        self.assertIn(".gitattributes", [s.rfilename for s in model_info.siblings])
        logger.info("Model card data %s", model_info.card_data)
        self.assertEqual(model_info.card_data, model_info.cardData)
        self.assertEqual(model_info.card_data.license, "apache-2.0")
        self.assertEqual(model_info.card_data.tags, ["mistral"])
        self.assertIsNone(model_info.safetensors)
