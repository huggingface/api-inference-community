import json
import logging
import os

from huggingface_hub import file_download, hf_api, hf_hub_download, model_info, utils


logger = logging.getLogger(__name__)


class OfflineBestEffortMixin(object):
    def _hub_repo_file(self, repo_id, filename, repo_type="model"):
        if self.offline_preferred:
            try:
                config_file = hf_hub_download(
                    repo_id,
                    filename,
                    token=self.use_auth_token,
                    local_files_only=True,
                    repo_type=repo_type,
                )
            except utils.LocalEntryNotFoundError:
                logger.info("Unable to fetch model index in local cache")
            else:
                return config_file

        return hf_hub_download(
            repo_id, filename, token=self.use_auth_token, repo_type=repo_type
        )

    def _hub_model_info(self, model_id):
        """
        This method tries to fetch locally cached model_info if any.
        If none, it requests the Hub. Useful for pre cached private models when no token is available
        """
        if self.offline_preferred:
            cache_root = os.getenv(
                "DIFFUSERS_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", "")
            )
            folder_name = file_download.repo_folder_name(
                repo_id=model_id, repo_type="model"
            )
            folder_path = os.path.join(cache_root, folder_name)
            logger.debug("Cache folder path %s", folder_path)
            filename = os.path.join(folder_path, "hub_model_info.json")
            try:
                with open(filename, "r") as f:
                    model_data = json.load(f)
            except OSError:
                logger.info(
                    "No cached model info found in file %s found for model %s. Fetching on the hub",
                    filename,
                    model_id,
                )
            else:
                model_data = hf_api.ModelInfo(**model_data)
                return model_data
        model_data = model_info(model_id, token=self.use_auth_token)
        return model_data
