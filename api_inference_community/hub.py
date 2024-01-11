import json
import logging
import os
import pathlib
import re
from typing import List, Optional

from huggingface_hub import ModelCard, constants, hf_api, try_to_load_from_cache
from huggingface_hub.file_download import repo_folder_name


logger = logging.getLogger(__name__)


def _cached_repo_root_path(cache_dir: pathlib.Path, repo_id: str) -> pathlib.Path:
    folder = pathlib.Path(repo_folder_name(repo_id=repo_id, repo_type="model"))
    return cache_dir / folder


def cached_revision_path(cache_dir, repo_id, revision) -> pathlib.Path:

    error_msg = f"No revision path found for {repo_id}, revision {revision}"

    if revision is None:
        revision = "main"

    repo_cache = _cached_repo_root_path(cache_dir, repo_id)

    if not repo_cache.is_dir():
        msg = f"Local repo {repo_cache} does not exist"
        logger.error(msg)
        raise Exception(msg)

    refs_dir = repo_cache / "refs"
    snapshots_dir = repo_cache / "snapshots"

    # Resolve refs (for instance to convert main to the associated commit sha)
    if refs_dir.is_dir():
        revision_file = refs_dir / revision
        if revision_file.exists():
            with revision_file.open() as f:
                revision = f.read()

    # Check if revision folder exists
    if not snapshots_dir.exists():
        msg = f"No local revision path {snapshots_dir} found for {repo_id}, revision {revision}"
        logger.error(msg)
        raise Exception(msg)

    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        logger.error(error_msg)
        raise Exception(error_msg)

    return snapshots_dir / revision


def _build_offline_model_info(
    repo_id: str, cache_dir: pathlib.Path, revision: str
) -> hf_api.ModelInfo:

    logger.info("Rebuilding offline model info for repo %s", repo_id)

    # Let's rebuild some partial model info from what we see in cache, info extracted should be enough
    # for most use cases
    card_path = try_to_load_from_cache(
        repo_id=repo_id,
        filename="README.md",
        cache_dir=cache_dir,
        revision=revision,
    )
    if not isinstance(card_path, str):
        raise Exception(
            "Unable to rebuild offline model info, no README could be found"
        )

    card_path = pathlib.Path(card_path)
    logger.debug("Loading model card from model readme %s", card_path)
    model_card = ModelCard.load(card_path)
    card_data = model_card.data.to_dict()

    repo = card_path.parent
    logger.debug("Repo path %s", repo)
    siblings = _build_offline_siblings(repo)
    model_info = hf_api.ModelInfo(
        private=False,
        downloads=0,
        likes=0,
        id=repo_id,
        card_data=card_data,
        siblings=siblings,
        **card_data,
    )
    logger.info("Offline model info for repo %s: %s", repo, model_info)
    return model_info


def _build_offline_siblings(repo: pathlib.Path) -> List[dict]:
    siblings = []
    prefix_pattern = re.compile(r"^" + re.escape(str(repo)) + r"(.*)$")
    for root, dirs, files in os.walk(repo):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.stat(filepath).st_size
            m = prefix_pattern.match(filepath)
            if not m:
                msg = (
                    f"File {filepath} does not match expected pattern {prefix_pattern}"
                )
                logger.error(msg)
                raise Exception(msg)
            filepath = m.group(1)
            filepath = filepath.strip(os.sep)
            sibling = dict(rfilename=filepath, size=size)
            siblings.append(sibling)
    return siblings


def _cached_model_info(
    repo_id: str, revision: str, cache_dir: pathlib.Path
) -> hf_api.ModelInfo:
    """
    Looks for a json file containing prefetched model info in the revision path.
    If none found we just rebuild model info with the local directory files.
    Note that this file is not automatically created by hub_download/snapshot_download.
    It is just a convenience we add here, just in case the offline info we rebuild from
    the local directories would not cover all use cases.
    """
    revision_path = cached_revision_path(cache_dir, repo_id, revision)
    model_info_basename = "hub_model_info.json"
    model_info_path = revision_path / model_info_basename
    logger.info("Checking if there are some cached model info at %s", model_info_path)
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            o = json.load(f)
        r = hf_api.ModelInfo(**o)
        logger.debug("Cached model info from file: %s", r)
    else:
        logger.debug(
            "No cached model info file %s found, "
            "rebuilding partial model info from cached model files",
            model_info_path,
        )
        # Let's rebuild some partial model info from what we see in cache, info extracted should be enough
        # for most use cases
        r = _build_offline_model_info(repo_id, cache_dir, revision)

    return r


def hub_model_info(
    repo_id: str,
    revision: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
    **kwargs,
) -> hf_api.ModelInfo:
    """
    Get Hub model info with offline support
    """
    if revision is None:
        revision = "main"

    if not constants.HF_HUB_OFFLINE:
        return hf_api.model_info(repo_id=repo_id, revision=revision, **kwargs)

    logger.info("Model info for offline mode")

    if cache_dir is None:
        cache_dir = pathlib.Path(constants.HF_HUB_CACHE)

    return _cached_model_info(repo_id, revision, cache_dir)
