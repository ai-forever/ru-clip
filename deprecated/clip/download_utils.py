# -*- coding: utf-8 -*-
import os

from transformers.file_utils import (
    cached_path,
    hf_bucket_url,
    is_remote_url,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


def download_file_from_hf(file_name: str) -> str:
    pretrained_model_name_or_path = 'ai-forever/ru-clip'
    # Load model
    if pretrained_model_name_or_path is not None:
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, file_name)):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, file_name)
            else:
                raise EnvironmentError(
                    'Error no file named {} found in directory {}'.format(
                        file_name,
                        pretrained_model_name_or_path,
                    )
                )
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            archive_file = pretrained_model_name_or_path
        else:
            archive_file = hf_bucket_url(
                pretrained_model_name_or_path,
                filename=file_name,
                revision=None,
                mirror=None,
            )

        try:
            # Load from URL or cache if already cached
            resolved_archive_file = cached_path(
                archive_file,
                cache_dir=None,
                force_download=False,
                proxies=None,
                resume_download=False,
                local_files_only=False,
            )
        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on"
                f"'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a"
                f'file named one of {file_name}.\n\n'
            )
            raise EnvironmentError(msg)

        if resolved_archive_file == archive_file:
            logger.info('loading weights file {}'.format(archive_file))
        else:
            logger.info('loading weights file {} from cache at {}'.format(archive_file, resolved_archive_file))
    else:
        resolved_archive_file = None

    return resolved_archive_file
