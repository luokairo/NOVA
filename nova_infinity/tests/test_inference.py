"""Exercise image return and CLI saving without loading model checkpoints."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('nova_runner_under_test', ROOT / 'tools/run_infinity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.amp, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(module, 'encode_prompt', lambda *args: ())
    return module


class ImageModel:
    def autoregressive_infer_cfg(self, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs['cfg_list'] == [4] * len(kwargs['scale_schedule'])
        return [], [], torch.full((1, 4, 4, 3), 37, dtype=torch.uint8), [], [None]


def test_image_return_contract(runner):
    args = dict(infinity_test=ImageModel(), vae=None, text_tokenizer=None, text_encoder=None,
                prompt='test', cfg_list=4, tau_list=0.5, scale_schedule=[(1, 1, 1)])
    image = runner.gen_one_img(**args)
    assert image.shape == (4, 4, 3) and image.dtype == torch.uint8
    image_with_indices, indices = runner.gen_one_img(**args, return_indices=True)
    torch.testing.assert_close(image_with_indices, image)
    assert indices == [None]


def test_cli_saves_to_requested_path(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'load_tokenizer', lambda **kwargs: (None, None))
    monkeypatch.setattr(runner, 'load_visual_tokenizer', lambda args: None)
    monkeypatch.setattr(runner, 'load_transformer', lambda *args: ImageModel())
    output = tmp_path / 'chosen' / 'image.png'
    runner.main(['--pn', '1M', '--model_path', 'unused', '--cfg', '4', '--save_file', str(output)])
    image = cv2.imread(str(output))
    assert image.shape == (4, 4, 3)
    assert (image == 37).all()


@pytest.mark.parametrize('flags,enabled', [
    ([], True), (['--skip-last-two-scales'], True), (['--no-skip-last-two-scales'], False),
])
def test_skip_scale_option_reaches_model(runner, monkeypatch, flags, enabled):
    parser = runner.argparse.ArgumentParser()
    runner.add_common_arguments(parser)
    args = parser.parse_args(['--pn', '1M', '--model_path', 'unused'] + flags)
    monkeypatch.setattr(runner, 'load_infinity', lambda **kwargs: SimpleNamespace(unregistered_blocks=[]))
    model = runner.load_transformer(None, args)
    assert model.skip_last_two_scales is enabled
