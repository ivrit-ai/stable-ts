import warnings
from .cli import cli
from .original_whisper import transcribe_stable, transcribe_minimal, load_model, modify_model
from .faster_whisper import load_faster_whisper
from .hf_whisper import load_hf_whisper
from .mlx_whisper import load_mlx_whisper
from .pywhispercpp import load_whispercpp


__all__ = ['load_model', 'modify_model', 'load_faster_whisper', 'load_hf_whisper', 'load_mlx_whisper', 'load_whispercpp',]

warnings.filterwarnings('ignore', module='whisper', message='.*Triton.*', category=UserWarning)


if __name__ == '__main__':
    cli()
