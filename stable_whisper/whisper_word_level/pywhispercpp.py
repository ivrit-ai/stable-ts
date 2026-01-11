from types import MethodType
from typing import Union, Optional, Callable

import numpy as np
from tqdm import tqdm

from ..result import Segment, WhisperResult
from ..non_whisper import transcribe_any
from ..utils import safe_print, isolate_useful_options
from ..audio import audioloader_not_supported, convert_demucs_kwargs

from ..whisper_compatibility import LANGUAGES

SAMPLE_RATE = 16000


def whispercpp_transcribe(
        model: "Model",
        audio: Union[str, bytes, np.ndarray],
        *,
        word_timestamps: bool = True,
        verbose: Optional[bool] = False,
        regroup: Union[bool, str] = True,
        suppress_silence: bool = True,
        suppress_word_ts: bool = True,
        use_word_position: bool = True,
        q_levels: int = 20,
        k_size: int = 5,
        denoiser: Optional[str] = None,
        denoiser_options: Optional[dict] = None,
        demucs: bool = False,
        demucs_options: dict = None,
        vad: Union[bool, dict] = False,
        vad_threshold: float = 0.35,
        vad_onnx: bool = False,
        min_word_dur: Optional[float] = None,
        min_silence_dur: Optional[float] = None,
        nonspeech_error: float = 0.1,
        only_voice_freq: bool = False,
        only_ffmpeg: bool = False,
        check_sorted: bool = True,
        progress_callback: Callable = None,
        **options
) -> WhisperResult:
    """
    Transcribe audio using pywhispercpp (https://github.com/abdeladim-s/pywhispercpp).

    This uses the transcribe method from pywhispercpp, while still allowing additional preprocessing 
    and postprocessing. The preprocessing performed on the audio includes: voice isolation / noise 
    removal and low/high-pass filter. The postprocessing performed on the transcription result 
    includes: adjusting timestamps with VAD and custom regrouping segments based punctuation and 
    speech gaps.

    Parameters
    ----------
    model : pywhispercpp.model.Model
        The pywhispercpp Model instance.
    audio : str or numpy.ndarray or torch.Tensor or bytes
        Path/URL to the audio file, the audio waveform, or bytes of audio file.
        If audio is :class:`numpy.ndarray` or :class:`torch.Tensor`, the audio must be already at sampled to 16kHz.
    verbose : bool or None, default False
        Whether to display the text being decoded to the console.
        Displays all the details if ``True``. Displays progressbar if ``False``. Display nothing if ``None``.
    word_timestamps : bool, default True
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.
        Disabling this will prevent segments from splitting/merging properly.
    regroup : bool or str, default True, meaning the default regroup algorithm
        String for customizing the regrouping algorithm. False disables regrouping.
        Ignored if ``word_timestamps = False``.
    suppress_silence : bool, default True
        Whether to enable timestamps adjustments based on the detected silence.
    suppress_word_ts : bool, default True
        Whether to adjust word timestamps based on the detected silence. Only enabled if ``suppress_silence = True``.
    use_word_position : bool, default True
        Whether to use position of the word in its segment to determine whether to keep end or start timestamps if
        adjustments are required. If it is the first word, keep end. Else if it is the last word, keep the start.
    q_levels : int, default 20
        Quantization levels for generating timestamp suppression mask; ignored if ``vad = true``.
        Acts as a threshold to marking sound as silent.
        Fewer levels will increase the threshold of volume at which to mark a sound as silent.
    k_size : int, default 5
        Kernel size for avg-pooling waveform to generate timestamp suppression mask; ignored if ``vad = true``.
        Recommend 5 or 3; higher sizes will reduce detection of silence.
    denoiser : str, optional
        String of the denoiser to use for preprocessing ``audio``.
        See ``stable_whisper.audio.SUPPORTED_DENOISERS`` for supported denoisers.
    denoiser_options : dict, optional
        Options to use for ``denoiser``.
    vad : bool or dict, default False
        Whether to use Silero VAD to generate timestamp suppression mask.
        Instead of ``True``, using a dict of keyword arguments will load the VAD with the arguments.
        Silero VAD requires PyTorch 1.12.0+. Official repo, https://github.com/snakers4/silero-vad.
    vad_threshold : float, default 0.35
        Threshold for detecting speech with Silero VAD. Low threshold reduces false positives for silence detection.
    min_word_dur : float or None, default None meaning use ``stable_whisper.default.DEFAULT_VALUES``
        Shortest duration each word is allowed to reach for silence suppression.
    min_silence_dur : float, optional
        Shortest duration of silence allowed for silence suppression.
    nonspeech_error : float, default 0.3
        Relative error of non-speech sections that appear in between a word for silence suppression.
    only_voice_freq : bool, default False
        Whether to only use sound between 200 - 5000 Hz, where majority of human speech are.
    only_ffmpeg : bool, default False
        Whether to use only FFmpeg (instead of not yt-dlp) for URls
    check_sorted : bool, default True
        Whether to raise an error when timestamps returned by pywhispercpp are not in ascending order.
    progress_callback : Callable, optional
        A function that will be called when transcription progress is updated.
        The callback need two parameters.
        The first parameter is a float for seconds of the audio that has been transcribed.
        The second parameter is a float for total duration of audio in seconds.
    options
        Additional options used for :meth:`pywhispercpp.model.Model.transcribe` and
        :func:`stable_whisper.non_whisper.transcribe_any`.

    Returns
    -------
    stable_whisper.result.WhisperResult
        All timestamps, words, probabilities, and other data from the transcription of ``audio``.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_whispercpp('base.en')
    >>> result = model.transcribe('audio.mp3', vad=True)
    >>> result.to_srt_vtt('audio.srt')
    Saved: audio.srt
    """
    audioloader_not_supported(audio)
    extra_options = isolate_useful_options(options, transcribe_any, pop=True)
    denoiser, denoiser_options = convert_demucs_kwargs(
        denoiser, denoiser_options, demucs=demucs, demucs_options=demucs_options
    )
    
    if not isinstance(audio, (str, bytes)):
        if 'input_sr' not in extra_options:
            extra_options['input_sr'] = SAMPLE_RATE

    if denoiser or only_voice_freq:
        if 'audio_type' not in extra_options:
            extra_options['audio_type'] = 'str'
        if 'model_sr' not in extra_options:
            extra_options['model_sr'] = SAMPLE_RATE
    else:
        # pywhispercpp needs a file path
        if 'audio_type' not in extra_options:
            extra_options['audio_type'] = 'str'
    
    whispercpp_options = options
    whispercpp_options['model'] = model
    whispercpp_options['audio'] = audio
    whispercpp_options['word_timestamps'] = word_timestamps
    whispercpp_options['verbose'] = verbose
    whispercpp_options['progress_callback'] = progress_callback

    return transcribe_any(
        inference_func=_inner_transcribe,
        audio=audio,
        inference_kwargs=whispercpp_options,
        verbose=verbose,
        regroup=regroup,
        suppress_silence=suppress_silence,
        suppress_word_ts=suppress_word_ts,
        q_levels=q_levels,
        k_size=k_size,
        denoiser=denoiser,
        denoiser_options=denoiser_options,
        vad=vad,
        vad_threshold=vad_threshold,
        vad_onnx=vad_onnx,
        min_word_dur=min_word_dur,
        min_silence_dur=min_silence_dur,
        nonspeech_error=nonspeech_error,
        use_word_position=use_word_position,
        only_voice_freq=only_voice_freq,
        only_ffmpeg=only_ffmpeg,
        force_order=True,
        check_sorted=check_sorted,
        **extra_options
    )


def _inner_transcribe(model, audio, verbose, word_timestamps=True, progress_callback=None, **whispercpp_options):
    """
    Inner transcription function that interfaces with pywhispercpp.
    """
    if isinstance(audio, bytes):
        import io
        import tempfile
        import os
        # pywhispercpp needs a file path, so write bytes to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio)
            temp_path = f.name
        try:
            return _transcribe_file(model, temp_path, verbose, word_timestamps, progress_callback, **whispercpp_options)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    elif isinstance(audio, np.ndarray):
        import tempfile
        import os
        import soundfile as sf
        # Convert numpy array to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        try:
            sf.write(temp_path, audio, SAMPLE_RATE)
            return _transcribe_file(model, temp_path, verbose, word_timestamps, progress_callback, **whispercpp_options)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        # Assume it's already a file path
        return _transcribe_file(model, audio, verbose, word_timestamps, progress_callback, **whispercpp_options)


def _transcribe_file(model, audio_path, verbose, word_timestamps, progress_callback, **whispercpp_options):
    """
    Transcribe from a file path using pywhispercpp.
    """
    # Get language if specified
    language = whispercpp_options.pop('language', None)
    task = whispercpp_options.pop('task', 'transcribe')
    
    # Build parameters dict for pywhispercpp
    transcribe_params = {}
    
    # Map common parameters to pywhispercpp format
    if language:
        transcribe_params['language'] = language
    
    # pywhispercpp uses n_threads, translate, etc.
    # Pass through any remaining options
    for key in ['n_threads', 'translate', 'no_context', 'single_segment', 
                'max_tokens', 'speed_up', 'audio_ctx', 'suppress_blank',
                'suppress_non_speech_tokens', 'temperature', 'max_initial_ts',
                'length_penalty', 'temperature_inc', 'entropy_thold',
                'logprob_thold', 'no_speech_thold']:
        if key in whispercpp_options:
            transcribe_params[key] = whispercpp_options[key]
    
    # Handle task parameter - pywhispercpp uses 'translate' boolean
    if task == 'translate':
        transcribe_params['translate'] = True
    
    if verbose is not None:
        model_name = getattr(model, 'model_name', 'unknown')
        print(f'Detected Language: {language if language else "auto"}')
        print(f'Transcribing with pywhispercpp ({model_name})...\r', end='')

    # Call the pywhispercpp transcribe method
    # The API returns a list of segments
    # Use transcribe_original which should be the original pywhispercpp method
    if not hasattr(model, 'transcribe_original'):
        raise RuntimeError("Model does not have transcribe_original method. "
                         "This should have been set by load_whispercpp().")
    segments_list = model.transcribe_original(audio_path, **transcribe_params)
    
    # Process segments
    final_segments = []
    total_duration = 0
    
    # Track progress if we can estimate duration
    task_name = task.title() if task else 'Transcribe'
    
    # pywhispercpp segments have: t0, t1, text attributes (times in centiseconds)
    # and optionally a words list with word, t0, t1, probability
    for seg in segments_list:
        # pywhispercpp uses t0/t1 in centiseconds (1/100th of a second)
        start_time = seg.t0 / 100.0
        end_time = seg.t1 / 100.0
        text = seg.text
            
        segment_dict = {
            'start': start_time,
            'end': end_time,
            'text': text,
        }
        
        # Add word-level timestamps if available
        if word_timestamps and hasattr(seg, 'words') and seg.words:
            segment_dict['words'] = []
            for word in seg.words:
                # Words also use t0/t1 in centiseconds
                word_start = word.t0 / 100.0
                word_end = word.t1 / 100.0
                word_text = word.word if hasattr(word, 'word') else word.text
                    
                word_dict = {
                    'word': word_text,
                    'start': word_start,
                    'end': word_end,
                    'probability': getattr(word, 'probability', 1.0)
                }
                segment_dict['words'].append(word_dict)
        
        if verbose:
            safe_print(Segment(**segment_dict, ignore_unused_args=True).to_display_str())
        
        final_segments.append(segment_dict)
        total_duration = max(total_duration, segment_dict['end'])
        
        if progress_callback is not None:
            progress_callback(segment_dict['end'], total_duration)
    
    if verbose:
        model_name = getattr(model, 'model_name', 'unknown')
        print(f'Completed transcription with pywhispercpp ({model_name}).')
    
    # Return in the expected format
    return dict(language=language if language else 'en', segments=final_segments)


def load_whispercpp(model_path: str, **model_init_options):
    """
    Load an instance of :class:`pywhispercpp.model.Model`.

    Parameters
    ----------
    model_path : str
        Path to the GGML model file (e.g., 'base.en', or path to .bin file).
    model_init_options
        Additional options to use for initialization of :class:`pywhispercpp.model.Model`.

    Returns
    -------
    pywhispercpp.model.Model
        A modified instance of :class:`pywhispercpp.model.Model`.

    Examples
    --------
    >>> import stable_whisper
    >>> model = stable_whisper.load_whispercpp('base.en')
    >>> result = model.transcribe('audio.mp3')
    >>> result.to_srt_vtt('audio.srt')
    """
    try:
        from pywhispercpp.model import Model
    except ImportError:
        raise ImportError(
            "pywhispercpp is not installed. "
            "Install it with: pip install pywhispercpp"
        )
    
    # Load the model
    whispercpp_model = Model(model_path, **model_init_options)
    whispercpp_model.model_name = model_path
    
    # Store original transcribe method - it should always exist
    if not hasattr(whispercpp_model, 'transcribe'):
        raise RuntimeError(
            f"pywhispercpp Model instance does not have a 'transcribe' method. "
            f"This is unexpected and may indicate an incompatible version of pywhispercpp."
        )
    whispercpp_model.transcribe_original = whispercpp_model.transcribe
    
    # Replace transcribe with our wrapped version
    whispercpp_model.transcribe = MethodType(whispercpp_transcribe, whispercpp_model)
    
    # Add alignment methods
    from ..alignment import align, align_words, refine
    whispercpp_model.align = MethodType(align, whispercpp_model)
    whispercpp_model.align_words = MethodType(align_words, whispercpp_model)
    whispercpp_model.refine = MethodType(refine, whispercpp_model)
    
    return whispercpp_model
