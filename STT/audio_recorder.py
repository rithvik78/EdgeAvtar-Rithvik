from typing import Iterable, List, Optional, Union
import torch.multiprocessing as mp
import numpy as np
from ctypes import c_bool
from openwakeword.model import Model
from scipy.signal import resample
from scipy import signal
import signal as system_signal
import torch, faster_whisper, openwakeword, collections, pvporcupine, traceback, threading, webrtcvad, itertools, datetime, platform, pyaudio, logging, struct, base64, queue, halo, time, copy, os, re, gc


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

INIT_MODEL_TRANSCRIPTION = "tiny"
INIT_MODEL_TRANSCRIPTION_REALTIME = "tiny"
INIT_REALTIME_PROCESSING_PAUSE = 0.2
INIT_REALTIME_INITIAL_PAUSE = 0.2
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_POST_SPEECH_SILENCE_DURATION = 0.6
INIT_MIN_LENGTH_OF_RECORDING = 0.5
INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_PRE_RECORDING_BUFFER_DURATION = 1.0
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

TIME_SLEEP = 0.02
SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True

class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe, model_path, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens):
        self.conn = conn
        self.stdout_pipe = stdout_pipe
        self.model_path = model_path
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.queue = queue.Queue()

    def custom_print(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    logging.error(f"Error receiving data from connection: {e}")
            else:
                time.sleep(TIME_SLEEP)

    def run(self):
        if __name__ == "__main__":
             system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
             __builtins__['print'] = self.custom_print

        logging.info(f"Initializing faster_whisper main transcription model {self.model_path}")

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
            )
        except Exception as e:
            logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
            raise

        self.ready_event.set()
        logging.debug("Faster_whisper main speech to text transcription model initialized successfully")

        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        segments, info = model.transcribe(
                            audio,
                            language=language if language else None,
                            beam_size=self.beam_size,
                            initial_prompt=self.initial_prompt,
                            suppress_tokens=self.suppress_tokens
                        )
                        transcription = " ".join(seg.text for seg in segments).strip()
                        logging.debug(f"Final text detected with main model: {transcription}")
                        self.conn.send(('success', (transcription, info)))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}")
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}")
        finally:
            __builtins__['print'] = print  
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  
            polling_thread.join()  

class bcolors:
    OKGREEN = '\033[92m'  
    WARNING = '\033[93m'  
    ENDC = '\033[0m'      

class AudioToTextRecorder:

    def __init__(self,
                 model: str = INIT_MODEL_TRANSCRIPTION,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 device: str = "cuda",
                 on_recording_start=None,
                 on_recording_stop=None,
                 on_transcription_start=None,
                 ensure_sentence_starting_uppercase=True,
                 ensure_sentence_ends_with_period=True,
                 use_microphone=True,
                 spinner=True,
                 level=logging.WARNING,

                 enable_realtime_transcription=False,
                 use_main_model_for_realtime=False,
                 realtime_model_type=INIT_MODEL_TRANSCRIPTION_REALTIME,
                 realtime_processing_pause=INIT_REALTIME_PROCESSING_PAUSE,
                 init_realtime_after_seconds=INIT_REALTIME_INITIAL_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
                 silero_use_onnx: bool = False,
                 silero_deactivity_detection: bool = False,
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
                 post_speech_silence_duration: float = (
                     INIT_POST_SPEECH_SILENCE_DURATION
                 ),
                 min_length_of_recording: float = (
                     INIT_MIN_LENGTH_OF_RECORDING
                 ),
                 min_gap_between_recordings: float = (
                     INIT_MIN_GAP_BETWEEN_RECORDINGS
                 ),
                 pre_recording_buffer_duration: float = (
                     INIT_PRE_RECORDING_BUFFER_DURATION
                 ),
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 wakeword_backend: str = "pvporcupine",
                 openwakeword_model_paths: str = None,
                 openwakeword_inference_framework: str = "onnx",
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wake_word_activation_delay: float = (
                    INIT_WAKE_WORD_ACTIVATION_DELAY
                 ),
                 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
                 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
                 on_wakeword_detected=None,
                 on_wakeword_timeout=None,
                 on_wakeword_detection_start=None,
                 on_wakeword_detection_end=None,
                 on_recorded_chunk=None,
                 debug_mode=False,
                 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = BUFFER_SIZE,
                 sample_rate: int = SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens: Optional[List[int]] = [-1],
                 print_transcription_time: bool = False,
                 early_transcription_on_silence: int = 0,
                 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
                 no_log_file: bool = False,
                 use_extended_logging: bool = False,
                 ):

        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index
        self.device = device
        self.wake_words = wake_words
        self.wake_word_activation_delay = wake_word_activation_delay
        self.wake_word_timeout = wake_word_timeout
        self.wake_word_buffer_duration = wake_word_buffer_duration
        self.ensure_sentence_starting_uppercase = (
            ensure_sentence_starting_uppercase
        )
        self.ensure_sentence_ends_with_period = (
            ensure_sentence_ends_with_period
        )
        self.use_microphone = mp.Value(c_bool, use_microphone)
        self.min_gap_between_recordings = min_gap_between_recordings
        self.min_length_of_recording = min_length_of_recording
        self.pre_recording_buffer_duration = pre_recording_buffer_duration
        self.post_speech_silence_duration = post_speech_silence_duration
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_wakeword_detected = on_wakeword_detected
        self.on_wakeword_timeout = on_wakeword_timeout
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_wakeword_detection_start = on_wakeword_detection_start
        self.on_wakeword_detection_end = on_wakeword_detection_end
        self.on_recorded_chunk = on_recorded_chunk
        self.on_transcription_start = on_transcription_start
        self.enable_realtime_transcription = enable_realtime_transcription
        self.use_main_model_for_realtime = use_main_model_for_realtime
        self.main_model_type = model
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.init_realtime_after_seconds = init_realtime_after_seconds
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.debug_mode = debug_mode
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.allowed_latency_limit = allowed_latency_limit

        self.level = level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_stop_time = 0
        self.wake_word_detect_time = 0
        self.silero_check_time = 0
        self.silero_working = False
        self.speech_end_silence_start = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_deactivity_detection = silero_deactivity_detection
        self.listen_start = 0
        self.spinner = spinner
        self.halo = None
        self.state = "inactive"
        self.wakeword_detected = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.stream = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.last_transcription_bytes = None
        self.last_transcription_bytes_b64 = None
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.print_transcription_time = print_transcription_time
        self.early_transcription_on_silence = early_transcription_on_silence
        self.use_extended_logging = use_extended_logging

        log_format = 'RealTimeSTT: %(name)s - %(levelname)s - %(message)s'

        file_log_format = '%(asctime)s.%(msecs)03d - ' + log_format

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  

        logger.handlers = []

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level) 
        console_handler.setFormatter(logging.Formatter(log_format))

        if not no_log_file:

            file_handler = logging.FileHandler('realtimesst.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                file_log_format,
                datefmt='%Y-%m-%d %H:%M:%S'
            ))

            logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.is_shut_down = False
        self.shutdown_event = mp.Event()

        try:

            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            logging.info(f"Start method has already been set. Details: {e}")

        logging.info("Starting RealTimeSTT")

        if use_extended_logging:
            logging.info("RealtimeSTT was called with these parameters:")
            for param, value in locals().items():
                logging.info(f"{param}: {value}")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.main_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()

        self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

        self.transcript_process = self._start_thread(
            target=AudioToTextRecorder._transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                model,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.main_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens
            )
        )

        if self.use_microphone.value:
            logging.info("Initializing audio recording"
                         " (creating pyAudio input stream,"
                         f" sample rate: {self.sample_rate}"
                         f" buffer size: {self.buffer_size}"
                         )
            self.reader_process = self._start_thread(
                target=AudioToTextRecorder._audio_data_worker,
                args=(
                    self.audio_queue,
                    self.sample_rate,
                    self.buffer_size,
                    self.input_device_index,
                    self.shutdown_event,
                    self.interrupt_stop_event,
                    self.use_microphone
                )
            )

        if self.enable_realtime_transcription and not self.use_main_model_for_realtime:
            try:
                logging.info("Initializing faster_whisper realtime "
                             f"transcription model {self.realtime_model_type}"
                             )
                self.realtime_model_type = faster_whisper.WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index
                )

            except Exception as e:
                logging.exception("Error initializing faster_whisper "
                                  f"realtime transcription model: {e}"
                                  )
                raise

            logging.debug("Faster_whisper realtime speech to text "
                          "transcription model initialized successfully")

        if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            self.wakeword_backend = wakeword_backend

            self.wake_words_list = [
                word.strip() for word in wake_words.lower().split(',')
            ]
            self.wake_words_sensitivity = wake_words_sensitivity
            self.wake_words_sensitivities = [
                float(wake_words_sensitivity)
                for _ in range(len(self.wake_words_list))
            ]

            if self.wakeword_backend in {'pvp', 'pvporcupine'}:

                try:
                    self.porcupine = pvporcupine.create(
                        keywords=self.wake_words_list,
                        sensitivities=self.wake_words_sensitivities
                    )
                    self.buffer_size = self.porcupine.frame_length
                    self.sample_rate = self.porcupine.sample_rate

                except Exception as e:
                    logging.exception(
                        "Error initializing porcupine "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logging.debug(
                    "Porcupine wake word detection engine initialized successfully"
                )

            elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:

                openwakeword.utils.download_models()

                try:
                    if openwakeword_model_paths:
                        model_paths = openwakeword_model_paths.split(',')
                        self.owwModel = Model(
                            wakeword_models=model_paths,
                            inference_framework=openwakeword_inference_framework
                        )
                        logging.info(
                            "Successfully loaded wakeword model(s): "
                            f"{openwakeword_model_paths}"
                        )
                    else:
                        self.owwModel = Model(
                            inference_framework=openwakeword_inference_framework)

                    self.oww_n_models = len(self.owwModel.models.keys())
                    if not self.oww_n_models:
                        logging.error(
                            "No wake word models loaded."
                        )

                    for model_key in self.owwModel.models.keys():
                        logging.info(
                            "Successfully loaded openwakeword model: "
                            f"{model_key}"
                        )

                except Exception as e:
                    logging.exception(
                        "Error initializing openwakeword "
                        f"wake word detection engine: {e}"
                    )
                    raise

                logging.debug(
                    "Open wake word detection engine initialized successfully"
                )

            else:
                logging.exception(f"Wakeword engine {self.wakeword_backend} unknown/unsupported. Please specify one of: pvporcupine, openwakeword.")

        try:
            logging.info("Initializing WebRTC voice with "
                         f"Sensitivity {webrtc_sensitivity}"
                         )
            self.webrtc_vad_model = webrtcvad.Vad()
            self.webrtc_vad_model.set_mode(webrtc_sensitivity)

        except Exception as e:
            logging.exception("Error initializing WebRTC voice "
                              f"activity detection engine: {e}"
                              )
            raise

        logging.debug("WebRTC VAD voice activity detection "
                      "engine initialized successfully"
                      )

        try:
            self.silero_vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                verbose=False,
                onnx=silero_use_onnx
            )

        except Exception as e:
            logging.exception(f"Error initializing Silero VAD "
                              f"voice activity detection engine: {e}"
                              )
            raise

        logging.debug("Silero VAD voice activity detection "
                      "engine initialized successfully"
                      )

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_duration)
        )
        self.last_words_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       0.3)
        )
        self.frames = []

        self.is_recording = False
        self.is_running = True
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False

        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()

        logging.debug('Waiting for main transcription model to start')
        self.main_transcription_ready_event.wait()
        logging.debug('Main transcription model ready')

        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        logging.debug('RealtimeSTT initialization completed successfully')

    def _start_thread(self, target=None, args=()):

        if (platform.system() == 'Linux'):
            thread = threading.Thread(target=target, args=args)
            thread.deamon = True
            thread.start()
            return thread
        else:
            thread = mp.Process(target=target, args=args)
            thread.start()
            return thread

    def _read_stdout(self):
        while not self.shutdown_event.is_set():
            try:
                if self.parent_stdout_pipe.poll(0.1):
                    logging.debug("Receive from stdout pipe")
                    message = self.parent_stdout_pipe.recv()
                    logging.info(message)
            except (BrokenPipeError, EOFError, OSError):

                pass
            except KeyboardInterrupt:  
                logging.info("KeyboardInterrupt in read from stdout detected, exiting...")
                break
            except Exception as e:
                logging.error(f"Unexpected error in read from stdout: {e}")
                logging.error(traceback.format_exc())  
                break 
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    @staticmethod
    def _audio_data_worker(audio_queue,
                        target_sample_rate,
                        buffer_size,
                        input_device_index,
                        shutdown_event,
                        interrupt_stop_event,
                        use_microphone):

        import pyaudio
        import numpy as np
        from scipy import signal

        if __name__ == '__main__':
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        def get_highest_sample_rate(audio_interface, device_index):

            try:
                device_info = audio_interface.get_device_info_by_index(device_index)
                max_rate = int(device_info['defaultSampleRate'])

                if 'supportedSampleRates' in device_info:
                    supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
                    if supported_rates:
                        max_rate = max(supported_rates)

                return max_rate
            except Exception as e:
                logging.warning(f"Failed to get highest sample rate: {e}")
                return 48000  

        def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
            nonlocal input_device_index

            def validate_device(device_index):

                try:
                    device_info = audio_interface.get_device_info_by_index(device_index)
                    if not device_info.get('maxInputChannels', 0) > 0:
                        return False

                    test_stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=target_sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=device_index,
                        start=False  
                    )

                    test_stream.start_stream()
                    test_data = test_stream.read(chunk_size, exception_on_overflow=False)
                    test_stream.stop_stream()
                    test_stream.close()

                    if len(test_data) == 0:
                        return False

                    return True

                except Exception as e:
                    logging.debug(f"Device validation failed: {e}")
                    return False

            while not shutdown_event.is_set():
                try:

                    input_devices = []
                    for i in range(audio_interface.get_device_count()):
                        try:
                            device_info = audio_interface.get_device_info_by_index(i)
                            if device_info.get('maxInputChannels', 0) > 0:
                                input_devices.append(i)
                        except Exception:
                            continue

                    if not input_devices:
                        raise Exception("No input devices found")

                    if input_device_index is None or input_device_index not in input_devices:

                        try:
                            default_device = audio_interface.get_default_input_device_info()
                            if validate_device(default_device['index']):
                                input_device_index = default_device['index']
                        except Exception:

                            for device_index in input_devices:
                                if validate_device(device_index):
                                    input_device_index = device_index
                                    break
                            else:
                                raise Exception("No working input devices found")

                    if not validate_device(input_device_index):
                        raise Exception("Selected device validation failed")

                    stream = audio_interface.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        input_device_index=input_device_index,
                    )

                    logging.info(f"Microphone connected and validated (input_device_index: {input_device_index})")
                    return stream

                except Exception as e:
                    logging.error(f"Microphone connection failed: {e}. Retrying...")
                    input_device_index = None
                    time.sleep(3)  
                    continue

        def preprocess_audio(chunk, original_sample_rate, target_sample_rate):

            if isinstance(chunk, np.ndarray):

                if chunk.ndim == 2:
                    chunk = np.mean(chunk, axis=1)

                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)

                chunk = chunk.astype(np.int16)
            else:

                chunk = np.frombuffer(chunk, dtype=np.int16)

                if original_sample_rate != target_sample_rate:
                    num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
                    chunk = signal.resample(chunk, num_samples)
                    chunk = chunk.astype(np.int16)

            return chunk.tobytes()

        audio_interface = None
        stream = None
        device_sample_rate = None
        chunk_size = 1024  

        def setup_audio():  
            nonlocal audio_interface, stream, device_sample_rate, input_device_index
            try:
                if audio_interface is None:
                    audio_interface = pyaudio.PyAudio()
                if input_device_index is None:
                    try:
                        default_device = audio_interface.get_default_input_device_info()
                        input_device_index = default_device['index']
                    except OSError as e:
                        input_device_index = None

                sample_rates_to_try = [16000]  
                if input_device_index is not None:
                    highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
                    if highest_rate != 16000:
                        sample_rates_to_try.append(highest_rate)
                else:
                    sample_rates_to_try.append(48000)  

                for rate in sample_rates_to_try:
                    try:
                        device_sample_rate = rate
                        stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
                        if stream is not None:
                            logging.debug(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")

                            return True
                    except Exception as e:
                        logging.warning(f"Failed to initialize audio23 stream at {device_sample_rate} Hz: {e}")
                        continue

                raise Exception("Failed to initialize audio stream12 with all sample rates.")

            except Exception as e:
                logging.exception(f"Error initializing pyaudio audio recording: {e}")
                if audio_interface:
                    audio_interface.terminate()
                return False

        if not setup_audio():
            raise Exception("Failed to set up audio recording.")

        buffer = bytearray()
        silero_buffer_size = 2 * buffer_size  

        time_since_last_buffer_message = 0

        try:
            while not shutdown_event.is_set():
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)

                    if use_microphone.value:
                        processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
                        buffer += processed_data

                        while len(buffer) >= silero_buffer_size:

                            to_process = buffer[:silero_buffer_size]
                            buffer = buffer[silero_buffer_size:]

                            if time_since_last_buffer_message:
                                time_passed = time.time() - time_since_last_buffer_message
                                if time_passed > 1:
                                    logging.debug("_audio_data_worker writing audio data into queue.")
                                    time_since_last_buffer_message = time.time()
                            else:
                                time_since_last_buffer_message = time.time()

                            audio_queue.put(to_process)

                except OSError as e:
                    if e.errno == pyaudio.paInputOverflowed:
                        logging.warning("Input overflowed. Frame dropped.")
                    else:
                        logging.error(f"OSError during recording: {e}")

                        logging.error("Attempting to reinitialize the audio stream...")

                        try:
                            if stream:
                                stream.stop_stream()
                                stream.close()
                        except Exception as e:
                            pass

                        time.sleep(1)

                        if not setup_audio():
                            logging.error("Failed to reinitialize audio stream. Exiting.")
                            break
                        else:
                            logging.error("Audio stream reinitialized successfully.")
                    continue

                except Exception as e:
                    logging.error(f"Unknown error during recording: {e}")
                    tb_str = traceback.format_exc()
                    logging.error(f"Traceback: {tb_str}")
                    logging.error(f"Error: {e}")

                    logging.info("Attempting to reinitialize the audio stream...")
                    try:
                        if stream:
                            stream.stop_stream()
                            stream.close()
                    except Exception as e:
                        pass

                    time.sleep(1)

                    if not setup_audio():
                        logging.error("Failed to reinitialize audio stream. Exiting.")
                        break
                    else:
                        logging.info("Audio stream reinitialized successfully.")
                    continue

        except KeyboardInterrupt:
            interrupt_stop_event.set()
            logging.debug("Audio data worker process finished due to KeyboardInterrupt")
        finally:

            if buffer:
                audio_queue.put(bytes(buffer))

            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
            except Exception as e:
                pass
            if audio_interface:
                audio_interface.terminate()

    def wakeup(self):

        self.listen_start = time.time()

    def abort(self):
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self._set_state("inactive")
        self.interrupt_stop_event.set()
        self.was_interrupted.wait()
        self.was_interrupted.clear()

    def wait_audio(self):

        try:
            logging.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            if not self.is_recording and not self.frames:
                self._set_state("listening")
                self.start_recording_on_voice_activity = True

                logging.debug('Waiting for recording start')
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            if self.is_recording:
                self.stop_recording_on_voice_deactivity = True

                logging.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if (self.stop_recording_event.wait(timeout=0.02)):
                        break

            audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            self.audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            self.frames.clear()

            self.recording_stop_time = 0
            self.listen_start = 0

            self._set_state("inactive")

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise  

    def transcribe(self):

        self._set_state("transcribing")
        audio_copy = copy.deepcopy(self.audio)
        start_time = 0
        with self.transcription_lock:

            try:
                if self.transcribe_count == 0:
                    logging.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  
                    self.parent_transcription_pipe.send((audio_copy, self.language))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    logging.debug(F"Receive from parent_transcription_pipe after sendiung transcription request, transcribe_count: {self.transcribe_count}")
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allowed_to_early_transcribe = True
                self._set_state("inactive")
                if status == 'success':
                    segments, info = result
                    self.detected_language = info.language if info.language_probability > 0 else None
                    self.detected_language_probability = info.language_probability
                    self.last_transcription_bytes = copy.deepcopy(audio_copy)                    
                    self.last_transcription_bytes_b64 = base64.b64encode(self.last_transcription_bytes.tobytes()).decode('utf-8')
                    transcription = self._preprocess_output(segments)
                    end_time = time.time()  
                    transcription_time = end_time - start_time

                    if start_time:
                        if self.print_transcription_time:
                            print(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                        else:
                            logging.debug(f"Model {self.main_model_type} completed transcription in {transcription_time:.2f} seconds")
                    return transcription
                else:
                    logging.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logging.error(f"Error during transcription: {str(e)}")
                raise e

    def _process_wakeword(self, data):

        if self.wakeword_backend in {'pvp', 'pvporcupine'}:
            pcm = struct.unpack_from(
                "h" * self.buffer_size,
                data
            )
            porcupine_index = self.porcupine.process(pcm)
            if self.debug_mode:
                logging.info(f"wake words porcupine_index: {porcupine_index}")
            return self.porcupine.process(pcm)

        elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
            pcm = np.frombuffer(data, dtype=np.int16)
            prediction = self.owwModel.predict(pcm)
            max_score = -1
            max_index = -1
            wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
            self.wake_words_sensitivities
            if wake_words_in_prediction:
                for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
                    scores = list(self.owwModel.prediction_buffer[mdl])
                    if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
                        max_score = scores[-1]
                        max_index = idx
                if self.debug_mode:
                    logging.info(f"wake words oww max_index, max_score: {max_index} {max_score}")
                return max_index  
            else:
                if self.debug_mode:
                    logging.info(f"wake words oww_index: -1")
                return -1

        if self.debug_mode:        
            logging.info("wake words no match")

        return -1

    def text(self,
             on_transcription_finished=None,
             ):

        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()
        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  

        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                            args=(self.transcribe(),)).start()
        else:
            return self.transcribe()

    def start(self):

        if (time.time() - self.recording_stop_time
                < self.min_gap_between_recordings):
            logging.info("Attempted to start recording "
                         "too soon after stopping."
                         )
            return self

        logging.info("recording started")
        self._set_state("recording")
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames = []
        self.is_recording = True
        self.recording_start_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self):

        if (time.time() - self.recording_start_time
                < self.min_length_of_recording):
            logging.info("Attempted to stop recording "
                         "too soon after starting."
                         )
            return self

        logging.info("recording stopped")
        self.is_recording = False
        self.recording_stop_time = time.time()
        self.is_silero_speech_active = False
        self.is_webrtc_speech_active = False
        self.silero_check_time = 0
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def listen(self):

        self.listen_start = time.time()
        self._set_state("listening")
        self.start_recording_on_voice_activity = True

    def feed_audio(self, chunk, original_sample_rate=16000):

        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        if isinstance(chunk, np.ndarray):

            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            chunk = chunk.astype(np.int16)

            chunk = chunk.tobytes()

        self.buffer += chunk
        buf_size = 2 * self.buffer_size  

        while len(self.buffer) >= buf_size:

            to_process = self.buffer[:buf_size]
            self.buffer = self.buffer[buf_size:]

            self.audio_queue.put(to_process)

    def set_microphone(self, microphone_on=True):

        logging.info("Setting microphone to: " + str(microphone_on))
        self.use_microphone.value = microphone_on

    def shutdown(self):

        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mRealtimeSTT shutting down\033[0m")

            self.is_shut_down = True
            self.start_recording_event.set()
            self.stop_recording_event.set()

            self.shutdown_event.set()
            self.is_recording = False
            self.is_running = False

            logging.debug('Finishing recording thread')
            if self.recording_thread:
                self.recording_thread.join()

            logging.debug('Terminating reader process')

            if self.use_microphone.value:
                self.reader_process.join(timeout=10)

                if self.reader_process.is_alive():
                    logging.warning("Reader process did not terminate "
                                    "in time. Terminating forcefully."
                                    )
                    self.reader_process.terminate()

            logging.debug('Terminating transcription process')
            self.transcript_process.join(timeout=10)

            if self.transcript_process.is_alive():
                logging.warning("Transcript process did not terminate "
                                "in time. Terminating forcefully."
                                )
                self.transcript_process.terminate()

            self.parent_transcription_pipe.close()

            logging.debug('Finishing realtime thread')
            if self.realtime_thread:
                self.realtime_thread.join()

            if self.enable_realtime_transcription:
                if self.realtime_model_type:
                    del self.realtime_model_type
                    self.realtime_model_type = None
            gc.collect()

    def _recording_worker(self):

        if self.use_extended_logging:
            logging.debug('Debug: Entering try block')

        last_inner_try_time = 0
        try:
            if self.use_extended_logging:
                logging.debug('Debug: Initializing variables')
            time_since_last_buffer_message = 0
            was_recording = False
            delay_was_passed = False
            wakeword_detected_time = None
            wakeword_samples_to_remove = None
            self.allowed_to_early_transcribe = True

            if self.use_extended_logging:
                logging.debug('Debug: Starting main loop')

            while self.is_running:

                if last_inner_try_time:
                    last_processing_time = time.time() - last_inner_try_time
                    if last_processing_time > 0.1:
                        if self.use_extended_logging:
                            logging.warning('### WARNING: PROCESSING TOOK TOO LONG')
                last_inner_try_time = time.time()
                try:

                    try:
                        data = self.audio_queue.get(timeout=0.01)
                        self.last_words_buffer.append(data)
                    except queue.Empty:

                        if not self.is_running:
                            if self.use_extended_logging:
                                logging.debug('Debug: Not running, breaking loop')
                            break

                        continue

                    if self.use_extended_logging:
                        logging.debug('Debug: Checking for on_recorded_chunk callback')
                    if self.on_recorded_chunk:
                        if self.use_extended_logging:
                            logging.debug('Debug: Calling on_recorded_chunk')
                        self.on_recorded_chunk(data)

                    if self.use_extended_logging:
                        logging.debug('Debug: Checking if handle_buffer_overflow is True')
                    if self.handle_buffer_overflow:
                        if self.use_extended_logging:
                            logging.debug('Debug: Handling buffer overflow')

                        if (self.audio_queue.qsize() >
                                self.allowed_latency_limit):
                            if self.use_extended_logging:
                                logging.debug('Debug: Queue size exceeds limit, logging warnings')
                            logging.warning("Audio queue size exceeds "
                                            "latency limit. Current size: "
                                            f"{self.audio_queue.qsize()}. "
                                            "Discarding old audio chunks."
                                            )

                        if self.use_extended_logging:
                            logging.debug('Debug: Discarding old chunks if necessary')
                        while (self.audio_queue.qsize() >
                                self.allowed_latency_limit):

                            data = self.audio_queue.get()

                except BrokenPipeError:
                    logging.error("BrokenPipeError _recording_worker")
                    self.is_running = False
                    break

                if self.use_extended_logging:
                    logging.debug('Debug: Updating time_since_last_buffer_message')

                if time_since_last_buffer_message:
                    time_passed = time.time() - time_since_last_buffer_message
                    if time_passed > 1:
                        if self.use_extended_logging:
                            logging.debug("_recording_worker processing audio data")
                        time_since_last_buffer_message = time.time()
                else:
                    time_since_last_buffer_message = time.time()

                if self.use_extended_logging:
                    logging.debug('Debug: Initializing failed_stop_attempt')
                failed_stop_attempt = False

                if self.use_extended_logging:
                    logging.debug('Debug: Checking if not recording')
                if not self.is_recording:
                    if self.use_extended_logging:
                        logging.debug('Debug: Handling not recording state')

                    time_since_listen_start = (time.time() - self.listen_start
                                            if self.listen_start else 0)

                    wake_word_activation_delay_passed = (
                        time_since_listen_start >
                        self.wake_word_activation_delay
                    )

                    if self.use_extended_logging:
                        logging.debug('Debug: Handling wake-word timeout callback')

                    if wake_word_activation_delay_passed \
                            and not delay_was_passed:

                        if self.use_wake_words and self.wake_word_activation_delay:
                            if self.on_wakeword_timeout:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Calling on_wakeword_timeout')
                                self.on_wakeword_timeout()
                    delay_was_passed = wake_word_activation_delay_passed

                    if self.use_extended_logging:
                        logging.debug('Debug: Setting state and spinner text')

                    if not self.recording_stop_time:
                        if self.use_wake_words \
                                and wake_word_activation_delay_passed \
                                and not self.wakeword_detected:
                            if self.use_extended_logging:
                                logging.debug('Debug: Setting state to "wakeword"')
                            self._set_state("wakeword")
                        else:
                            if self.listen_start:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Setting state to "listening"')
                                self._set_state("listening")
                            else:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Setting state to "inactive"')
                                self._set_state("inactive")

                    if self.use_extended_logging:
                        logging.debug('Debug: Checking wake word conditions')
                    if self.use_wake_words and wake_word_activation_delay_passed:
                        try:
                            if self.use_extended_logging:
                                logging.debug('Debug: Processing wakeword')
                            wakeword_index = self._process_wakeword(data)

                        except struct.error:
                            logging.error("Error unpacking audio data "
                                        "for wake word processing.")
                            continue

                        except Exception as e:
                            logging.error(f"Wake word processing error: {e}")
                            continue

                        if self.use_extended_logging:
                            logging.debug('Debug: Checking if wake word detected')

                        if wakeword_index >= 0:
                            if self.use_extended_logging:
                                logging.debug('Debug: Wake word detected, updating variables')
                            self.wake_word_detect_time = time.time()
                            wakeword_detected_time = time.time()
                            wakeword_samples_to_remove = int(self.sample_rate * self.wake_word_buffer_duration)
                            self.wakeword_detected = True
                            if self.on_wakeword_detected:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Calling on_wakeword_detected')
                                self.on_wakeword_detected()

                    if self.use_extended_logging:
                        logging.debug('Debug: Checking voice activity conditions')

                    if ((not self.use_wake_words
                        or not wake_word_activation_delay_passed)
                            and self.start_recording_on_voice_activity) \
                            or self.wakeword_detected:

                        if self.use_extended_logging:
                            logging.debug('Debug: Checking if voice is active')
                        if self._is_voice_active():
                            if self.use_extended_logging:
                                logging.debug('Debug: Voice activity detected')
                            logging.info("voice activity detected")

                            if self.use_extended_logging:
                                logging.debug('Debug: Starting recording')
                            self.start()

                            self.start_recording_on_voice_activity = False

                            if self.use_extended_logging:
                                logging.debug('Debug: Adding buffered audio to frames')

                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()

                            if self.use_extended_logging:
                                logging.debug('Debug: Resetting Silero VAD model states')
                            self.silero_vad_model.reset_states()
                        else:
                            if self.use_extended_logging:
                                logging.debug('Debug: Checking voice activity')
                            data_copy = data[:]
                            self._check_voice_activity(data_copy)

                    if self.use_extended_logging:
                        logging.debug('Debug: Resetting speech_end_silence_start')
                    self.speech_end_silence_start = 0

                else:
                    if self.use_extended_logging:
                        logging.debug('Debug: Handling recording state')

                    if wakeword_samples_to_remove and wakeword_samples_to_remove > 0:
                        if self.use_extended_logging:
                            logging.debug('Debug: Removing wakeword samples')

                        samples_removed = 0
                        while wakeword_samples_to_remove > 0 and self.frames:
                            frame = self.frames[0]
                            frame_samples = len(frame) // 2  
                            if wakeword_samples_to_remove >= frame_samples:
                                self.frames.pop(0)
                                samples_removed += frame_samples
                                wakeword_samples_to_remove -= frame_samples
                            else:
                                self.frames[0] = frame[wakeword_samples_to_remove * 2:]
                                samples_removed += wakeword_samples_to_remove
                                samples_to_remove = 0

                        wakeword_samples_to_remove = 0

                    if self.use_extended_logging:
                        logging.debug('Debug: Checking if stop_recording_on_voice_deactivity is True')

                    if self.stop_recording_on_voice_deactivity:
                        if self.use_extended_logging:
                            logging.debug('Debug: Determining if speech is detected')
                        is_speech = (
                            self._is_silero_speech(data) if self.silero_deactivity_detection
                            else self._is_webrtc_speech(data, True)
                        )

                        if self.use_extended_logging:
                            logging.debug('Debug: Formatting speech_end_silence_start')
                        if not self.speech_end_silence_start:
                            str_speech_end_silence_start = "0"
                        else:
                            str_speech_end_silence_start = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]
                        if self.use_extended_logging:
                            logging.debug(f"is_speech: {is_speech}, str_speech_end_silence_start: {str_speech_end_silence_start}")

                        if self.use_extended_logging:
                            logging.debug('Debug: Checking if speech is not detected')
                        if not is_speech:
                            if self.use_extended_logging:
                                logging.debug('Debug: Handling voice deactivity')

                            if self.speech_end_silence_start == 0 and \
                                (time.time() - self.recording_start_time > self.min_length_of_recording):

                                self.speech_end_silence_start = time.time()

                            if self.use_extended_logging:
                                logging.debug('Debug: Checking early transcription conditions')
                            if self.speech_end_silence_start and self.early_transcription_on_silence and len(self.frames) > 0 and \
                                (time.time() - self.speech_end_silence_start > self.early_transcription_on_silence) and \
                                self.allowed_to_early_transcribe:
                                    if self.use_extended_logging:
                                        logging.debug("Debug:Adding early transcription request")
                                    self.transcribe_count += 1
                                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                    audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE

                                    if self.use_extended_logging:
                                        logging.debug("Debug: early transcription request pipe send")
                                    self.parent_transcription_pipe.send((audio, self.language))
                                    if self.use_extended_logging:
                                        logging.debug("Debug: early transcription request pipe send return")
                                    self.allowed_to_early_transcribe = False

                        else:
                            if self.use_extended_logging:
                                logging.debug('Debug: Handling speech detection')
                            if self.speech_end_silence_start:
                                if self.use_extended_logging:
                                    logging.info("Resetting self.speech_end_silence_start")
                                self.speech_end_silence_start = 0
                                self.allowed_to_early_transcribe = True

                        if self.use_extended_logging:
                            logging.debug('Debug: Checking if silence duration exceeds threshold')

                        if self.speech_end_silence_start and time.time() - \
                                self.speech_end_silence_start >= \
                                self.post_speech_silence_duration:

                            if self.use_extended_logging:
                                logging.debug('Debug: Formatting silence start time')

                            silence_start_time = datetime.datetime.fromtimestamp(self.speech_end_silence_start).strftime('%H:%M:%S.%f')[:-3]

                            if self.use_extended_logging:
                                logging.debug('Debug: Calculating time difference')

                            time_diff = time.time() - self.speech_end_silence_start

                            if self.use_extended_logging:
                                logging.debug('Debug: Logging voice deactivity detection')
                                logging.info(f"voice deactivity detected at {silence_start_time}, "
                                        f"time since silence start: {time_diff:.3f} seconds")

                                logging.debug('Debug: Appending data to frames and stopping recording')
                            self.frames.append(data)
                            self.stop()
                            if not self.is_recording:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Resetting speech_end_silence_start')
                                self.speech_end_silence_start = 0

                                if self.use_extended_logging:
                                    logging.debug('Debug: Handling non-wake word scenario')
                            else:
                                if self.use_extended_logging:
                                    logging.debug('Debug: Setting failed_stop_attempt to True')
                                failed_stop_attempt = True

                if self.use_extended_logging:
                    logging.debug('Debug: Checking if recording stopped')
                if not self.is_recording and was_recording:
                    if self.use_extended_logging:
                        logging.debug('Debug: Resetting after stopping recording')

                    self.stop_recording_on_voice_deactivity = False

                if self.use_extended_logging:
                    logging.debug('Debug: Checking Silero time')
                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                if self.use_extended_logging:
                    logging.debug('Debug: Handling wake word timeout')

                if self.wake_word_detect_time and time.time() - \
                        self.wake_word_detect_time > self.wake_word_timeout:

                    self.wake_word_detect_time = 0
                    if self.wakeword_detected and self.on_wakeword_timeout:
                        if self.use_extended_logging:
                            logging.debug('Debug: Calling on_wakeword_timeout')
                        self.on_wakeword_timeout()
                    self.wakeword_detected = False

                if self.use_extended_logging:
                    logging.debug('Debug: Updating was_recording')
                was_recording = self.is_recording

                if self.use_extended_logging:
                    logging.debug('Debug: Checking if recording and not failed stop attempt')
                if self.is_recording and not failed_stop_attempt:
                    if self.use_extended_logging:
                        logging.debug('Debug: Appending data to frames')
                    self.frames.append(data)

                if self.use_extended_logging:
                    logging.debug('Debug: Checking if not recording or speech end silence start')
                if not self.is_recording or self.speech_end_silence_start:
                    if self.use_extended_logging:
                        logging.debug('Debug: Appending data to audio buffer')
                    self.audio_buffer.append(data)

        except Exception as e:
            logging.debug('Debug: Caught exception in main try block')
            if not self.interrupt_stop_event.is_set():
                logging.error(f"Unhandled exeption in _recording_worker: {e}")
                raise

        if self.use_extended_logging:
            logging.debug('Debug: Exiting _recording_worker method')

    def _realtime_worker(self):

        try:

            logging.debug('Starting realtime worker')

            if not self.enable_realtime_transcription:
                return

            while self.is_running:

                if self.is_recording:

                    time.sleep(self.realtime_processing_pause)

                    audio_array = np.frombuffer(
                        b''.join(self.frames),
                        dtype=np.int16
                        )

                    logging.debug(f"Current realtime buffer size: {len(audio_array)}")

                    audio_array = audio_array.astype(np.float32) / \
                        INT16_MAX_ABS_VALUE

                    if self.use_main_model_for_realtime:
                        with self.transcription_lock:
                            try:
                                self.parent_transcription_pipe.send((audio_array, self.language))
                                if self.parent_transcription_pipe.poll(timeout=5):  
                                    logging.debug("Receive from realtime worker after transcription request to main model")
                                    status, result = self.parent_transcription_pipe.recv()
                                    if status == 'success':
                                        segments, info = result
                                        self.detected_realtime_language = info.language if info.language_probability > 0 else None
                                        self.detected_realtime_language_probability = info.language_probability
                                        realtime_text = segments
                                        logging.debug(f"Realtime text detected with main model: {realtime_text}")
                                    else:
                                        logging.error(f"Realtime transcription error: {result}")
                                        continue
                                else:
                                    logging.warning("Realtime transcription timed out")
                                    continue
                            except Exception as e:
                                logging.error(f"Error in realtime transcription: {str(e)}")
                                continue
                    else:

                        segments, info = self.realtime_model_type.transcribe(
                            audio_array,
                            language=self.language if self.language else None,
                            beam_size=self.beam_size_realtime,
                            initial_prompt=self.initial_prompt,
                            suppress_tokens=self.suppress_tokens,
                        )

                        self.detected_realtime_language = info.language if info.language_probability > 0 else None
                        self.detected_realtime_language_probability = info.language_probability
                        realtime_text = " ".join(
                            seg.text for seg in segments
                        )
                        logging.debug(f"Realtime text detected: {realtime_text}")

                    if self.is_recording and time.time() - \
                            self.recording_start_time > self.init_realtime_after_seconds:

                        self.realtime_transcription_text = realtime_text
                        self.realtime_transcription_text = \
                            self.realtime_transcription_text.strip()

                        self.text_storage.append(
                            self.realtime_transcription_text
                            )

                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]

                            prefix = os.path.commonprefix(
                                [last_two_texts[0], last_two_texts[1]]
                                )

                            if len(prefix) >= \
                                    len(self.realtime_stabilized_safetext):

                                self.realtime_stabilized_safetext = prefix

                        matching_pos = self._find_tail_match_in_text(
                            self.realtime_stabilized_safetext,
                            self.realtime_transcription_text
                            )

                        if matching_pos < 0:
                            if self.realtime_stabilized_safetext:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_stabilized_safetext,
                                        True
                                    )
                                )
                            else:
                                self._on_realtime_transcription_stabilized(
                                    self._preprocess_output(
                                        self.realtime_transcription_text,
                                        True
                                    )
                                )
                        else:

                            output_text = self.realtime_stabilized_safetext + \
                                self.realtime_transcription_text[matching_pos:]

                            self._on_realtime_transcription_stabilized(
                                self._preprocess_output(output_text, True)
                                )

                        self._on_realtime_transcription_update(
                            self._preprocess_output(
                                self.realtime_transcription_text,
                                True
                            )
                        )

                else:
                    time.sleep(TIME_SLEEP)

        except Exception as e:
            logging.error(f"Unhandled exeption in _realtime_worker: {e}")
            raise

    def _is_silero_speech(self, chunk):

        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        self.silero_working = True
        audio_chunk = np.frombuffer(chunk, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_chunk),
            SAMPLE_RATE).item()
        is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
        if is_silero_speech_active:
            if not self.is_silero_speech_active and self.use_extended_logging:
                logging.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
        elif self.is_silero_speech_active and self.use_extended_logging:
            logging.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
        self.is_silero_speech_active = is_silero_speech_active
        self.silero_working = False
        return is_silero_speech_active

    def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):

        speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
        silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(
                pcm_data, 16000, self.sample_rate)
            chunk = data_16000.astype(np.int16).tobytes()

        frame_length = int(16000 * 0.01)  
        num_frames = int(len(chunk) / (2 * frame_length))
        speech_frames = 0

        for i in range(num_frames):
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = chunk[start_byte:end_byte]
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames += 1
                if not all_frames_must_be_true:
                    if self.debug_mode:
                        logging.info(f"Speech detected in frame {i + 1}"
                              f" of {num_frames}")
                    if not self.is_webrtc_speech_active and self.use_extended_logging:
                        logging.info(speech_str)
                    self.is_webrtc_speech_active = True
                    return True
        if all_frames_must_be_true:
            if self.debug_mode and speech_frames == num_frames:
                logging.info(f"Speech detected in {speech_frames} of "
                      f"{num_frames} frames")
            elif self.debug_mode:
                logging.info(f"Speech not detected in all {num_frames} frames")
            speech_detected = speech_frames == num_frames
            if speech_detected and not self.is_webrtc_speech_active and self.use_extended_logging:
                logging.info(speech_str)
            elif not speech_detected and self.is_webrtc_speech_active and self.use_extended_logging:
                logging.info(silence_str)
            self.is_webrtc_speech_active = speech_detected
            return speech_detected
        else:
            if self.debug_mode:
                logging.info(f"Speech not detected in any of {num_frames} frames")
            if self.is_webrtc_speech_active and self.use_extended_logging:
                logging.info(silence_str)
            self.is_webrtc_speech_active = False
            return False

    def _check_voice_activity(self, data):

        self._is_webrtc_speech(data)

        if self.is_webrtc_speech_active:

            if not self.silero_working:
                self.silero_working = True

                threading.Thread(
                    target=self._is_silero_speech,
                    args=(data,)).start()

    def clear_audio_queue(self):

        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except:

            pass

    def _is_voice_active(self):

        return self.is_webrtc_speech_active and self.is_silero_speech_active

    def _set_state(self, new_state):

        if new_state == self.state:
            return

        old_state = self.state

        self.state = new_state

        logging.info(f"State changed from '{old_state}' to '{new_state}'")

        if old_state == "listening":
            if self.on_vad_detect_stop:
                self.on_vad_detect_stop()
        elif old_state == "wakeword":
            if self.on_wakeword_detection_end:
                self.on_wakeword_detection_end()

        if new_state == "listening":
            if self.on_vad_detect_start:
                self.on_vad_detect_start()
            self._set_spinner("speak now")
            if self.spinner and self.halo:
                self.halo._interval = 250
        elif new_state == "wakeword":
            if self.on_wakeword_detection_start:
                self.on_wakeword_detection_start()
            self._set_spinner(f"say {self.wake_words}")
            if self.spinner and self.halo:
                self.halo._interval = 500
        elif new_state == "transcribing":
            if self.on_transcription_start:
                self.on_transcription_start()
            self._set_spinner("transcribing")
            if self.spinner and self.halo:
                self.halo._interval = 50
        elif new_state == "recording":
            self._set_spinner("recording")
            if self.spinner and self.halo:
                self.halo._interval = 100
        elif new_state == "inactive":
            if self.spinner and self.halo:
                self.halo.stop()
                self.halo = None

    def _set_spinner(self, text):

        if self.spinner:

            if self.halo is None:
                self.halo = halo.Halo(text=text)
                self.halo.start()

            else:
                self.halo.text = text

    def _preprocess_output(self, text, preview=False):

        text = re.sub(r'\s+', ' ', text.strip())

        if self.ensure_sentence_starting_uppercase:
            if text:
                text = text[0].upper() + text[1:]

        if not preview:
            if self.ensure_sentence_ends_with_period:
                if text and text[-1].isalnum():
                    text += '.'

        return text

    def _find_tail_match_in_text(self, text1, text2, length_of_match=10):

        if len(text1) < length_of_match or len(text2) < length_of_match:
            return -1

        target_substring = text1[-length_of_match:]

        for i in range(len(text2) - length_of_match + 1):

            current_substring = text2[len(text2) - i - length_of_match:
                                      len(text2) - i]

            if current_substring == target_substring:

                return len(text2) - i

        return -1

    def _on_realtime_transcription_stabilized(self, text):

        if self.on_realtime_transcription_stabilized:
            if self.is_recording:
                self.on_realtime_transcription_stabilized(text)

    def _on_realtime_transcription_update(self, text):

        if self.on_realtime_transcription_update:
            if self.is_recording:
                self.on_realtime_transcription_update(text)

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.shutdown()