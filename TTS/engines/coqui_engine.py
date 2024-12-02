import traceback, requests, logging, pyaudio, torch, time, json, sys, io, os, re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union, List
import torch.multiprocessing as mp
from .base_engine import BaseEngine
from threading import Lock, Thread
from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.layers.xtts.xtts_manager import SpeakerManager

TIME_SLEEP_DEVICE_RESET = 2

class QueueWriter(io.TextIOBase):

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def write(self, msg):

        if msg.strip():  
            self.queue.put(msg)

class CoquiVoice:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name}"

class CoquiEngine(BaseEngine):
    def __init__(
        self,
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        specific_model="v2.0.2",
        local_models_path=None,
        voices_path=None,
        voice: Union[str, List[str]] = "",
        language="en",
        speed=1.0,
        thread_count=6,
        stream_chunk_size=20,
        overlap_wav_len=1024,
        temperature=0.85,
        length_penalty=1.0,
        repetition_penalty=7.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=True,
        full_sentences=False,
        level=logging.WARNING,
        use_deepspeed=False,
        device: str = None,
        prepare_text_for_synthesis_callback=None,
        add_sentence_filter=False,
        pretrained=False,
        comma_silence_duration=0.3,
        sentence_silence_duration=0.6,
        default_silence_duration=0.3,
        print_realtime_factor=False,
        load_balancing=False,
        load_balancing_buffer_length=0,
        load_balancing_cut_off=0,
    ):

        self._synthesize_lock = Lock()
        self.model_name = model_name
        self.pretrained = pretrained
        self.language = language
        self.level = level
        self.thread_count = thread_count
        self.stream_chunk_size = stream_chunk_size
        self.overlap_wav_len = overlap_wav_len
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.enable_text_splitting = enable_text_splitting
        self.full_sentences = full_sentences
        self.use_deepspeed = use_deepspeed
        self.device = device
        self.add_sentence_filter = add_sentence_filter
        self.comma_silence_duration = comma_silence_duration
        self.sentence_silence_duration = sentence_silence_duration
        self.default_silence_duration = default_silence_duration
        self.print_realtime_factor = print_realtime_factor
        self.load_balancing = load_balancing
        self.load_balancing_buffer_length = load_balancing_buffer_length
        self.load_balancing_cut_off = load_balancing_cut_off

        self.cloning_reference_wav = voice
        self.speed = speed
        self.specific_model = specific_model
        if not local_models_path:
            local_models_path = os.environ.get("COQUI_MODEL_PATH")
            if local_models_path and len(local_models_path) > 0:
                logging.info(
                    "Local models path from environment variable "
                    f'COQUI_MODEL_PATH: "{local_models_path}"'
                )
        self.local_models_path = local_models_path
        self.prepare_text_callback = prepare_text_for_synthesis_callback

        self.voices_path = voices_path

        self.model_path = None
        if not self.specific_model:

            logging.info("Download most recent XTTS Model if available")
            ModelManager().download_model(model_name)
        else:
            logging.info(f'Local XTTS Model: "{specific_model}" specified')
            self.model_path = self.download_model(
                specific_model, self.local_models_path
            )

        try:

            if sys.platform.startswith('linux') or sys.platform == 'darwin':  
                mp.set_start_method("spawn")
            elif mp.get_start_method(allow_none=True) is None:
                mp.set_start_method("spawn")
        except RuntimeError as e:
            print("Start method has already been set. Details:", e)

        self.create_worker_process()

    def create_worker_process(self):
        self.output_queue = mp.Queue()

        def output_worker(queue):

            while True:
                message = queue.get()
                if message == "STOP":  
                    break
                print(message)

        self.output_worker_thread = Thread(
            target=output_worker, args=(self.output_queue,)
        )
        self.output_worker_thread.daemon = True
        self.output_worker_thread.start()

        self.main_synthesize_ready_event = mp.Event()
        self.parent_synthesize_pipe, child_synthesize_pipe = mp.Pipe()
        self.voices_list = []
        self.retrieve_coqui_voices()

        self.synthesize_process = mp.Process(
            target=CoquiEngine._synthesize_worker,
            args=(
                self.output_queue,
                child_synthesize_pipe,
                self.model_name,
                self.cloning_reference_wav,
                self.language,
                self.main_synthesize_ready_event,
                self.level,
                self.speed,
                self.thread_count,
                self.stream_chunk_size,
                self.full_sentences,
                self.overlap_wav_len,
                self.temperature,
                self.length_penalty,
                self.repetition_penalty,
                self.top_k,
                self.top_p,
                self.enable_text_splitting,
                self.model_path,
                self.use_deepspeed,
                self.device,
                self.voices_path,
                self.voices_list,
                self.pretrained,
                self.comma_silence_duration,
                self.sentence_silence_duration,
                self.default_silence_duration,
                self.print_realtime_factor,
                self.load_balancing,
                self.load_balancing_buffer_length,
                self.load_balancing_cut_off,
            ),
        )
        self.synthesize_process.start()

        logging.debug("Waiting for coqui model start")
        self.main_synthesize_ready_event.wait()
        logging.info("Coqui synthesis model ready")

    def post_init(self):
        self.engine_name = "coqui"

    @staticmethod
    def _synthesize_worker(
        output_queue,
        conn,
        model_name,
        cloning_reference_wav: Union[str, List[str]],
        language,
        ready_event,
        loglevel,
        speed,
        thread_count,
        stream_chunk_size,
        full_sentences,
        overlap_wav_len,
        temperature,
        length_penalty,
        repetition_penalty,
        top_k,
        top_p,
        enable_text_splitting,
        local_model_path,
        use_deepspeed,
        device,
        voices_path,
        predefined_voices,
        pretrained,
        comma_silence_duration,
        sentence_silence_duration,
        default_silence_duration,
        print_realtime_factor,
        load_balancing,
        load_balancing_buffer_length,
        load_balancing_cut_off,
    ):

        sys.stdout = QueueWriter(output_queue)
        sys.stderr = QueueWriter(output_queue)
        
        tts = None

        logging.basicConfig(format="CoquiEngine: %(message)s", level=loglevel)

        logging.info("Starting CoquiEngine")

        def get_conditioning_latents(filenames: Union[str, List[str]], tts):

            if not isinstance(filenames, list):
                filenames = [filenames]

            logging.debug("Computing speaker latents")

            if len(filenames) == 0 or not filenames[0]:
                logging.debug("Using coqui_default_voice.wav as default voice")
                filenames = ["coqui_default_voice.wav"]

            if len(filenames) == 1:
                logging.debug("Handling of single voice file")

                filename = filenames[0]
                if filename.endswith(".json"):
                    filename_json = filename
                    filename = filename[:-5]
                    filename_wav = filename + "wav"
                elif filename.endswith(".wav"):
                    filename_json = filename[:-3] + "json"
                    filename = filename[:-3]
                    filename_wav = filename + "wav"
                else:
                    filename_json = filename + ".json"
                    filename_wav = filename + ".wav"

                if voices_path:
                    filename_voice_wav = os.path.join(voices_path, filename_wav)
                    filename_voice_json = os.path.join(voices_path, filename_json)
                else:
                    filename_voice_wav = filename_wav
                    filename_voice_json = filename_json

                if not os.path.exists(filename_voice_json) and not os.path.exists(
                    filename_voice_wav
                ):
                    for predefined_voice in predefined_voices:
                        if filename.lower() in predefined_voice.lower():
                            speaker_file_path = os.path.join(
                                checkpoint, "speakers_xtts.pth"
                            )
                            speaker_manager = SpeakerManager(speaker_file_path)
                            gpt_cond_latent, speaker_embedding = (
                                speaker_manager.speakers[predefined_voice].values()
                            )
                            return gpt_cond_latent, speaker_embedding

                    if len(filename) > 0:
                        logging.info(
                            f"Using default voice, both {filename_voice_json} and {filename_voice_wav} not found."
                        )
                    else:
                        logging.info(
                            "Using default voice, no cloning source specified."
                        )

                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    filename_voice_json = os.path.join(
                        current_dir, "coqui_default_voice.json"
                    )
                    if not os.path.exists(filename_voice_json):
                        raise ValueError(
                            f"Default voice file {filename_voice_json} not found."
                        )

                if os.path.exists(filename_voice_json):
                    logging.debug(
                        f"Latents already computed, reading from {filename_voice_json}"
                    )
                    with open(filename_voice_json, "r") as new_file:
                        latents = json.load(new_file)

                    speaker_embedding = (
                        torch.tensor(latents["speaker_embedding"])
                        .unsqueeze(0)
                        .unsqueeze(-1)
                    )
                    gpt_cond_latent = (
                        torch.tensor(latents["gpt_cond_latent"])
                        .reshape((-1, 1024))
                        .unsqueeze(0)
                    )

                    return gpt_cond_latent, speaker_embedding

                logging.debug(f"Computing latents for {filename}")

                gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(
                    audio_path=filename_voice_wav, gpt_cond_len=30, max_ref_length=60
                )

                latents = {
                    "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                    "speaker_embedding": speaker_embedding.cpu()
                    .squeeze()
                    .half()
                    .tolist(),
                }
                with open(filename_voice_json, "w") as new_file:
                    json.dump(latents, new_file)

                return gpt_cond_latent, speaker_embedding

            else:
                audio_path_list = []
                for filename in filenames:

                    if filename.endswith(".wav"):
                        if voices_path:
                            filename_voice_wav = os.path.join(voices_path, filename)
                        else:
                            filename_voice_wav = filename
                        audio_path_list.append(filename_voice_wav)
                        logging.debug(
                            f"Added {filename_voice_wav} (#{len(audio_path_list)}) to audio_path_list"
                        )

                if len(audio_path_list) == 0:
                    logging.info(
                        "Using default female voice, no cloning source specified."
                    )

                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    filename_voice_json = os.path.join(
                        current_dir, "coqui_default_voice.json"
                    )
                    if not os.path.exists(filename_voice_json):
                        raise ValueError(
                            f"Default voice file {filename_voice_json} not found."
                        )

                logging.debug(f"Computing latents for {filename}")

                gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(
                    audio_path=audio_path_list, gpt_cond_len=30, max_ref_length=60
                )

                latents = {
                    "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
                    "speaker_embedding": speaker_embedding.cpu()
                    .squeeze()
                    .half()
                    .tolist(),
                }
                filename_voice_json = audio_path_list[0][:-3] + "json"
                with open(filename_voice_json, "w") as new_file:
                    json.dump(latents, new_file)

                return gpt_cond_latent, speaker_embedding

        def postprocess_wave(chunk):

            if isinstance(chunk, list):
                chunk = torch.cat(chunk, dim=0)
            chunk = chunk.clone().detach().cpu().numpy()
            chunk = chunk[None, : int(chunk.shape[0])]
            chunk = np.clip(chunk, -1, 1)
            chunk = chunk.astype(np.float32)
            return chunk

        def load_model(checkpoint, tts):
            global config
            try:
                if tts:
                    import gc

                    del tts
                    torch.cuda.empty_cache()
                    gc.collect()
                    from numba import cuda

                    current_device = cuda.get_current_device()
                    current_device.reset()
                    tts = None
                    import time

                    time.sleep(TIME_SLEEP_DEVICE_RESET)

                torch.set_num_threads(int(str(thread_count)))
                torch_device = torch.device(
                    "cuda"
                    if device == "cuda" and torch.cuda.is_available()
                    else "mps"
                    if device == "mps"
                    and torch.backends.mps.is_available()
                    and torch.backends.mps.is_built()
                    else "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                config = load_config((os.path.join(checkpoint, "config.json")))
                tts = setup_tts_model(config)
                logging.debug(f"  xtts load_checkpoint({checkpoint})")

                tts.load_checkpoint(
                    config,
                    checkpoint_dir=checkpoint,
                    checkpoint_path=None,
                    vocab_path=None,
                    eval=True,
                    use_deepspeed=use_deepspeed,
                )
                tts.to(torch_device)
            except Exception as e:
                print(f"Error loading model for checkpoint {checkpoint}: {e}")
                raise
            return tts

        def get_user_data_dir(appname):
            TTS_HOME = os.environ.get("TTS_HOME")
            XDG_DATA_HOME = os.environ.get("XDG_DATA_HOME")
            if TTS_HOME is not None:
                ans = Path(TTS_HOME).expanduser().resolve(strict=False)
            elif XDG_DATA_HOME is not None:
                ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
            elif sys.platform == "win32":
                import winreg  

                key = winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
                )
                dir_, _ = winreg.QueryValueEx(key, "Local AppData")
                ans = Path(dir_).resolve(strict=False)
            elif sys.platform == "darwin":
                ans = Path("~/Library/Application Support/").expanduser()
            else:
                ans = Path.home().joinpath(".local/share")
            return ans.joinpath(appname)

        logging.debug(f"Initializing coqui model {model_name}")
        logging.debug(f" - cloning reference {cloning_reference_wav}")
        logging.debug(f" - language {language}")
        logging.debug(f" - user data dir {get_user_data_dir('tts')}")
        logging.debug(f" - local model path {local_model_path}")

        try:
            path_tts = os.path.join(
                get_user_data_dir("tts"), model_name.replace("/", "--")
            )
            checkpoint = local_model_path if local_model_path else path_tts
            logging.debug(f" - checkpoint {checkpoint}")
            tts = load_model(checkpoint, tts)

            gpt_cond_latent, speaker_embedding = get_conditioning_latents(
                cloning_reference_wav, tts
            )

        except Exception as e:
            logging.exception(f"Error initializing main coqui engine model: {e}")
            raise

        ready_event.set()

        logging.info("Coqui text to speech synthesize model initialized successfully")

        try:
            while True:
                timeout = 0.1
                if conn.poll(timeout):  
                    try:
                        message = conn.recv()
                    except Exception as e:
                        logging.error(
                            f"conn.recv() error: {e} occurred in the "
                            "synthesize worker thread of Coqui engine."
                        )
                        time.sleep(1)
                        continue
                else:

                    time.sleep(0.01)
                    continue

                command = message["command"]
                data = message["data"]

                if command == "update_reference":
                    new_wav_path = data["cloning_reference_wav"]
                    logging.info(f"Updating reference WAV to {new_wav_path}")
                    gpt_cond_latent, speaker_embedding = get_conditioning_latents(
                        new_wav_path, tts
                    )
                    conn.send(("success", "Reference updated successfully"))

                elif command == "set_speed":
                    speed = data["speed"]
                    conn.send(("success", "Speed updated successfully"))

                elif command == "set_model":
                    checkpoint = data["checkpoint"]
                    logging.info(f"Updating model checkpoint to {checkpoint}")
                    tts = load_model(checkpoint, tts)
                    conn.send(("success", "Model updated successfully"))

                elif command == "shutdown":
                    logging.info("Shutdown command received. Exiting worker process.")
                    conn.send(("shutdown", "shutdown"))
                    break  

                elif command == "synthesize":
                    text = data["text"]
                    language = data["language"]

                    logging.debug(f"Starting inference for text: {text}")

                    time_start = time.time()
                    seconds_to_first_chunk = 0.0
                    full_generated_seconds = 0.0
                    raw_inference_start = 0.0
                    first_chunk_length_seconds = 0.0

                    chunks = tts.inference_stream(
                        text,
                        language,
                        gpt_cond_latent,
                        speaker_embedding,
                        stream_chunk_size=stream_chunk_size,
                        overlap_wav_len=overlap_wav_len,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                        top_k=top_k,
                        top_p=top_p,
                        speed=speed,
                        enable_text_splitting=enable_text_splitting,
                    )

                    if full_sentences:
                        chunklist = []

                        for i, chunk in enumerate(chunks):
                            chunk = postprocess_wave(chunk)
                            chunk_bytes = chunk.tobytes()
                            chunklist.append(chunk_bytes)
                            chunk_duration = len(chunk_bytes) / (4 * 24000)
                            full_generated_seconds += chunk_duration
                            if i == 0:
                                first_chunk_length_seconds = chunk_duration
                                raw_inference_start = time.time()
                                seconds_to_first_chunk = (
                                    raw_inference_start - time_start
                                )

                        for i, chunk in enumerate(chunks):
                            chunk = postprocess_wave(chunk)
                            chunklist.append(chunk.tobytes())

                        for chunk in chunklist:
                            conn.send(("success", chunk))
                    else:
                        for i, chunk in enumerate(chunks):
                            chunk = postprocess_wave(chunk)
                            chunk_bytes = chunk.tobytes()

                            conn.send(("success", chunk_bytes))
                            chunk_duration = len(chunk_bytes) / (
                                4 * 24000
                            )  
                            full_generated_seconds += chunk_duration
                            if i == 0:
                                first_chunk_length_seconds = chunk_duration
                                raw_inference_start = time.time()
                                seconds_to_first_chunk = (
                                    raw_inference_start - time_start
                                )
                            else:
                                chunk_production_seconds = time.time() - time_start
                                generated_audio_seconds = full_generated_seconds

                                if load_balancing:
                                    if chunk_production_seconds < (generated_audio_seconds + load_balancing_buffer_length):
                                        waiting_time = generated_audio_seconds - chunk_production_seconds - load_balancing_cut_off
                                        if waiting_time > 0:
                                            print(f"Waiting for {waiting_time} seconds")
                                            time.sleep(waiting_time)

                    time_end = time.time()
                    seconds = time_end - time_start

                    if (
                        full_generated_seconds > 0
                        and (full_generated_seconds - first_chunk_length_seconds) > 0
                    ):
                        realtime_factor = seconds / full_generated_seconds
                        raw_inference_time = seconds - seconds_to_first_chunk
                        raw_inference_factor = raw_inference_time / (
                            full_generated_seconds - first_chunk_length_seconds
                        )
                        if print_realtime_factor:
                            print(f"Realtime Factor: {realtime_factor}")
                            print(f"Raw Inference Factor: {raw_inference_factor}")

                    sample_rate = config.audio.sample_rate

                    end_sentence_delimeters = ".!?…。¡¿"
                    mid_sentence_delimeters = ";:,\n()[]{}-“”„”—/|《》"

                    if text[-1] in end_sentence_delimeters:
                        silence_duration = sentence_silence_duration
                    elif text[-1] in mid_sentence_delimeters:
                        silence_duration = comma_silence_duration
                    else:
                        silence_duration = default_silence_duration

                    silent_samples = int(sample_rate * silence_duration)
                    silent_chunk = np.zeros(silent_samples, dtype=np.float32)
                    conn.send(("success", silent_chunk.tobytes()))

                    conn.send(("finished", ""))

                    end_time = time.time()

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received. " "Exiting worker process.")
            conn.send(("shutdown", "shutdown"))

        except Exception as e:
            logging.error(
                f"General synthesis error: {e} occured in "
                "synthesize worker thread of coqui engine."
            )

            tb_str = traceback.format_exc()
            print(f"Traceback: {tb_str}")
            print(f"Error: {e}")

            conn.send(("error", str(e)))

        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def send_command(self, command, data):

        message = {"command": command, "data": data}
        self.parent_synthesize_pipe.send(message)

    def set_cloning_reference(self, cloning_reference_wav: str):

        if not isinstance(cloning_reference_wav, list):
            cloning_reference_wav = [cloning_reference_wav]
        self.send_command(
            "update_reference", {"cloning_reference_wav": cloning_reference_wav}
        )

        status, result = self.parent_synthesize_pipe.recv()
        if status == "success":
            logging.info("Reference WAV updated successfully")
        else:
            logging.error(f"Error updating reference WAV: {cloning_reference_wav}")

        return status, result

    def set_speed(self, speed: float):

        self.send_command("set_speed", {"speed": speed})

        status, result = self.parent_synthesize_pipe.recv()
        if status == "success":
            logging.info("Speed updated successfully")
        else:
            logging.error("Error updating speed")

        return status, result

    def set_model(self, checkpoint: str):

        self.shutdown()

        self.specific_model = checkpoint
        self.model_path = self.download_model(checkpoint, self.local_models_path)

        self.create_worker_process()

    def get_stream_info(self):

        return pyaudio.paFloat32, 1, 24000

    def _prepare_text_for_synthesis(self, text: str):

        logging.debug(f'Preparing text for synthesis: "{text}"')

        if self.prepare_text_callback:
            return self.prepare_text_callback(text)

        text = text.strip()
        text = text.replace("</s>", "")

        text = re.sub("\\(.*?\\)", "", text, flags=re.DOTALL)
        text = text.replace("```", "")
        text = text.replace("...", " ")
        text = text.replace("»", "")
        text = text.replace("«", "")
        text = re.sub(" +", " ", text)

        try:
            if len(text) > 2 and text[-1] in ["."]:
                text = text[:-1]
            elif len(text) > 2 and text[-1] in ["!", "?", ","]:
                text = text[:-1] + " " + text[-1]
            elif len(text) > 3 and text[-2] in ["."]:
                text = text[:-2]
            elif len(text) > 3 and text[-2] in ["!", "?", ","]:
                text = text[:-2] + " " + text[-2]

        except Exception as e:
            logging.warning(
                f'Error fixing sentence end punctuation: {e}, Text: "{text}"'
            )

        text = text.strip()

        logging.debug(f'Text after preparation: "{text}"')

        return text

    def synthesize(self, text: str) -> bool:

        with self._synthesize_lock:
            if self.add_sentence_filter:
                text = self._prepare_text_for_synthesis(text)

            if len(text) < 1:
                return

            data = {"text": text, "language": self.language}
            self.send_command("synthesize", data)

            status, result = self.parent_synthesize_pipe.recv()

            while "finished" not in status:
                if "shutdown" in status or "error" in status:
                    if "error" in status:
                        logging.error(f"Error synthesizing text: {text}")
                        logging.error(f"Error: {result}")
                    return False

                self.queue.put(result)
                status, result = self.parent_synthesize_pipe.recv()

            return True

    @staticmethod
    def download_file(url, destination):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024

        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(destination, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

    @staticmethod
    def download_model(model_name="v2.0.2", local_models_path=None):

        if local_models_path and len(local_models_path) > 0:
            model_folder = os.path.join(local_models_path, f"{model_name}")
            logging.info(f'Local models path: "{model_folder}"')
        else:
            model_folder = os.path.join(os.getcwd(), "models", f"{model_name}")
            logging.info(
                f'Checking for models within application directory: "{model_folder}"'
            )

        os.makedirs(model_folder, exist_ok=True)

        files = {
            "config.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_name}/config.json",
            "model.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_name}/model.pth?download=true",
            "vocab.json": f"https://huggingface.co/coqui/XTTS-v2/raw/{model_name}/vocab.json",
            "speakers_xtts.pth": f"https://huggingface.co/coqui/XTTS-v2/resolve/{model_name}/speakers_xtts.pth",
        }

        for file_name, url in files.items():
            file_path = os.path.join(model_folder, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name} to {file_path}...")
                CoquiEngine.download_file(url, file_path)
                logging.info(f"{file_name} downloaded successfully.")
            else:
                logging.info(f"{file_name} exists in {file_path} (no download).")

        return model_folder

    def get_voices(self):

        voice_objects = []
        voices_appended = []

        files = os.listdir(self.voices_path)
        for file in files:

            if file.endswith(".wav"):
                file = file[:-4]
            elif file.endswith(".json"):
                file = file[:-5]
            else:
                continue

            if file in voices_appended:
                continue

            voices_appended.append(file)
            voice_objects.append(CoquiVoice(file))

        for voice in self.voices_list:
            voice_objects.append(CoquiVoice(voice))

        return voice_objects

    def set_voice(self, voice: str):

        if isinstance(voice, CoquiVoice):
            self.set_cloning_reference(voice.name)
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
                    self.set_cloning_reference(installed_voice.name)
                    return
            self.set_cloning_reference(voice)

    def set_voice_parameters(self, **voice_parameters):

        pass

    def shutdown(self):

        logging.info("Sending shutdown command to the worker process")
        self.send_command("shutdown", {})

        self.output_queue.put("STOP")
        self.output_worker_thread.join()

        try:
            status, _ = self.parent_synthesize_pipe.recv()
            if "shutdown" in status:
                logging.info("Worker process acknowledged shutdown")
        except EOFError:

            logging.warning(
                "Worker process pipe was closed before shutdown acknowledgement"
            )

        self.parent_synthesize_pipe.close()

        logging.info("Terminating the worker process")
        self.synthesize_process.terminate()

        self.synthesize_process.join()
        logging.info("Worker process has been terminated")

    def retrieve_coqui_voices(self):

        if self.pretrained:
            return self.voices_list

        from TTS.tts.layers.xtts.xtts_manager import SpeakerManager

        if self.local_models_path:
            local_models_path = self.local_models_path
        else:
            local_models_path = os.path.join(
                os.getcwd(), "models", f"{self.specific_model}"
            )

        speaker_file_path = os.path.join(local_models_path, "speakers_xtts.pth")
        if not os.path.exists(speaker_file_path):
            speaker_file_path = os.path.join(
                local_models_path, self.specific_model, "speakers_xtts.pth"
            )

        speaker_manager = SpeakerManager(speaker_file_path)
        self.voices_list = []
        for speaker_name in speaker_manager.name_to_id:
            self.voices_list.append(speaker_name)
        return self.voices_list