from .threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator
from .stream_player import StreamPlayer, AudioConfiguration
from typing import Union, Iterator, List
from .engines import BaseEngine
import stream2sentence as s2s
import numpy as np
import threading
import traceback
import logging
import pyaudio
import queue
import time
import wave

class TextToAudioStream:
    def __init__(
        self,
        engine: Union[BaseEngine, List[BaseEngine]],
        log_characters: bool = False,
        on_text_stream_start=None,
        on_text_stream_stop=None,
        on_audio_stream_start=None,
        on_audio_stream_stop=None,
        on_character=None,
        output_device_index=None,
        tokenizer: str = "nltk",
        language: str = "en",
        muted: bool = False,
        level=logging.WARNING,
    ):

        self.log_characters = log_characters
        self.on_text_stream_start = on_text_stream_start
        self.on_text_stream_stop = on_text_stream_stop
        self.on_audio_stream_start = on_audio_stream_start
        self.on_audio_stream_stop = on_audio_stream_stop
        self.output_device_index = output_device_index
        self.output_wavfile = None
        self.chunk_callback = None
        self.wf = None
        self.abort_events = []
        self.tokenizer = tokenizer
        self.language = language
        self.global_muted = muted
        self.player = None
        self.play_lock = threading.Lock()
        self.is_playing_flag = False

        self._create_iterators()

        logging.info(f"Initializing tokenizer {tokenizer} " f"for language {language}")
        s2s.init_tokenizer(tokenizer, language)

        self.play_thread = None

        self.generated_text = ""

        self.stream_running = False

        self.on_character = on_character

        self.engine_index = 0
        if isinstance(engine, list):

            self.engines = engine
        else:

            self.engines = [engine]

        self.load_engine(self.engines[self.engine_index])

    def load_engine(self, engine: BaseEngine):

        self.engine = engine

        format, channels, rate = self.engine.get_stream_info()

        config = AudioConfiguration(
            format,
            channels,
            rate,
            self.output_device_index,
            muted=self.global_muted,
        )

        self.player = StreamPlayer(
            self.engine.queue, config, on_playback_start=self._on_audio_stream_start
        )

        logging.info(f"loaded engine {self.engine.engine_name}")

    def feed(self, text_or_iterator: Union[str, Iterator[str]]):

        self.char_iter.add(text_or_iterator)
        return self

    def play_async(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = True,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "",
        tokenize_sentences=None,
        language: str = "",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。",
        force_first_fragment_after_words=30,
        debug=False,
    ):

        if not self.is_playing_flag:
            self.is_playing_flag = True
            args = (
                fast_sentence_fragment,
                fast_sentence_fragment_allsentences,
                fast_sentence_fragment_allsentences_multiple,
                buffer_threshold_seconds,
                minimum_sentence_length,
                minimum_first_fragment_length,
                log_synthesized_text,
                reset_generated_text,
                output_wavfile,
                on_sentence_synthesized,
                before_sentence_synthesized,
                on_audio_chunk,
                tokenizer,
                tokenize_sentences,
                language,
                context_size,
                context_size_look_overhead,
                muted,
                sentence_fragment_delimiters,
                force_first_fragment_after_words,
                True,
                debug,
            )
            self.play_thread = threading.Thread(target=self.play, args=args)
            self.play_thread.start()
        else:
            logging.warning("play_async() called while already playing audio, skipping")

    def play(
        self,
        fast_sentence_fragment: bool = True,
        fast_sentence_fragment_allsentences: bool = False,
        fast_sentence_fragment_allsentences_multiple: bool = False,
        buffer_threshold_seconds: float = 0.0,
        minimum_sentence_length: int = 10,
        minimum_first_fragment_length: int = 10,
        log_synthesized_text=False,
        reset_generated_text: bool = True,
        output_wavfile: str = None,
        on_sentence_synthesized=None,
        before_sentence_synthesized=None,
        on_audio_chunk=None,
        tokenizer: str = "nltk",
        tokenize_sentences=None,
        language: str = "en",
        context_size: int = 12,
        context_size_look_overhead: int = 12,
        muted: bool = False,
        sentence_fragment_delimiters: str = ".?!;:,\n…。",
        force_first_fragment_after_words=30,
        is_external_call=True,
        debug=False,
    ):

        if self.global_muted:
            muted = True

        if is_external_call:
            if not self.play_lock.acquire(blocking=False):
                logging.warning("play() called while already playing audio, skipping")
                return

        self.is_playing_flag = True

        logging.info("stream start")

        tokenizer = tokenizer if tokenizer else self.tokenizer
        language = language if language else self.language

        self.stream_start_time = time.time()
        self.stream_running = True
        abort_event = threading.Event()
        self.abort_events.append(abort_event)

        if self.player:
            self.player.mute(muted)
        elif hasattr(self.engine, "set_muted"):
            self.engine.set_muted(muted)

        self.output_wavfile = output_wavfile
        self.chunk_callback = on_audio_chunk

        if output_wavfile:
            if self._is_engine_mpeg():
                self.wf = open(output_wavfile, "wb")
            else:
                self.wf = wave.open(output_wavfile, "wb")
                _, channels, rate = self.engine.get_stream_info()
                self.wf.setnchannels(channels)
                self.wf.setsampwidth(2)
                self.wf.setframerate(rate)

        if reset_generated_text:
            self.generated_text = ""

        if self.engine.can_consume_generators:
            try:

                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                self.char_iter.log_characters = self.log_characters

                self.engine.synthesize(self.char_iter)

            finally:

                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

                if is_external_call:
                    if self.on_audio_stream_stop:
                        self.on_audio_stream_stop()

                logging.info("stream stop")

                self.generated_text += self.char_iter.iterated_text

                self._create_iterators()

                if is_external_call:
                    self.is_playing_flag = False
                    self.play_lock.release()
        else:
            try:

                if self.player:
                    self.player.start()
                    self.player.on_audio_chunk = self._on_audio_chunk

                generate_sentences = s2s.generate_sentences(
                    self.thread_safe_char_iter,
                    context_size=context_size,
                    context_size_look_overhead=context_size_look_overhead,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    quick_yield_single_sentence_fragment=fast_sentence_fragment,
                    quick_yield_for_all_sentences=fast_sentence_fragment_allsentences,
                    quick_yield_every_fragment=fast_sentence_fragment_allsentences_multiple,
                    cleanup_text_links=True,
                    cleanup_text_emojis=True,
                    tokenize_sentences=tokenize_sentences,
                    tokenizer=tokenizer,
                    language=language,
                    log_characters=self.log_characters,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    debug=debug,
                )

                chunk_generator = self._synthesis_chunk_generator(
                    generate_sentences, buffer_threshold_seconds, log_synthesized_text
                )

                sentence_queue = queue.Queue()

                def synthesize_worker():
                    while not abort_event.is_set():
                        sentence = sentence_queue.get()
                        if sentence is None:  
                            break

                        synthesis_successful = False
                        if log_synthesized_text:
                            print(f"\033[96m\033[1m⚡ synthesizing\033[0m \033[37m→ \033[2m'\033[22m{sentence}\033[2m'\033[0m")

                        while not synthesis_successful:
                            try:
                                if abort_event.is_set():
                                    break

                                if before_sentence_synthesized:
                                    before_sentence_synthesized(sentence)
                                success = self.engine.synthesize(sentence)
                                if success:
                                    if on_sentence_synthesized:
                                        on_sentence_synthesized(sentence)
                                    synthesis_successful = True
                                else:
                                    logging.warning(
                                        f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}", unknown error'
                                    )

                            except Exception as e:
                                logging.warning(
                                    f'engine {self.engine.engine_name} failed to synthesize sentence "{sentence}" with error: {e}'
                                )
                                tb_str = traceback.format_exc()
                                print(f"Traceback: {tb_str}")
                                print(f"Error: {e}")

                            if not synthesis_successful:
                                if len(self.engines) == 1:
                                    time.sleep(0.2)
                                    logging.warning(
                                        f"engine {self.engine.engine_name} is the only engine available, can't switch to another engine"
                                    )
                                    break
                                else:
                                    logging.warning(
                                        "fallback engine(s) available, switching to next engine"
                                    )
                                    self.engine_index = (self.engine_index + 1) % len(
                                        self.engines
                                    )

                                    self.player.stop()
                                    self.load_engine(self.engines[self.engine_index])
                                    self.player.start()
                                    self.player.on_audio_chunk = self._on_audio_chunk

                        sentence_queue.task_done()

                worker_thread = threading.Thread(target=synthesize_worker)
                worker_thread.start()

                for sentence in chunk_generator:
                    if abort_event.is_set():
                        break
                    sentence = sentence.strip()
                    if sentence:
                        sentence_queue.put(sentence)
                    else:
                        continue  

                sentence_queue.put(None)
                worker_thread.join()

            except Exception as e:
                logging.warning(
                    f"error in play() with engine {self.engine.engine_name}: {e}"
                )
                tb_str = traceback.format_exc()
                print(f"Traceback: {tb_str}")
                print(f"Error: {e}")

            finally:
                try:
                    if self.player:
                        self.player.stop()

                    self.abort_events.remove(abort_event)
                    self.stream_running = False
                    logging.info("stream stop")

                    self.output_wavfile = None
                    self.chunk_callback = None

                finally:
                    if output_wavfile and self.wf:
                        self.wf.close()
                        self.wf = None

            if (len(self.char_iter.items) > 0
                and self.char_iter.iterated_text == ""
                and not self.char_iter.immediate_stop.is_set()):

                self.play(
                    fast_sentence_fragment=fast_sentence_fragment,
                    buffer_threshold_seconds=buffer_threshold_seconds,
                    minimum_sentence_length=minimum_sentence_length,
                    minimum_first_fragment_length=minimum_first_fragment_length,
                    log_synthesized_text=log_synthesized_text,
                    reset_generated_text=False,
                    output_wavfile=output_wavfile,
                    on_sentence_synthesized=on_sentence_synthesized,
                    on_audio_chunk=on_audio_chunk,
                    tokenizer=tokenizer,
                    language=language,
                    context_size=context_size,
                    muted=muted,
                    sentence_fragment_delimiters=sentence_fragment_delimiters,
                    force_first_fragment_after_words=force_first_fragment_after_words,
                    is_external_call=False,
                    debug=debug,
                )

            if is_external_call:
                if self.on_audio_stream_stop:
                    self.on_audio_stream_stop()

                self.is_playing_flag = False
                self.play_lock.release()

    def pause(self):

        if self.is_playing():
            logging.info("stream pause")
            self.player.pause()

    def resume(self):

        if self.is_playing():
            logging.info("stream resume")
            self.player.resume()

    def stop(self):

        for abort_event in self.abort_events:
            abort_event.set()

        if self.is_playing():
            self.char_iter.stop()
            self.player.stop(immediate=True)
            self.stream_running = False

        if self.play_thread is not None:
            if self.play_thread.is_alive():
                self.play_thread.join()
            self.play_thread = None

        self._create_iterators()

    def text(self):

        if self.generated_text:
            return self.generated_text
        return self.thread_safe_char_iter.accumulated_text()

    def is_playing(self):

        return self.stream_running

    def _on_audio_stream_start(self):

        latency = time.time() - self.stream_start_time
        logging.info(f"Audio stream start, latency to first chunk: {latency:.2f}s")

        if self.on_audio_stream_start:
            self.on_audio_stream_start()

    def _on_audio_chunk(self, chunk):

        format, _, _ = self.engine.get_stream_info()

        if format == pyaudio.paFloat32:
            audio_data = np.frombuffer(chunk, dtype=np.float32)
            audio_data = np.int16(audio_data * 32767)
            chunk = audio_data.tobytes()

        if self.output_wavfile and self.wf:
            if self._is_engine_mpeg():
                self.wf.write(chunk)
            else:
                self.wf.writeframes(chunk)

        if self.chunk_callback:
            self.chunk_callback(chunk)

    def _on_last_character(self):

        if self.on_text_stream_stop:
            self.on_text_stream_stop()

        if self.log_characters:
            print()

        self._create_iterators()

    def _create_iterators(self):

        self.char_iter = CharIterator(
            on_character=self._on_character,
            on_first_text_chunk=self.on_text_stream_start,
            on_last_text_chunk=self._on_last_character,
        )

        self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)

    def _on_character(self, char: str):

        if self.on_character:
            self.on_character(char)

        self.generated_text += char

    def _is_engine_mpeg(self):

        format, channel, rate = self.engine.get_stream_info()
        return format == pyaudio.paCustomFormat and channel == -1 and rate == -1

    def _synthesis_chunk_generator(
        self,
        generator: Iterator[str],
        buffer_threshold_seconds: float = 2.0,
        log_synthesis_chunks: bool = False,
    ) -> Iterator[str]:

        synthesis_chunk = ""

        for chunk in generator:

            if self.player:
                buffered_audio_seconds = self.player.get_buffered_seconds()
            else:
                buffered_audio_seconds = 0

            synthesis_chunk += chunk + " "

            if (
                buffered_audio_seconds < buffer_threshold_seconds
                or buffer_threshold_seconds <= 0
            ):

                if log_synthesis_chunks:
                    logging.info(
                        f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:1f}s'
                    )

                yield synthesis_chunk
                synthesis_chunk = ""

            else:
                logging.info(
                    f"summing up chunks because buffer {buffered_audio_seconds:.1f} > threshold ({buffer_threshold_seconds:.1f}s)"
                )

        if synthesis_chunk:

            if log_synthesis_chunks:
                logging.info(
                    f'-- ["{synthesis_chunk}"], buffered {buffered_audio_seconds:.1f}s'
                )

            yield synthesis_chunk