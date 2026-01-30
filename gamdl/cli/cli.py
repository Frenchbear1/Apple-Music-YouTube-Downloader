import asyncio
import logging
import sys
import shutil
import time
import threading
import os
from functools import wraps
from pathlib import Path

import click
import httpx
import httpcore
import colorama
from dataclass_click import dataclass_click

from .. import __version__
from ..api import AppleMusicApi, ItunesApi
from ..downloader import (
    AppleMusicBaseDownloader,
    AppleMusicDownloader,
    AppleMusicMusicVideoDownloader,
    AppleMusicSongDownloader,
    AppleMusicUploadedVideoDownloader,
    DownloadItem,
    DownloadMode,
    GamdlError,
    RemuxMode,
)
from ..downloader.constants import TEMP_PATH_TEMPLATE
from ..interface import (
    AppleMusicInterface,
    AppleMusicMusicVideoInterface,
    AppleMusicSongInterface,
    AppleMusicUploadedVideoInterface,
    SongCodec,
)
from .cli_config import CliConfig
from .config_file import ConfigFile
from .constants import X_NOT_IN_PATH
from .utils import CustomLoggerFormatter, prompt_path

logger = logging.getLogger(__name__)
_current_progress = None


class ProgressDisplay:
    def __init__(self, total: int = 100, label: str = "Overall", stream=None):
        self.total = max(1, total)
        self.label = label
        self.stream = stream or sys.stdout
        self.current = 0
        self.active = False
        self._lock = threading.RLock()

    def start(self):
        with self._lock:
            self.active = True
            self.render()

    def update(self, steps: int):
        if steps <= 0:
            return
        with self._lock:
            if not self.active:
                return
            self.current = min(self.total, self.current + steps)
            self.render()

    def clear(self):
        if not self.active:
            return
        self.stream.write("\r\x1b[2K")
        self.stream.flush()

    def render(self):
        if not self.active:
            return
        percent = int((self.current / self.total) * 100)
        width = 30
        filled = int(width * percent / 100)
        bar = "#" * filled + "-" * (width - filled)
        self.stream.write(f"\r{self.label} [{bar}] {percent:3d}%")
        self.stream.flush()

    def finish(self):
        with self._lock:
            if not self.active:
                return
            self.current = self.total
            self.render()
            self.stream.write("\n")
            self.stream.flush()
            self.active = False


class ProgressAwareStreamHandler(logging.StreamHandler):
    def emit(self, record):
        progress = _current_progress
        if progress and progress.active:
            progress.clear()
            super().emit(record)
            progress.render()
        else:
            super().emit(record)


def echo_stderr(message: str, progress: ProgressDisplay | None):
    if progress and progress.active:
        progress.clear()
    click.echo(message, file=sys.stderr)
    if progress and progress.active:
        progress.render()


def make_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@click.command()
@click.help_option("-h", "--help")
@click.version_option(__version__, "-v", "--version")
@dataclass_click(CliConfig)
@ConfigFile.loader
@make_sync
async def main(config: CliConfig):
    global _current_progress
    colorama.just_fix_windows_console()

    root_logger = logging.getLogger(__name__.split(".")[0])
    root_logger.setLevel(config.log_level)
    root_logger.propagate = False

    stream_handler = ProgressAwareStreamHandler(sys.stderr)
    stream_handler.setFormatter(CustomLoggerFormatter())
    root_logger.addHandler(stream_handler)

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setFormatter(CustomLoggerFormatter(use_colors=False))
        root_logger.addHandler(file_handler)

    logger.info(f"Starting Gamdl {__version__}")

    if config.use_wrapper:
        apple_music_api = await AppleMusicApi.create_from_wrapper(
            wrapper_account_url=config.wrapper_account_url,
            language=config.language,
        )
    else:
        cookies_path = prompt_path(config.cookies_path)
        apple_music_api = await AppleMusicApi.create_from_netscape_cookies(
            cookies_path=cookies_path,
            language=config.language,
        )

    itunes_api = ItunesApi(
        apple_music_api.storefront,
        apple_music_api.language,
    )

    if not apple_music_api.active_subscription:
        logger.critical(
            "No active Apple Music subscription found, you won't be able to download"
            " anything"
        )
        return
    if apple_music_api.account_restrictions and config.log_level == "DEBUG":
        logger.debug(
            "Your account has content restrictions enabled, some content may not be"
            " downloadable"
        )

    interface = AppleMusicInterface(
        apple_music_api,
        itunes_api,
    )
    song_interface = AppleMusicSongInterface(interface)
    music_video_interface = AppleMusicMusicVideoInterface(interface)
    uploaded_video_interface = AppleMusicUploadedVideoInterface(interface)

    if config.read_urls_as_txt:
        urls_from_file = []
        for url in config.urls:
            if Path(url).is_file() and Path(url).exists():
                urls_from_file.extend(
                    [
                        line.strip()
                        for line in Path(url).read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                )
        urls = urls_from_file
    else:
        urls = config.urls

    if not urls:
        raw_urls = click.prompt(
            "Paste Apple Music URL(s) (space-separated)",
            default="",
            show_default=False,
        ).strip()
        if raw_urls:
            urls = raw_urls.split()

    download_folder_name = click.prompt(
        "Download folder name (created in current directory)",
        default="Apple Music",
        show_default=True,
    )
    download_path = Path.cwd() / download_folder_name
    download_path.mkdir(parents=True, exist_ok=True)

    use_artist_folders = click.confirm(
        "Group songs into artist folders?",
        default=False,
    )

    flat_folder_template = ""
    flat_file_template = "{artist} - {title}"
    artist_folder_template = "{artist}"
    truncate = config.truncate
    if truncate is None and use_artist_folders:
        truncate = 100
    no_synced_lyrics = False if config.synced_lyrics_only else True

    temp_path = download_path / ".gamdl_temp"
    base_downloader = AppleMusicBaseDownloader(
        output_path=str(download_path),
        temp_path=str(temp_path),
        wvd_path=config.wvd_path,
        overwrite=config.overwrite,
        save_cover=config.save_cover,
        save_playlist=config.save_playlist,
        nm3u8dlre_path=config.nm3u8dlre_path,
        mp4decrypt_path=config.mp4decrypt_path,
        ffmpeg_path=config.ffmpeg_path,
        mp4box_path=config.mp4box_path,
        amdecrypt_path=config.amdecrypt_path,
        use_wrapper=config.use_wrapper,
        wrapper_decrypt_ip=config.wrapper_decrypt_ip,
        download_mode=config.download_mode,
        remux_mode=config.remux_mode,
        cover_format=config.cover_format,
        album_folder_template=artist_folder_template if use_artist_folders else flat_folder_template,
        compilation_folder_template=artist_folder_template if use_artist_folders else flat_folder_template,
        no_album_folder_template=artist_folder_template if use_artist_folders else flat_folder_template,
        single_disc_file_template=flat_file_template,
        multi_disc_file_template=flat_file_template,
        no_album_file_template=flat_file_template,
        playlist_file_template=config.playlist_file_template,
        date_tag_template=config.date_tag_template,
        exclude_tags=config.exclude_tags,
        cover_size=config.cover_size,
        truncate=truncate,
        silent=True,
    )
    song_downloader = AppleMusicSongDownloader(
        base_downloader=base_downloader,
        interface=song_interface,
        codec=config.song_codec,
        synced_lyrics_format=config.synced_lyrics_format,
        no_synced_lyrics=no_synced_lyrics,
        synced_lyrics_only=config.synced_lyrics_only,
        use_album_date=config.use_album_date,
        fetch_extra_tags=config.fetch_extra_tags,
    )
    music_video_downloader = AppleMusicMusicVideoDownloader(
        base_downloader=base_downloader,
        interface=music_video_interface,
        codec_priority=config.music_video_codec_priority,
        remux_format=config.music_video_remux_format,
        resolution=config.music_video_resolution,
    )
    uploaded_video_downloader = AppleMusicUploadedVideoDownloader(
        base_downloader=base_downloader,
        interface=uploaded_video_interface,
        quality=config.uploaded_video_quality,
    )
    downloader = AppleMusicDownloader(
        interface=interface,
        base_downloader=base_downloader,
        song_downloader=song_downloader,
        music_video_downloader=music_video_downloader,
        uploaded_video_downloader=uploaded_video_downloader,
    )

    if not config.synced_lyrics_only:
        if not base_downloader.full_ffmpeg_path and (
            config.remux_mode == RemuxMode.FFMPEG
            or config.download_mode == DownloadMode.NM3U8DLRE
        ):
            logger.critical(X_NOT_IN_PATH.format("ffmpeg", config.ffmpeg_path))
            return

        if (
            not base_downloader.full_mp4box_path
            and config.remux_mode == RemuxMode.MP4BOX
        ):
            logger.critical(X_NOT_IN_PATH.format("MP4Box", config.mp4box_path))
            return

        if not base_downloader.full_mp4decrypt_path and (
            config.song_codec not in (SongCodec.AAC_LEGACY, SongCodec.AAC_HE_LEGACY)
            or config.remux_mode == RemuxMode.MP4BOX
        ):
            logger.critical(X_NOT_IN_PATH.format("mp4decrypt", config.mp4decrypt_path))
            return

        if (
            config.download_mode == DownloadMode.NM3U8DLRE
            and not base_downloader.full_nm3u8dlre_path
        ):
            logger.critical(X_NOT_IN_PATH.format("N_m3u8DL-RE", config.nm3u8dlre_path))
            return

        if config.use_wrapper and not base_downloader.full_amdecrypt_path:
            logger.critical(X_NOT_IN_PATH.format("amdecrypt", config.amdecrypt_path))
            return

        if not config.song_codec.is_legacy() and not config.use_wrapper:
            logger.warning(
                "You have chosen an experimental song codec"
                " without enabling wrapper."
                "They're not guaranteed to work due to API limitations."
            )

    error_count = 0
    for url_index, url in enumerate(urls, 1):
        url_progress = click.style(f"[URL {url_index}/{len(urls)}]", dim=True)
        logger.info(url_progress + f' Processing "{url}"')
        download_queue = None
        phase_accum = [0.0, 0.0, 0.0]
        phase_applied = [0, 0, 0]
        phase_weights = [25, 25, 50]
        total_tracks = 0
        progress_lock = asyncio.Lock()
        progress = ProgressDisplay(total=100, label="Overall", stream=sys.stdout)
        _current_progress = progress
        progress.start()

        def update_phase(progress, phase_index: int, steps_total: int):
            if steps_total <= 0:
                return
            phase_accum[phase_index] += phase_weights[phase_index] / steps_total
            while phase_applied[phase_index] < int(phase_accum[phase_index]):
                progress.update(1)
                phase_applied[phase_index] += 1

        def fill_phase(progress, phase_index: int):
            remaining = phase_weights[phase_index] - phase_applied[phase_index]
            if remaining > 0:
                progress.update(remaining)
                phase_applied[phase_index] += remaining
        try:
            url_info = downloader.get_url_info(url)
            if not url_info:
                logger.warning(
                    url_progress + f' Could not parse "{url}", skipping.',
                )
                progress.finish()
                _current_progress = None
                continue

            def progress_total_cb(count: int):
                nonlocal total_tracks
                if total_tracks <= 0:
                    total_tracks = max(1, count)

            def progress_cb(_: int):
                if total_tracks > 0:
                    update_phase(progress, 0, total_tracks)

            download_queue = await downloader.get_download_queue(
                url_info,
                progress_cb=progress_cb,
                progress_total_cb=progress_total_cb,
            )
            if not download_queue:
                logger.warning(
                    url_progress
                    + f' No downloadable media found for "{url}", skipping.',
                )
                progress.finish()
                _current_progress = None
                continue

            if total_tracks <= 0:
                total_tracks = len(download_queue)
            logger.info(
                click.style(f"[Queue] {total_tracks} track(s) queued", dim=True)
            )
        except KeyboardInterrupt:
            exit(1)
        except Exception:
            error_count += 1
            logger.error(
                url_progress + f' Error processing "{url}"',
                exc_info=not config.no_exceptions,
            )
            progress.finish()
            _current_progress = None
            continue
        echo_stderr("Phase: processing & preparing temp files", progress)
        fill_phase(progress, 0)
        temp_path.mkdir(parents=True, exist_ok=True)
        phase_two_step_delay = 0.0
        for item in download_queue:
            if getattr(item, "random_uuid", None):
                (temp_path / TEMP_PATH_TEMPLATE.format(item.random_uuid)).mkdir(
                    parents=True,
                    exist_ok=True,
                )
            update_phase(progress, 1, total_tracks)
            if phase_two_step_delay:
                await asyncio.sleep(phase_two_step_delay)
        fill_phase(progress, 1)

        echo_stderr("Phase: downloading", progress)
        cpu_limit = max(1, os.cpu_count() or 1)
        semaphore = asyncio.Semaphore(cpu_limit)

        async def download_one(index: int, item: DownloadItem):
            nonlocal error_count
            async with semaphore:
                current_item = item
                for attempt in range(3):
                    try:
                        await downloader.download(current_item)
                        break
                    except (httpx.ReadError, httpcore.ReadError):
                        if attempt >= 2:
                            logger.warning(
                                click.style(f"[Track {index}/{total_tracks}]", dim=True)
                                + " Skipping track (network error)"
                            )
                            break
                        current_item = await downloader.get_single_download_item_no_filter(
                            current_item.media_metadata,
                            current_item.playlist_metadata,
                        )
                    except FileNotFoundError:
                        logger.warning(
                            click.style(f"[Track {index}/{total_tracks}]", dim=True)
                            + " Skipping track (temp file missing)"
                        )
                        break
                    except GamdlError as e:
                        logger.warning(
                            click.style(f"[Track {index}/{total_tracks}]", dim=True)
                            + f" Skipping track: {e}"
                        )
                        break
                    except KeyboardInterrupt:
                        exit(1)
                    except Exception:
                        error_count += 1
                        logger.error(
                            click.style(f"[Track {index}/{total_tracks}]", dim=True)
                            + " Error downloading track",
                            exc_info=not config.no_exceptions,
                        )
                        break
                async with progress_lock:
                    update_phase(progress, 2, total_tracks)

        tasks = [
            asyncio.create_task(download_one(i, item))
            for i, item in enumerate(download_queue, 1)
        ]
        await asyncio.gather(*tasks)
        fill_phase(progress, 2)
        progress.finish()
        _current_progress = None

        try:
            if temp_path.exists():
                for child in temp_path.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                temp_path.rmdir()
        except Exception:
            logger.debug("Temp cleanup failed", exc_info=True)

    logger.info(f"Finished with {error_count} error(s)")
