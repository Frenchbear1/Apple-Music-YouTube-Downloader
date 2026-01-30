# youtube_to_mp3.py
# Downloads YouTube videos / playlists → M4A (no conversion, fastest)
# Saves them into a user-named folder next to this script

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yt_dlp

def download_to_mp3(urls, output_folder):
    # Make sure the folder exists
    os.makedirs(output_folder, exist_ok=True)

    if not urls:
        print("No valid URLs provided.")
        return

    print(f"\nSaving {len(urls)} item(s) to folder: {output_folder}\n")

    def _fmt_bytes(num):
        if num is None:
            return "0B"
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if num < 1024 or unit == "TiB":
                return f"{num:.2f}{unit}" if unit != "B" else f"{int(num)}B"
            num /= 1024

    def _fmt_time(seconds):
        if seconds is None:
            return "00:00:00"
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    _progress_state = {
        "current_id": None,
        "printed_title": False,
        "last_print_ts": 0.0,
        "last_percent": None,
    }

    def _progress_hook(d):
        info = d.get("info_dict") or {}
        title = info.get("title") or "Unknown title"
        video_id = info.get("id")

        if video_id != _progress_state["current_id"]:
            _progress_state["current_id"] = video_id
            _progress_state["printed_title"] = False

        playlist_index = info.get("playlist_index")
        playlist_total = info.get("n_entries") or info.get("playlist_count") or info.get("playlist_size")
        prefix = f"({playlist_index}/{playlist_total}) " if playlist_index and playlist_total else ""

        if d.get("status") == "downloading":
            # Quiet mode: no per-chunk progress spam
            return

        if d.get("status") == "finished":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            elapsed = d.get("elapsed")
            speed = d.get("speed")
            print(" " * 140, end="\r", flush=True)
            print(f"{prefix}{title} — 100% of {_fmt_bytes(total)} in {_fmt_time(elapsed)} at {_fmt_bytes(speed)}/s")

    class _QuietLogger:
        def debug(self, msg):
            pass
        def info(self, msg):
            pass
        def warning(self, msg):
            pass
        def error(self, msg):
            pass

    ydl_opts = {
        # Prefer native M4A when available to avoid any conversion
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'logger': _QuietLogger(),
        'progress_hooks': [_progress_hook],
        'continuedl': True,
        'ignoreerrors': True,  # keep going when a playlist item is unavailable
        'retries': 10,
        'fragment_retries': 10,
        # Speed up HLS/DASH downloads by fetching multiple fragments in parallel
        'concurrent_fragment_downloads': 4,
        'windowsfilenames': True,
        'noplaylist': False,

        # Current 2026 YouTube workarounds
        'extractor_args': {
            'youtube': {
                'player_client': ['default', 'web'],
                'skip': ['android_sdkless', 'web_safari'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        },
    }

    def _download_one(url):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\n→ Processing: {url}")
            try:
                info = ydl.extract_info(url, download=True)
                if 'entries' in info:
                    print(f"   Playlist: {info.get('title', 'Untitled')} ({len(info['entries'])} items)")
                else:
                    print(f"   Saved: {info.get('title', 'Unknown title')}")
            except Exception as e:
                print(f"   Error: {str(e)}")
            print("-" * 70)

    max_workers = min(4, os.cpu_count() or 1, len(urls))
    if max_workers <= 1:
        for url in urls:
            _download_one(url)
    else:
        print(f"Using up to {max_workers} parallel downloads...\n")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download_one, url) for url in urls]
            for _ in as_completed(futures):
                pass

    print("\n" + "═" * 70)
    print(f"Finished! All files saved in: ./{output_folder}/")
    print("═" * 70 + "\n")


def run_cli():
    print("═" * 22 + " YouTube → M4A Downloader " + "═" * 22)
    print("All downloads will go into a folder in the current directory\n")

    urls_input = input("Paste YouTube URL(s) or playlist link(s)\n(comma or newline separated):\n> ").strip()
    urls = [u.strip() for u in urls_input.replace("\n", ",").split(",") if u.strip()]

    if not urls:
        print("No URLs entered. Exiting.")
        input("\nPress Enter to close the window...")
        return

    print("\nFolder options:")
    print("[1] Make a new folder in the current directory")
    print("[2] Add to an existing folder by path")
    while True:
        folder_choice = input("Choose [1] or [2]: ").strip()
        if folder_choice in {"1", "2"}:
            break
        print("Please enter 1 or 2.")

    if folder_choice == "1":
        folder_name = input("Download folder name (created in current directory) [YouTube]: ").strip()
        if not folder_name:
            folder_name = "YouTube"
        folder_name = "".join(c for c in folder_name if c.isalnum() or c in " -_()")
        if not folder_name:
            folder_name = "YouTube"
        output_folder = str(Path.cwd() / folder_name)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    else:
        raw_folder_path = input("Folder path: ").strip()
        if raw_folder_path.startswith('"') and raw_folder_path.endswith('"'):
            raw_folder_path = raw_folder_path[1:-1].strip()
        output_folder = str(Path(raw_folder_path))
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print(f"Using folder: {output_folder}")

    download_to_mp3(urls, output_folder)

    input("\nPress Enter to close the window...")


if __name__ == "__main__":
    run_cli()
