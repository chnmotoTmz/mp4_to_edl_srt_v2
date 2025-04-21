import re

class TimecodeConverter:
    """
    Handles timecode conversions for a specific frame rate.
    Assumes non-drop frame.
    """
    def __init__(self, frame_rate: int = 60):
        self.frame_rate = frame_rate

    def ms_to_hhmmssff(self, ms: int) -> str:
        """Converts milliseconds to HH:MM:SS:FF format."""
        total_seconds = ms / 1000.0
        total_frames = int(total_seconds * self.frame_rate)
        hours = total_frames // (3600 * self.frame_rate)
        minutes = (total_frames % (3600 * self.frame_rate)) // (60 * self.frame_rate)
        secs = (total_frames % (60 * self.frame_rate)) // self.frame_rate
        frames = total_frames % self.frame_rate
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}:{int(frames):02d}"

    def ms_to_hhmmssmmm(self, ms: int) -> str:
        """Converts milliseconds to HH:MM:SS,MMM format."""
        hours = ms // (1000 * 3600)
        ms %= (1000 * 3600)
        minutes = ms // (1000 * 60)
        ms %= (1000 * 60)
        secs = ms // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def hhmmssff_to_ms(self, tc: str) -> int:
        """Converts HH:MM:SS:FF to milliseconds."""
        h, m, s, f = map(int, tc.split(":"))
        total_frames = h * 3600 * self.frame_rate + m * 60 * self.frame_rate + s * self.frame_rate + f
        return int(total_frames * 1000 / self.frame_rate)

    def hhmmssmmm_to_ms(self, tc: str) -> int:
        """Converts HH:MM:SS,MMM to milliseconds."""
        parts = re.split(r'[:,]', tc)
        h, m, s, ms = map(int, parts)
        return h * 3600 * 1000 + m * 60 * 1000 + s * 1000 + ms

    def ms_to_srt_time(self, ms: int) -> str:
        """Converts milliseconds to SRT time format HH:MM:SS,mmm."""
        if ms < 0:
            ms = 0 # Ensure non-negative time
        # Use integer division and modulo for clarity and correctness
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        milliseconds = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}" 