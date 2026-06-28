import time
import config
from utils import point_in_rect, get_frame_center


class TrackData:
    __slots__ = (
        "track_id", "first_seen_frame", "zone_dwell", "total_dwell_frames",
        "last_seen_frame", "last_zone", "center_history",
    )

    def __init__(self, track_id, first_frame):
        self.track_id = track_id
        self.first_seen_frame = first_frame
        self.zone_dwell = {}          # zone_label -> accumulated frames
        self.total_dwell_frames = 0
        self.last_seen_frame = first_frame
        self.last_zone = None
        self.center_history = []      # list of (x, y)


class Analytics:
    def __init__(self, fps):
        self.fps = fps
        self.tracks = {}              # track_id -> TrackData
        self.peak_occupancy = 0
        self.zone_counts = {z[4]: 0 for z in config.ZONES}
        self.entries = 0
        self.exits = 0
        self._seen_ids = set()
        self._gone_frames = {}        # track_id -> frames since last seen

    def update(self, results, current_frame):
        active_ids = set()
        if results is None or results.boxes is None:
            self._mark_missing(active_ids, current_frame)
            return

        boxes = results.boxes.xywh.cpu()
        ids = results.boxes.id

        for i in range(len(boxes)):
            if ids is None or ids[i] is None:
                continue
            tid = int(ids[i])
            cx, cy = get_frame_center(boxes[i].numpy())

            active_ids.add(tid)
            self._gone_frames.pop(tid, None)

            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                self.entries += 1
                self.tracks[tid] = TrackData(tid, current_frame)

            td = self.tracks[tid]
            td.last_seen_frame = current_frame
            td.center_history.append((cx, cy))
            if len(td.center_history) > 120:
                td.center_history = td.center_history[-120:]

            # zone assignment
            current_zone = self._point_to_zone(cx, cy)
            if current_zone and current_zone != td.last_zone:
                if td.last_zone:
                    td.zone_dwell[td.last_zone] = td.zone_dwell.get(td.last_zone, 0) + 1
                td.last_zone = current_zone

            if current_zone:
                td.zone_dwell[current_zone] = td.zone_dwell.get(current_zone, 0) + 1

            td.total_dwell_frames += 1

        self._mark_missing(active_ids, current_frame)

        # occupancy
        occupancy = len(active_ids)
        self.peak_occupancy = max(self.peak_occupancy, occupancy)

        # per-zone counts
        self.zone_counts = {z[4]: 0 for z in config.ZONES}
        for tid in active_ids:
            td = self.tracks.get(tid)
            if td and td.last_zone:
                self.zone_counts[td.last_zone] = self.zone_counts.get(td.last_zone, 0) + 1

    def _mark_missing(self, active_ids, current_frame):
        for tid in list(self._gone_frames.keys()):
            if tid not in active_ids:
                self._gone_frames[tid] += 1
                if self._gone_frames[tid] > config.REAPPEAR_FRAMES:
                    self._gone_frames.pop(tid)
                    self._seen_ids.discard(tid)
                    self.exits += 1
        for tid in set(self._seen_ids) - active_ids:
            self._gone_frames[tid] = self._gone_frames.get(tid, 0) + 1

    def _point_to_zone(self, px, py):
        for zone in config.ZONES:
            if point_in_rect(px, py, zone):
                return zone[4]
        return None

    def get_dwell_sec(self, track_id):
        td = self.tracks.get(track_id)
        if td is None:
            return 0
        return td.total_dwell_frames / self.fps

    def get_zone_dwell_sec(self, track_id, zone_label):
        td = self.tracks.get(track_id)
        if td is None:
            return 0
        frames = td.zone_dwell.get(zone_label, 0)
        return frames / self.fps

    def get_avg_store_dwell(self):
        dwells = [self.get_dwell_sec(tid) for tid in self.tracks]
        return sum(dwells) / len(dwells) if dwells else 0

    def get_zone_popularity(self):
        """Return sorted list of (zone_label, total_dwell_seconds_across_all_tracks)."""
        totals = {}
        for td in self.tracks.values():
            for z_label, frames in td.zone_dwell.items():
                totals[z_label] = totals.get(z_label, 0) + frames
        return sorted(totals.items(), key=lambda x: -x[1])

    def get_zone_coverage_ratio(self):
        """Fraction of zones that have at least some dwell."""
        active_zones = sum(1 for td in self.tracks.values() if td.zone_dwell)
        total_zones = len(config.ZONES)
        return active_zones / total_zones if total_zones else 0

    def business_health_score(self):
        if not self.tracks:
            return 0

        occupancy_score = min(100, (self.peak_occupancy / max(len(self.tracks), 1)) * 100)

        avg_dwell = self.get_avg_store_dwell()
        dwell_score = min(100, (avg_dwell / config.LONG_DWELL_SEC) * 100) if config.LONG_DWELL_SEC else 0

        coverage_score = self.get_zone_coverage_ratio() * 100

        score = (
            config.BHS_OCCUPANCY_W * occupancy_score
            + config.BHS_DWELL_W * dwell_score
            + config.BHS_ZONE_COVERAGE_W * coverage_score
        )
        return min(100, max(0, score))
