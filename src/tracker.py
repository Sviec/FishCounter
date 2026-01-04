import numpy as np
from collections import deque
import time


class FishTracker:
    def __init__(self,
                 max_disappeared=30,
                 max_distance=100,
                 iou_threshold=0.1,
                 min_hits=10):
        self.next_id = 0
        self.tracks = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.confirmed_tracks = set()

    @staticmethod
    def _calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    @staticmethod
    def _get_center(bbox):
        x, y, w, h = bbox
        return (int(x + w / 2), int(y + h / 2))

    def update(self, detections):
        if len(self.tracks) == 0:
            for bbox in detections:
                self._create_track(bbox)
            return self._get_active_tracks()

        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self._remove_track(track_id)
            return self._get_active_tracks()

        used_detections = set()

        for track_id, track in list(self.tracks.items()):
            if track['disappeared'] > 5:
                continue

            track_center = self._get_center(track['bbox'])
            best_match_idx = -1
            best_distance = float('inf')

            for i, bbox in enumerate(detections):
                if i in used_detections:
                    continue

                det_center = self._get_center(bbox)
                distance = np.sqrt((det_center[0] - track_center[0]) ** 2 +
                                   (det_center[1] - track_center[1]) ** 2)

                if distance < best_distance and distance < self.max_distance:
                    if det_center[0] > track_center[0] - 30:
                        best_distance = distance
                        best_match_idx = i

            if best_match_idx != -1:
                self.tracks[track_id]['bbox'] = detections[best_match_idx]
                self.tracks[track_id]['disappeared'] = 0
                self.tracks[track_id]['centroids'].append(self._get_center(detections[best_match_idx]))
                self.tracks[track_id]['hits'] += 1
                self.tracks[track_id]['age'] += 1

                if not self.tracks[track_id]['confirmed'] and self.tracks[track_id]['hits'] >= self.min_hits:
                    self.tracks[track_id]['confirmed'] = True
                    self.confirmed_tracks.add(track_id)

                used_detections.add(best_match_idx)
            else:
                self.tracks[track_id]['disappeared'] += 1
                self.tracks[track_id]['age'] += 1

        for i, bbox in enumerate(detections):
            if i not in used_detections:
                too_close = False
                det_center = self._get_center(bbox)

                for track_id, track in self.tracks.items():
                    if track['disappeared'] > 5:
                        continue

                    track_center = self._get_center(track['bbox'])
                    distance = np.sqrt((det_center[0] - track_center[0]) ** 2 +
                                       (det_center[1] - track_center[1]) ** 2)

                    if distance < 60:
                        too_close = True
                        break

                if not too_close:
                    self._create_track(bbox)

        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                self._remove_track(track_id)

        return self._get_active_tracks()

    def _create_track(self, bbox):
        self.tracks[self.next_id] = {
            'bbox': bbox,
            'centroids': deque([self._get_center(bbox)], maxlen=15),
            'age': 0,
            'hits': 1,
            'disappeared': 0,
            'confirmed': False,
            'created_at': time.time()
        }
        self.next_id += 1

    def _remove_track(self, track_id):
        if track_id in self.tracks:
            if track_id in self.confirmed_tracks:
                self.confirmed_tracks.remove(track_id)
            del self.tracks[track_id]

    def _get_active_tracks(self):
        active_tracks = {}

        for track_id, track in self.tracks.items():
            if track['disappeared'] <= self.max_disappeared and track['confirmed']:
                active_tracks[track_id] = {
                    'bbox': track['bbox'],
                    'center': track['centroids'][-1],
                    'centroids': list(track['centroids']),
                    'age': track['age'],
                    'hits': track['hits'],
                    'confirmed': track['confirmed']
                }

        return active_tracks