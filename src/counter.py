import time


class FishCounter:
    def __init__(self, count_line_x, direction='right', min_frames=3):
        self.count_line_x = count_line_x
        self.direction = direction
        self.min_frames = min_frames
        self.counted_ids = set()
        self.total_count = 0
        self.last_count_time = time.time()

    def update(self, tracks):
        """Обновление счетчика"""
        new_count = 0

        for track_id, track in tracks.items():
            if track_id in self.counted_ids:
                continue

            center_x, _ = track['center']

            # Проверяем историю движения
            if len(track['centroids']) >= self.min_frames:
                # Определяем направление движения
                first_x = track['centroids'][0][0]
                last_x = track['centroids'][-1][0]

                if self.direction == 'right':
                    if first_x < self.count_line_x < last_x:
                        self.counted_ids.add(track_id)
                        new_count += 1

        self.total_count += new_count

        return self.total_count