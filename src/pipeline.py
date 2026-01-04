import cv2

from src.counter import FishCounter
from src.detector import FishDetector
from src.tracker import FishTracker


class FishDetectionPipeline:
    def __init__(self,
                 frame_width,
                 count_line_ratio: float = 0.85,
                 direction: str = 'right',
                 history: int = 100,
                 varThreshold: int = 50,
                 min_area: int = 700,
                 max_area: int = 10000,
                 clipLimit: float = 2.5,
                 tileGridkernel: int = 4,
                 morph_kernel: int = 7,
                 g_blur: int = 7,
                 learning_rate: float = 0.001,
                 merge_threshold: int = 50,
                 max_disappeared=15,
                 max_distance=300,
                 iou_threshold=0.3,
                 min_hits=10,
                 visualize: bool = True):

        self.detector = FishDetector(
            history=history,
            varThreshold=varThreshold,
            min_area=min_area,
            max_area=max_area,
            clipLimit=clipLimit,
            tileGridkernel=tileGridkernel,
            morph_kernel=morph_kernel,
            g_blur=g_blur,
            learning_rate=learning_rate,
            merge_threshold=merge_threshold
        )
        self.tracker = FishTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance,
            iou_threshold=iou_threshold,
            min_hits=min_hits
        )
        self.counter = FishCounter(
            count_line_x=int(frame_width * count_line_ratio),
            direction=direction,
            min_frames=3
        )
        self.frame_count = 0
        self.visualize = visualize

    def process_frame(self, frame):
        """Обработка одного кадра"""
        # 1. Детекция
        bboxes, mask, contours, processed_frame = self.detector.detect(frame)

        # 2. Трекинг
        tracks = self.tracker.update(bboxes)

        # 3. Подсчет
        total_fish = self.counter.update(tracks)

        # 4. Визуализация
        if self.visualize:
            result_frame = self._visualize(
                processed_frame, tracks, contours,
                mask, total_fish
            )
        else:
            result_frame = None

        self.frame_count += 1

        # Возвращаем разные значения в зависимости от визуализации
        if self.visualize:
            return result_frame, mask, total_fish, tracks
        else:
            return mask, total_fish, tracks, bboxes

    def _visualize(self, frame, tracks, contours, mask, total_fish):
        """Визуализация результатов (только если visualize=True)"""
        display = frame.copy()

        # Линия подсчета
        cv2.line(display,
                 (self.counter.count_line_x, 0),
                 (self.counter.count_line_x, frame.shape[0]),
                 (0, 255, 255), 3)

        # Контуры
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 1)

        # Треки
        for track_id, track in tracks.items():
            # Bbox
            x, y, w, h = track['bbox']
            color = (0, 255, 0) if track_id not in self.counter.counted_ids else (0, 0, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # ID и возраст
            cv2.putText(display, f"ID:{track_id}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Линия пути
            centroids = track.get('centroids', [])
            if len(centroids) > 1:
                for i in range(1, len(centroids)):
                    cv2.line(display, centroids[i - 1], centroids[i], (255, 255, 0), 1)

            # Центр
            center = track['center']
            cv2.circle(display, center, 4, color, -1)

            # Статус подсчета
            if track_id in self.counter.counted_ids:
                cv2.putText(display, "COUNTED",
                            (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Информационная панель
        info_y = 30
        cv2.putText(display, f"Total: {total_fish}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"Active tracks: {len(tracks)}",
                    (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {self.frame_count}",
                    (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Координата линии
        cv2.putText(display, f"Line X: {self.counter.count_line_x}",
                    (self.counter.count_line_x - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return display