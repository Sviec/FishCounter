import cv2


class FishDetector:
    def __init__(self,
                 history: int = 100,
                 varThreshold: int = 50,
                 min_area: int = 800,
                 max_area: int = 5000,
                 clipLimit: float = 2.0,
                 tileGridkernel: int = 4,
                 morph_kernel: int = 7,
                 g_blur: int = 7,
                 learning_rate: float = 0.01,
                 merge_threshold: int = 50):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=varThreshold,
            detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self.clipLimit = clipLimit
        self.tileGridkernel = tileGridkernel
        self.g_blur = g_blur
        self.learning_rate = learning_rate
        self.merge_threshold = merge_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))

    def preprocess(self, frame):
        """Подготовка кадра - обрезка боков и улучшение контраста"""
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Улучшение контраста для темных рыб
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileGridkernel, self.tileGridkernel))
        enhanced = clahe.apply(gray)

        # Легкое размытие для подавления шума
        blurred = cv2.GaussianBlur(enhanced, (self.g_blur, self.g_blur), 0)

        return blurred, frame

    def detect(self, frame):
        """Обнаружение рыб с улучшенной морфологией"""
        processed, original = self.preprocess(frame)

        # Фоновое вычитание
        fg_mask = self.bg_subtractor.apply(processed, learningRate=self.learning_rate)

        # Улучшенная морфология
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.dilate(fg_mask, self.kernel, iterations=1)

        # Находим контуры
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bboxes = []
        valid_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Фильтрация по размеру и форме
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)

                # Фильтр по соотношению сторон
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    bboxes.append((x, y, w, h))
                    valid_contours.append(contour)
        if len(bboxes) > 1:
            bboxes = self._merge_close_boxes(bboxes)

        return bboxes, fg_mask, valid_contours, original

    def _merge_close_boxes(self, boxes):
        """Объединяет близко расположенные bounding boxes"""
        if not boxes:
            return boxes

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []
        current = list(boxes[0])

        for box in boxes[1:]:
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = box

            # Проверяем расстояние между боксами
            horizontal_gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            vertical_gap = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            min_distance = max(horizontal_gap, vertical_gap)

            # Проверяем перекрытие
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            # Объединяем 
            if inter_area > 0 or min_distance < self.merge_threshold:
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                current = [new_x, new_y, new_w, new_h]
            else:
                merged.append(tuple(current))
                current = list(box)

        merged.append(tuple(current))
        return merged