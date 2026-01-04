import cv2
import numpy as np
import time

from src.pipeline import FishDetectionPipeline
from utils.metrics import mse, mae


def manual_test():
    results = []
    videos = [i for i in range(140, 150)]
    counts = [55, 21, 21, 16, 45, 53, 0, 21, 19, 7]
    visualize = False

    local_result = []
    fps_results = []
    processing_times = []

    for i in range(len(videos)):
        print(f'Видео: {videos[i]}')

        pipeline = FishDetectionPipeline(
            frame_width=640,
            count_line_ratio=0.85,
            direction='right',
            history=55,
            varThreshold=35,
            min_area=700,
            max_area=10000,
            clipLimit=2.5,
            tileGridkernel=7,
            morph_kernel=3,
            g_blur=7,
            learning_rate=0.072,
            merge_threshold=110,
            max_disappeared=30,
            max_distance=280,
            iou_threshold=0.6,
            min_hits=7,
            visualize=visualize
        )

        cap = cv2.VideoCapture(f'data/videos/my_video-{videos[i]}.mkv')

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pipeline.counter.count_line_x = int(frame_width * 0.85)

        frame_count = 0
        start_time = time.time()
        frame_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            if visualize:
                result, mask, total_fish, tracks = pipeline.process_frame(frame)

                frame_end = time.time()
                frame_time = frame_end - frame_start
                frame_times.append(frame_time)

                current_fps = 1.0 / frame_time if frame_time > 0 else 0

                cv2.putText(result, f'FPS: {current_fps:.1f}',
                            (frame_width - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Detection', result)
                cv2.imshow('Mask', mask)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                mask, total_fish, tracks, bbox = pipeline.process_frame(frame)
                frame_end = time.time()
                frame_times.append(frame_end - frame_start)

            frame_count += 1

            # # Выводим прогресс каждые 100 кадров
            # if frame_count % 100 == 0:
            #     elapsed = time.time() - start_time
            #     avg_fps = frame_count / elapsed if elapsed > 0 else 0
            #     print(f'  Обработано кадров: {frame_count}/{total_frames} '
            #           f'({frame_count/total_frames*100:.1f}%), '
            #           f'FPS: {avg_fps:.1f}')

        end_time = time.time()
        total_processing_time = end_time - start_time

        avg_frame_time = np.mean(frame_times) if frame_times else 0
        min_frame_time = np.min(frame_times) if frame_times else 0
        max_frame_time = np.max(frame_times) if frame_times else 0
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        min_fps = 1.0 / max_frame_time if max_frame_time > 0 else 0
        max_fps = 1.0 / min_frame_time if min_frame_time > 0 else 0

        print(f'  Время обработки: {total_processing_time:.2f} сек')
        print(f'  Обработано кадров: {frame_count}')
        print(f'  Среднее время на кадр: {avg_frame_time * 1000:.1f} мс')
        print(f'  Средний FPS: {avg_fps:.1f}')
        print(f'  Минимальный FPS: {min_fps:.1f}')
        print(f'  Максимальный FPS: {max_fps:.1f}')
        print(f'  Получено рыб: {total_fish}')
        print(f'  Ошибка: {abs(total_fish - counts[i])}')

        cap.release()
        local_result.append(total_fish)
        fps_results.append(avg_fps)
        processing_times.append(total_processing_time)

    cv2.destroyAllWindows()

    print("Общая статистика:")

    for i in range(len(videos)):
        print(f'Видео {videos[i]}: {local_result[i]}/{counts[i]} рыб, '
              f'FPS: {fps_results[i]:.1f}, '
              f'Время: {processing_times[i]:.2f} сек')

    print(f'\nСредние показатели:')
    print(f'  Средний FPS: {np.mean(fps_results):.1f}')
    print(f'  Минимальный FPS: {np.min(fps_results):.1f}')
    print(f'  Максимальный FPS: {np.max(fps_results):.1f}')
    print(f'  Общее время обработки: {np.sum(processing_times):.2f} сек')
    print(f'  Среднее время на видео: {np.mean(processing_times):.2f} сек')

    results.append({
        'mse': mse(np.array(counts), np.array(local_result)),
        'mae': mae(np.array(counts), np.array(local_result)),
        'avg_fps': np.mean(fps_results),
        'min_fps': np.min(fps_results),
        'max_fps': np.max(fps_results),
        'total_time': np.sum(processing_times),
        'predictions': local_result.copy()
    })

    results_sorted = sorted(results, key=lambda x: x['mse'])
    print("РЕЗУЛЬТАТЫ ПО МЕТРИКАМ:")

    for i, r in enumerate(results_sorted[:5]):
        print(f"{i + 1}. MSE: {r['mse']:.2f}, "
              f"MAE: {r['mae']:.2f}, "
              f"Средний FPS: {r['avg_fps']:.1f}")
