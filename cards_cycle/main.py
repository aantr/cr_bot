# ml setup
# avd setup


from collections import deque
import multiprocessing
from os import path
import time

from ultralytics import YOLO

from adb_touch import ADBTouchController
from cv import find_and_draw_pattern
import cv2

import subprocess
import socket
import struct
from image2cards import get_card_by_index, get_image_cards_format
from ml.efficient_net_predict import load_trained_model, predict_single_image
import numpy as np
import torch


def init_device():
    # Get device ABI and SDK version
    abi = subprocess.run(
        ["adb", "shell", "getprop", "ro.product.cpu.abi"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    sdk = subprocess.run(
        ["adb", "shell", "getprop", "ro.build.version.sdk"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Push minicap binary and library
    subprocess.run(
        ["adb", "push", f"minicap/prebuilt/{abi}/bin/minicap", "/data/local/tmp/"],
        check=True,
    )
    subprocess.run(
        [
            "adb",
            "push",
            f"minicap/prebuilt/{abi}/lib/android-{sdk}/minicap.so",
            "/data/local/tmp/",
        ],
        check=True,
    )

    # Set permissions
    subprocess.run(["adb", "shell", "chmod", "777", "/data/local/tmp/minicap"])


def start_minicap_process(port=1313, device_serial=None):
    """Run minicap in a separate process"""
    # Set up ADB with specific device if provided
    adb_prefix = ["adb"]
    if device_serial:
        adb_prefix = ["adb", "-s", device_serial]

    # Get screen size
    screen_size = subprocess.run(
        adb_prefix + ["shell", "wm", "size"], capture_output=True, text=True
    ).stdout
    size = screen_size.split(":")[1].strip()

    # Start minicap service on device
    subprocess.Popen(
        adb_prefix
        + [
            "shell",
            f"LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/minicap "
            f"-P {size}@{size}/0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait a moment for minicap to initialize
    time.sleep(1)

    # Setup port forwarding
    subprocess.run(
        adb_prefix + ["forward", f"tcp:{port}", "localabstract:minicap"], check=True
    )

    print(f"Minicap started on port {port}")

    # Keep the process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping minicap...")
        # Clean up port forwarding
        subprocess.run(adb_prefix + ["forward", "--remove", f"tcp:{port}"])


class MinicapClient:
    def __init__(self, host="127.0.0.1", port=1313):
        self.connection = socket.create_connection((host, port))
        self.read_banner()

    def read_bytes(self, length):
        data = bytearray()
        while length > 0:
            tmp = self.connection.recv(length)
            if not tmp:
                break
            length -= len(tmp)
            data.extend(tmp)
        return bytes(data)

    def read_banner(self):
        # Read banner header (24 bytes)
        banner_data = self.read_bytes(24)

        # Parse banner structure
        version = banner_data[0]
        banner_size = banner_data[1]
        pid = struct.unpack("<I", banner_data[2:6])[0]
        real_width = struct.unpack("<I", banner_data[6:10])[0]
        real_height = struct.unpack("<I", banner_data[10:14])[0]
        virtual_width = struct.unpack("<I", banner_data[14:18])[0]
        virtual_height = struct.unpack("<I", banner_data[18:22])[0]
        orientation = banner_data[22]
        quirks = banner_data[23]

        print(
            f"Banner received - Version: {version}, Real size: {real_width}x{real_height}"
        )
        return {
            "version": version,
            "real_width": real_width,
            "real_height": real_height,
            "virtual_width": virtual_width,
            "virtual_height": virtual_height,
        }

    def read_frames(self):
        while True:
            # Read frame size (4 bytes, little-endian)
            frame_bytes = self.read_bytes(4)
            if len(frame_bytes) < 4:
                break
            total = struct.unpack("<I", frame_bytes)[0]

            # Read JPEG data
            jpeg_data = self.read_bytes(total)
            if jpeg_data:
                yield jpeg_data


def display_frames_process(port=1313, target_fps=30, callback=None):
    """Run display in main process"""
    client = MinicapClient(port=port)

    frame_interval = 1.0 / target_fps
    last_frame_time = time.time()

    print(f"Display started - Target FPS: {target_fps}")
    print("Press 'q' to quit")

    for jpeg_data in client.read_frames():
        img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is not None:
            current_time = time.time()
            elapsed = current_time - last_frame_time

            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            last_frame_time = time.time()
            if callback:
                result = callback(img)
                if result:
                    frame_interval = 1.0 / result

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def advanced_average_difference(video_path, buffer_size=5, threshold=30):
    """
    Расширенная версия с дополнительными возможностями.

    Args:
        adaptive_threshold: использовать адаптивный порог вместо фиксированного
        save_masks: сохранять маски в файлы
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        return

    frame_buffer = deque(maxlen=buffer_size)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = get_image_cards_format(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        frame_buffer.append(gray)

        if len(frame_buffer) == buffer_size:
            # Вычисляем средний кадр
            average_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)

            # Вычисляем разность
            diff = cv2.absdiff(gray, average_frame)

            # Создаем маску
            if adaptive_threshold:
                # Адаптивный порог для лучшего выделения в разных условиях освещения
                mask = cv2.adaptiveThreshold(
                    diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            else:
                _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

            # Улучшаем маску
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Вычисляем статистику разности
            diff_mean = np.mean(diff)
            diff_std = np.std(diff)
            motion_percentage = (np.sum(mask > 0) / mask.size) * 100

            # Отображаем информацию
            info_frame = frame.copy()
            cv2.putText(
                info_frame,
                f"Motion: {motion_percentage:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                info_frame,
                f"Diff mean: {diff_mean:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Отображаем
            cv2.imshow("Current Frame", info_frame)
            cv2.imshow("Average Frame", average_frame)
            cv2.imshow("Difference (Gray)", diff)
            cv2.imshow("Mask", mask)

            # Цветовая маска для наглядности
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
            cv2.imshow("Overlay", overlay)

        frame_count += 1

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):  # Сохранить текущий кадр
            cv2.imwrite(f"saved_frame_{frame_count}.png", frame)
            cv2.imwrite(f"saved_mask_{frame_count}.png", mask)
            print(f"Сохранен кадр {frame_count}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Обработано кадров: {frame_count}")


class CardsCycle:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "ml/best_model_cards.pth"  # путь к вашей модели
        self.model, self.classes = load_trained_model(
            self.model_path, self.device, verbose=False
        )

        self.model_yolo = YOLO("ml/best.pt")

        self.nickname_im = cv2.imread("res/nickname.png")
        self.spectate_im = cv2.imread("res/spectate.png")

        self.touch = ADBTouchController()
        self.in_battle = True

        self.buffer_size = 5
        self.frame_buffer = deque(maxlen=self.buffer_size)

        self.last_time_played = [time.time() for _ in range(8)]
        self.deck = [i for i in range(8)]

        self.dataset_path = "ml/cards_dataset/dataset"
        self.card_ims = {}

    def predict_image(self, image):

        results = self.model_yolo.predict(image, imgsz=224, verbose=False)

        # Get prediction details
        result = results[0]
        top_class = result.probs.top1
        top_confidence = result.probs.top1conf
        class_name = result.names[top_class]
        return class_name

    def get_deck_im(self, img_cards):
        height, width = img_cards.shape[:2]
        result = img_cards.copy()
        for i in range(8):
            card_im = get_card_by_index(img_cards, self.deck[i])
            card_width = width // 8
            for x in range(card_width):
                for y in range(height):
                    result[y, card_width * i + x] = card_im[y, x]
        return result[:, : width // 2]

    def get_deck_im_predicted(self, predicted):
        ims = []
        for idx in range(4):
            i = self.deck[idx]
            if predicted[i] not in self.card_ims:
                self.card_ims[predicted[i]] = cv2.imread(
                    path.join(
                        self.dataset_path, predicted[i], f"{predicted[i]}_001.png"
                    )
                )
            im = self.card_ims[predicted[i]]

            ims.append(im)

        # Приводим все изображения к одинаковой высоте
        heights = [img.shape[0] for img in ims]
        max_height = max(heights)

        resized_images = []
        for img in ims:
            # Сохраняем пропорции при изменении размера
            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = int(max_height * aspect_ratio)
            resized = cv2.resize(img, (new_width, max_height))
            resized_images.append(resized)

            # Горизонтальная склейка
        horizontal_img = np.hstack(resized_images)
        return horizontal_img

    def on_card_played(self, idx):
        if time.time() - self.last_time_played[idx] >= 3:
            self.last_time_played[idx] = time.time()
            self.deck.remove(idx)
            self.deck.append(idx)

    def tap(self, x, y):
        self.touch.tap(x, y)

    def process_frame(self, img):
        result_fps = None

        # img = cv2.imread('test/test3.png')
        # cv2.imwrite(r'test\test4.png', img)
        if not self.in_battle:
            result_fps = 24
            img, nickname_matches = find_and_draw_pattern(
                img, self.nickname_im, threshold=0.8
            )
            img, specate_matches = find_and_draw_pattern(
                img, self.spectate_im, threshold=0.5
            )

            def is_intersect(a, b, c, d):
                return max(a, c) - min(b, d) > 0

            for nick in nickname_matches:
                for spectate in specate_matches:
                    if is_intersect(
                        nick[0],
                        nick[0] + nick[2],
                        spectate[0],
                        spectate[0] + spectate[2],
                    ):
                        x, y = (
                            spectate[0] + spectate[2] // 2,
                            spectate[1] + spectate[3] // 2,
                        )
                        print("found specate", (x, y))
                        self.tap(x, y)
                        self.in_battle = True
            cv2.imshow("Android Screen Stream", img)

        else:
            result_fps = 40
            img_cards = get_image_cards_format(img)
            # cv2.imwrite(r'test\test_cards.png', img_cards)
            # cv2.imshow('cards', img_cards)
            cv2.imshow("deck", self.get_deck_im(img_cards))

            # predicted = [predict_single_image(self.model, get_card_by_index(img_cards, i),
            #                                   self.classes, self.device, verbose=False) for i in range(8)]  # for efficient_net prediction

            predicted = [
                self.predict_image(get_card_by_index(img_cards, i)) for i in range(8)
            ]  # for yolo prediction
            cv2.imshow("deck predicted", self.get_deck_im_predicted(predicted))

            # for i in range(8):
            #     cv2.imwrite(f'test/idx_{i}.png', get_card_by_index(img_cards, i))

            gray = cv2.cvtColor(img_cards, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            self.frame_buffer.append(gray)

            if len(self.frame_buffer) == self.buffer_size:
                # Вычисляем средний кадр
                average_frame = np.mean(self.frame_buffer, axis=0).astype(np.uint8)

                # Вычисляем разность
                diff = cv2.absdiff(gray, average_frame)

                # Создаем маску
                _, mask = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)

                # Улучшаем маску
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # cv2.imshow('Mask', mask)

                height, width = mask.shape[:2]

                count = [0 for _ in range(8)]
                for y in range(height):
                    for x in range(width):
                        pixel = mask[y, x]
                        if pixel == 255:
                            count[x * 8 // width] += 1
                        pass
                for i in range(8):
                    if count[i] >= 2500:
                        print(predicted)
                        print("deck:", self.deck)
                        self.on_card_played(i)
        return result_fps


if __name__ == "__main__":
    # Create and start minicap process
    minicap_process = multiprocessing.Process(
        target=start_minicap_process,
        args=(1313,),  # port
        # kwargs={'device_serial': 'your_device_serial'}  # Optional: specify device
    )
    minicap_process.daemon = True  # Will exit when main process exits
    minicap_process.start()

    # Give minicap time to initialize
    time.sleep(3)

    # Run display in main process
    try:
        cards_cycle = CardsCycle()
        display_frames_process(
            port=1313, target_fps=24, callback=cards_cycle.process_frame
        )

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Clean up
        minicap_process.terminate()
        minicap_process.join()
