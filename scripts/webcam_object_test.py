from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None


def parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teste simples de deteccao YOLO pela webcam."
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help=(
            "Modelo YOLO usado no teste. Use yolov8n.pt para COCO ou o best.pt "
            "treinado para impressao 3D."
        ),
    )
    parser.add_argument("--source", default="0", help="Indice da webcam. Ex.: 0 ou 1.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Ex.: cpu, 0, cuda:0.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=None,
        help=(
            "Opcional: mostra apenas classes especificas. Ex.: printed_part "
            "spaghetti. Sem esse parametro, mostra todas."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Opcional: pasta para salvar capturas quando pressionar 's'.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if YOLO is None:
        missing.append("ultralytics")

    if missing:
        raise SystemExit(
            "Dependencias ausentes: "
            + ", ".join(missing)
            + "\nInstale com: pip install -r requirements.txt"
        )


def open_capture(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam/fonte: {source}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def filter_result_boxes(result, model_names: dict[int, str], target_classes: set[str] | None):
    if not target_classes:
        return result

    keep_indexes = []
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model_names[class_id]
        if class_name in target_classes:
            keep_indexes.append(index)

    result.boxes = result.boxes[keep_indexes]
    return result


def draw_header(frame, text: str) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 44), (20, 20, 20), -1)
    cv2.putText(
        frame,
        text,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_frame(frame, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"webcam_yolo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(file_path), frame)
    return file_path


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    model = YOLO(args.weights)
    target_classes = set(args.target_classes) if args.target_classes else None
    save_dir = Path(args.save_dir) if args.save_dir else None

    capture = open_capture(
        parse_source(args.source),
        width=args.camera_width,
        height=args.camera_height,
    )

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print("Teste YOLO webcam iniciado.")
    print("Teclas: q = sair | s = salvar frame")
    print(f"Modelo: {args.weights}")
    if target_classes:
        print(f"Filtro de classes: {', '.join(sorted(target_classes))}")

    frame_counter = 0
    fps = 0.0
    last_fps_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Frame nao recebido da webcam.")
                break

            frame_counter += 1
            result = model.predict(frame, **predict_kwargs)[0]
            result = filter_result_boxes(result, model.names, target_classes)
            plotted = result.plot()

            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                last_fps_time = now

            detection_count = len(result.boxes)
            header = (
                f"YOLO webcam | deteccoes: {detection_count} | "
                f"conf: {args.conf:.2f} | fps: {fps:.1f}"
            )
            draw_header(plotted, header)

            cv2.imshow("Teste YOLO - Webcam", plotted)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if save_dir is None:
                    save_dir = Path("runs/webcam_object_test")
                saved_path = save_frame(plotted, save_dir)
                print(f"Frame salvo em: {saved_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
