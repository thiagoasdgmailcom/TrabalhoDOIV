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


DEFAULT_PRINTED_CLASSES = [
    "printed_part",
    "printed_object",
    "3d_printed_object",
    "successful_print",
    "3d_print",
]


def parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta apenas objetos impressos em 3D usando um modelo YOLO treinado."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/yolov8n_3d_print_defects/weights/best.pt",
        help="Caminho do modelo treinado para detectar objetos impressos em 3D.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam, video ou stream. Ex.: 0, 1, video.mp4.",
    )
    parser.add_argument(
        "--printed-classes",
        nargs="+",
        default=DEFAULT_PRINTED_CLASSES,
        help="Nomes das classes que representam objetos impressos em 3D.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Ex.: cpu, 0, cuda:0.")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument(
        "--save-dir",
        default="runs/detect_3d_printed_objects",
        help="Pasta para salvar frames quando pressionar 's'.",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="Opcional: caminho para salvar o video com as deteccoes.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
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


def normalize_class_name(class_name: str) -> str:
    return (
        class_name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def normalize_model_names(model_names) -> dict[int, str]:
    if not isinstance(model_names, dict):
        return dict(enumerate(model_names))
    return model_names


def resolve_printed_class_ids(
    model_names: dict[int, str],
    printed_classes: list[str],
) -> set[int]:
    wanted = {normalize_class_name(class_name) for class_name in printed_classes}
    return {
        class_id
        for class_id, class_name in model_names.items()
        if normalize_class_name(class_name) in wanted
    }


def open_capture(source: str | int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {source}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def open_video_writer(
    output_path: str | None,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter | None:
    if output_path is None:
        return None

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps or 30.0, (width, height))


def draw_detection(frame, xyxy, label: str, confidence: float) -> None:
    x1, y1, x2, y2 = [int(value) for value in xyxy]
    color = (0, 190, 80)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    y_text = max(0, y1 - text_height - 8)
    cv2.rectangle(
        frame,
        (x1, y_text),
        (x1 + text_width + 8, y_text + text_height + 8),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x1 + 4, y_text + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
    file_path = save_dir / f"printed_object_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(file_path), frame)
    return file_path


def main() -> None:
    args = parse_args()
    ensure_dependencies()

    weights_path = Path(args.weights)
    if not weights_path.exists() and args.weights != "yolov8n.pt":
        raise SystemExit(
            f"Modelo nao encontrado: {weights_path}\n"
            "Treine o modelo ou informe outro caminho em --weights."
        )

    model = YOLO(args.weights)
    model_names = normalize_model_names(model.names)
    printed_class_ids = resolve_printed_class_ids(model_names, args.printed_classes)
    if not printed_class_ids:
        available = ", ".join(model_names.values())
        wanted = ", ".join(args.printed_classes)
        raise SystemExit(
            "O modelo carregado nao possui nenhuma classe de objeto impresso em 3D.\n"
            f"Classes procuradas: {wanted}\n"
            f"Classes do modelo: {available}\n"
            "Use um modelo treinado com a classe printed_part ou ajuste --printed-classes."
        )

    capture = open_capture(
        parse_source(args.source),
        width=args.camera_width,
        height=args.camera_height,
    )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or args.camera_width)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.camera_height)
    fps_source = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = open_video_writer(args.save_video, fps_source, width, height)

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print("Deteccao de objetos impressos em 3D iniciada.")
    print("Teclas: q = sair | s = salvar frame")
    print(f"Modelo: {args.weights}")
    print(
        "Classes usadas: "
        + ", ".join(model_names[class_id] for class_id in sorted(printed_class_ids))
    )

    frame_count = 0
    fps_counter = 0
    fps = 0.0
    last_fps_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1
            fps_counter += 1

            result = model.predict(frame, **predict_kwargs)[0]
            display_frame = frame.copy()
            printed_detections = 0

            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in printed_class_ids:
                    continue

                confidence = float(box.conf[0])
                xyxy = box.xyxy[0].detach().cpu().tolist()
                draw_detection(
                    display_frame,
                    xyxy=xyxy,
                    label=model_names[class_id],
                    confidence=confidence,
                )
                printed_detections += 1

            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                last_fps_time = now

            draw_header(
                display_frame,
                (
                    "Objetos impressos 3D | "
                    f"deteccoes: {printed_detections} | "
                    f"conf: {args.conf:.2f} | fps: {fps:.1f}"
                ),
            )

            if writer is not None:
                writer.write(display_frame)

            cv2.imshow("Deteccao - Objetos Impressos 3D", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                saved_path = save_frame(display_frame, Path(args.save_dir))
                print(f"Frame salvo em: {saved_path}")

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
