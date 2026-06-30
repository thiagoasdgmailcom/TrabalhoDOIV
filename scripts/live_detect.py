from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_DEFECT_CLASSES = [
    "spaghetti",
    "warping",
    "stringing",
    "layer_shift",
    "under_extrusion",
]


def parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteccao em tempo real de falhas em impressao 3D."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/yolov8n_3d_print_defects/weights/best.pt",
        help="Caminho para os pesos treinados.",
    )
    parser.add_argument("--source", default="0", help="Indice da camera ou video.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--defect-classes", nargs="+", default=DEFAULT_DEFECT_CLASSES)
    parser.add_argument(
        "--alert-frames",
        type=int,
        default=5,
        help="Quantidade de frames consecutivos para confirmar alerta.",
    )
    parser.add_argument("--save-video", default=None, help="Caminho para salvar video.")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    defect_classes = set(args.defect_classes)

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte de video: {args.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = open_video_writer(args.save_video, fps, width, height)

    consecutive_defect_frames = 0
    frame_count = 0

    print("Pressione 'q' para encerrar.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_count += 1
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
            result = results[0]
            plotted = result.plot()

            detected_defects = []
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                if class_name in defect_classes:
                    detected_defects.append((class_name, confidence))

            if detected_defects:
                consecutive_defect_frames += 1
            else:
                consecutive_defect_frames = 0

            alert_active = consecutive_defect_frames >= args.alert_frames
            if alert_active:
                names = ", ".join(sorted({name for name, _ in detected_defects}))
                cv2.rectangle(plotted, (0, 0), (width, 58), (0, 0, 255), -1)
                cv2.putText(
                    plotted,
                    f"ALERTA DE FALHA: {names}",
                    (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(plotted)

            cv2.imshow("YOLOv8 - Impressao 3D", plotted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
