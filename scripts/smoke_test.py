"""Check the saved LeafVision crop classifier on one labeled image."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    from src import CROP_CLASSES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Saved Keras checkpoint")
    parser.add_argument("--image", required=True, type=Path, help="JPEG to classify")
    parser.add_argument("--expected", required=True, choices=CROP_CLASSES, help="Ground-truth crop")
    args = parser.parse_args()

    for flag, path in (("--model", args.model), ("--image", args.image)):
        if not path.is_file():
            parser.error(f"{flag} must point to an existing file: {path}")

    import tensorflow as tf

    from src.preprocessing import IMAGE_SIZE, load_jpeg

    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=True)
    input_shape = (None, *IMAGE_SIZE, 3)
    output_shape = (None, len(CROP_CLASSES))
    if model.input_shape != input_shape or model.output_shape != output_shape:
        parser.error(
            f"Expected model shapes {input_shape} -> {output_shape}, "
            f"got {model.input_shape} -> {model.output_shape}"
        )

    image = load_jpeg(args.image)
    scores = model(image[tf.newaxis, ...], training=False)[0]
    tf.debugging.assert_all_finite(scores, "The model returned a non-finite score")
    tf.debugging.assert_near(tf.reduce_sum(scores), 1.0, atol=1e-5)

    class_index = int(tf.argmax(scores))
    predicted = CROP_CLASSES[class_index]
    print(f"expected={args.expected} predicted={predicted} score={float(scores[class_index]):.8f}")
    return 0 if predicted == args.expected else 1


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
