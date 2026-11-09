import argparse
import config
from particle_counter import ParticleCounter


def main():
    parser = argparse.ArgumentParser(description="Video particle counting and classification")
    parser.add_argument("--input", default=config.INPUT_VIDEO, help="Input video path or camera index")
    parser.add_argument("--output", default=config.OUTPUT_VIDEO_FILE, help="Output annotated video")
    parser.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    parser.add_argument("--scale", type=float, default=config.SCALE_FACTOR)
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    args = parser.parse_args()

    config.OUTPUT_VIDEO_FILE = args.output
    if args.no_save:
        config.SAVE_OUTPUT_VIDEO = False
    source = int(args.input) if args.input.isdigit() else args.input
    counter = ParticleCounter(source, scale=args.scale, show_preview=not args.no_preview)
    counter.run()


if __name__ == "__main__":
    main()
