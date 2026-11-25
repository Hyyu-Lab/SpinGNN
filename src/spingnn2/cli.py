"""Command-line entry point for compact checkpoint export."""

import argparse

from .export import export_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spingnn2-export", description=__doc__)
    parser.add_argument("checkpoint", help="Training directory or .pth state_dict")
    parser.add_argument("-o", "--output", required=True, help="Output stem")
    parser.add_argument(
        "--config", help="Training config (default: adjacent config.yaml)"
    )
    parser.add_argument("--source-root", help="SpinGNN2 source checkout")
    args = parser.parse_args(argv)
    paths = export_checkpoint(
        args.checkpoint, args.output, config=args.config, source_root=args.source_root
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
