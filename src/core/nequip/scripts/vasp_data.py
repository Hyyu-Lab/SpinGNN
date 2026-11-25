"""Thin backward-compatible module entry forwarding to `spingnn2.data.vasp_spin` (`spingnn-data`)."""

from nequip.scripts._source_guard import guard_source_entry


def main(args=None):
    guard_source_entry()
    from spingnn2.data.vasp_spin import main as _spingnn_data_main

    return _spingnn_data_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
