"""Entry point for python -m forge and the `forge` console script."""


def main() -> None:
    from forge.app import ForgeApp

    app = ForgeApp()
    app.run()


if __name__ == "__main__":
    main()
