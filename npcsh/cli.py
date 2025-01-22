import argparse
from .serve import start_flask_server


# from npcsh.helpers import init_npc_project
#


def main():
    parser = argparse.ArgumentParser(description="NPC utilities")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the Flask server")
    serve_parser.add_argument("--port", "-p", help="Optional port")

    # Init command
    # init_parser = subparsers.add_parser("init", help="Initialize a new NPC project")
    # init_parser.add_argument(
    #    "directory", nargs="?", default=".", help="Directory to initialize project in"
    # )

    args = parser.parse_args()

    if args.command == "serve":
        start_flask_server(port=args.port if args.port else 5337)
    # elif args.command == 'init':
    #    init_npc_project(args.directory)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
