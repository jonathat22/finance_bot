"""Dispatcher for `python -m finance_bot <subcommand>`."""

import sys


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m finance_bot <command> [args]\n"
            "\nCommands:\n"
            "  data        Fetch financial data for a ticker\n"
            "  valuation   Run value investing analysis\n"
            "  research    Web research with value investing lens\n"
            "  visualize   Generate ASCII charts and tables\n",
            file=sys.stderr,
        )
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module works
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "data":
        from finance_bot.data import main as data_main

        data_main()
    elif command == "valuation":
        from finance_bot.valuation import main as valuation_main

        valuation_main()
    elif command == "research":
        from finance_bot.research import main as research_main

        research_main()
    elif command == "visualize":
        from finance_bot.visualize import main as visualize_main

        visualize_main()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
