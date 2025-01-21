import os
from configargparse import ArgParser
import argparse

from bomberman_rl import settings as s


def parse(argv=None):
    parser = ArgParser(default_config_files=[])

    parser.add(
        "--seed",
        type=int,
        help="Seed the env's random number generator for the sake of reproducibility",
    )
    parser.add(
        "--no-gui",
        default=False,
        action="store_true",
        help="Disable GUI rendering to increase speed",
    )
    parser.add(
        "--passive",
        default=False,
        action="store_true",
        help="Passive mode: no external agent",
    )
    parser.add(
        "--train",
        default=False,
        action="store_true",
        help="Whether training callbacks on agent should be called",
    )
    parser.add(
        "--players",
        nargs='*',
        help="Set agents that participate playing",
    )
    parser.add(
        "--learners",
        nargs='*',
        help="Set agents that participate playing and learning",
    )
    parser.add(
        "--match-name",
        help="Match name (used for e.g. displaying, separating recordings, etc.)",
    )
    parser.add(
        "--silence-errors",
        default=False,
        action="store_true",
        help="Ignore errors from agents",
    )
    parser.add(
        "--user-play",
        default=False,
        action="store_true",
        help="Wait for key press until next movement",
    )
    parser.add(
        "--log-dir",
        default=os.path.dirname(os.path.abspath(__file__)) + "/logs"
    )
    parser.add(
        "--video",
        nargs="?",
        const=os.path.dirname(os.path.abspath(__file__)) + "/replays",
        help="Record the session",
    )
    parser.add(
        "--scenario",
        default="classic",
        choices=s.SCENARIOS
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Specify 'fresh' for new weights or timestamp (e.g., '20240315_143022') to load specific checkpoint"
    )
    parser.add(
        "--use-double-dqn",
        default=False,
        action="store_true",
        help="Use double DQN instead of single DQN"
    )

    args = parser.parse_args(argv)

    # Default players
    if args.players is None and args.learners is None:
        args.players = ["rule_based_agent"] * 3

    # Render mode
    if args.video:
        args.render_mode = "rgb_array"
    elif not args.no_gui:
        args.render_mode = "human"
    else:
        args.render_mode = None

    return args