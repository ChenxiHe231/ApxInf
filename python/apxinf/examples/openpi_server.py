#!/usr/bin/env python3
"""Minimal websocket policy server (openpi-compatible wire) over a ``apxinf`` policy.

Loads an in-process ``AutoPolicy`` (optionally under a named robot wire preset)
and serves it with ``apxinf.serving.WebsocketPolicyServer`` — the same transport
shell the production launcher (scripts/pi05_openpi_websocket_server.py) uses,
reduced to the essentials. An unmodified ``openpi_client`` connects to it; see
``openpi_client.py`` for the other end.

Requires the ``apxinf_py`` CUDA binding plus the transport deps
(``websockets`` / ``msgpack``; see scripts/requirements-pi05-websocket.txt).

    python examples/openpi_server.py --model-dir /path/to/checkpoint \
        --robot franka_libero \
        --policy-options '{"norm_key":"x2_normal"}'
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from _common import json_object, policy_kwargs  # noqa: E402 (also installs source path shim)

from apxinf import AutoPolicy, available_robots, build_robot_policy
from apxinf.serving import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--robot",
        choices=available_robots(include_aliases=True),
        help="optional named robot wire contract (for example franka_libero)",
    )
    parser.add_argument("--action-dim", type=int, default=0, help="0 keeps the full vector")
    parser.add_argument(
        "--policy-options",
        type=json_object,
        default={},
        metavar="JSON",
        help="extra concrete-policy options as a JSON object",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    options = policy_kwargs(
        args.policy_options,
        device=args.device,
        precision=args.precision,
        action_dim=args.action_dim,
        metadata={"protocol": "openpi.websocket_policy", "precision": args.precision},
    )
    policy = (
        build_robot_policy(args.robot, args.model_dir, **options)
        if args.robot
        else AutoPolicy.from_pretrained(args.model_dir, **options)
    )
    server = WebsocketPolicyServer(policy, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
