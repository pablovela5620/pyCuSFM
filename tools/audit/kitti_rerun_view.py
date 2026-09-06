# SPDX-License-Identifier: Apache-2.0
"""Frame the KITTI 06 comparison recording for a report screenshot.

`colsfm.bench_cli --save` writes a comparison `.rrd` whose saved blueprint opens
on a three-panel layout with a default camera. KITTI 06 is a 457 m corridor only
22 m wide, so that camera looks straight down the road and the loop reads as a
vertical smear. Dragging the orbit camera by hand is not reproducible; this sends
a blueprint with an explicit `EyeControls3D` instead, so the same command always
produces the same frame.

The world frame is KITTI's camera 0 of frame 0: x right, y down, z forward. The
eye therefore sits above the route (negative y), tilted a little along -x, with
`eye_up` on +x so the corridor runs left to right across a wide, short viewport.

Run against a viewer already showing the recording:

    rerun --headless --port 9878 /tmp/colsfm_runs/kitti_raco_caspar_compare.rrd &
    pixi run -e colsfm python -m tools.audit.kitti_rerun_view

Then drive the screenshot through the viewer MCP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import rerun as rr
import rerun.blueprint as rrb
import tyro

ROUTE_CENTRE: Final[tuple[float, float, float]] = (-10.0, -1.0, 72.0)
"""Mid-point of the KITTI 06 ground-truth route, metres, in the frame-0 camera frame."""

ROUTE_SOUTH_END_Z: Final[float] = -157.0
"""World z of the southern turnaround, the far end of the outbound carriageway."""


@dataclass(frozen=True)
class KittiRerunViewConfig:
    """Where the viewer is and how far back to put the eye."""

    endpoint: str = "rerun+http://127.0.0.1:9878/proxy"
    """gRPC proxy URL of the running viewer."""
    application_id: str = "colsfm_bench"
    """Application id of the comparison recording."""
    recording_id: str = "robocap_compare"
    """Recording id `colsfm.bench_cli` writes for a two-run comparison."""
    eye_height_meters: float = 4.0
    """How far above the road the eye sits."""
    eye_behind_meters: float = -137.0
    """How far south of the route's southern turnaround the eye sits."""
    look_ahead_z_meters: float = 130.0
    """Where along the corridor the eye points, in world z."""


def main(config: KittiRerunViewConfig) -> None:
    """Send a wide top-down blueprint to the running viewer.

    Args:
        config: Viewer endpoint, recording identity and eye placement.
    """
    centre_x, _, _ = ROUTE_CENTRE
    position: tuple[float, float, float] = (
        centre_x,
        -config.eye_height_meters,
        ROUTE_SOUTH_END_Z - config.eye_behind_meters,
    )
    look_target: tuple[float, float, float] = (centre_x, -1.0, config.look_ahead_z_meters)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                contents="/world/**",
                name="KITTI 06 — blob and colsfm RaCo native + CASPAR",
                eye_controls=rrb.EyeControls3D(position=position, look_target=look_target, eye_up=(0.0, -1.0, 0.0)),
                time_ranges=rrb.VisibleTimeRange(
                    "frame_time",
                    start=rrb.TimeRangeBoundary.infinite(),
                    end=rrb.TimeRangeBoundary.infinite(),
                ),
            ),
            rrb.TextDocumentView(origin="/report", name="Benchmark report"),
            column_shares=[2.15, 1.0],
        ),
        collapse_panels=True,
    )
    rr.init(config.application_id, recording_id=config.recording_id)
    rr.connect_grpc(url=config.endpoint)
    rr.send_blueprint(blueprint, make_active=True, make_default=True)
    print(f"sent blueprint: eye {position} looking at {look_target}")


if __name__ == "__main__":
    main(tyro.cli(KittiRerunViewConfig))
