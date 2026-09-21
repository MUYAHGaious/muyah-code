"""Live visualization of the agent: a local web page fed by the event stream (see `muyah_code.events`)."""

from muyah_code.viz.server import VizServer, find_events_file

__all__ = ["VizServer", "find_events_file"]
