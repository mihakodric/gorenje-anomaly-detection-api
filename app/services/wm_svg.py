"""Helpers for rendering a colorized washing machine SVG."""

from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

from app.core.config import settings


SVG_NAMESPACES = {
    "": "http://www.w3.org/2000/svg",
    "inkscape": "http://www.inkscape.org/namespaces/inkscape",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
}
INKSCAPE_LABEL_ATTR = "{http://www.inkscape.org/namespaces/inkscape}label"
SOURCE_COMPONENT_COLORS = ("#669900", "#FFCC00", "#CC0000")
COMPONENT_LABELS = {
    "heater": "Heater",
    "motor": "Motor",
    "pump": "Pump",
}
WM_SVG_PATH = Path(__file__).resolve().parents[1] / "assets" / "svg" / "WM.svg"


for prefix, namespace in SVG_NAMESPACES.items():
    ET.register_namespace(prefix, namespace)


def build_wm_svg(component_colors: Mapping[str, str]) -> str:
    """Return WM.svg as an HTML-safe SVG fragment with recolored component groups."""
    root = ET.fromstring(WM_SVG_PATH.read_text(encoding="utf-8"))

    for group in root.iterfind(".//svg:g", {"svg": SVG_NAMESPACES[""]}):
        label = group.attrib.get(INKSCAPE_LABEL_ATTR)
        if label not in COMPONENT_LABELS.values():
            continue

        component_name = next(
            name for name, component_label in COMPONENT_LABELS.items() if component_label == label
        )
        color = component_colors[component_name]
        _colorize_group(group, color)

    svg_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    return svg_bytes.decode("utf-8")


def _colorize_group(group: ET.Element, color: str) -> None:
    """Replace known component colors on a target group and its descendants."""
    for element in group.iter():
        for attr_name in ("style", "stroke", "fill"):
            attr_value = element.attrib.get(attr_name)
            if not attr_value:
                continue
            element.attrib[attr_name] = _replace_known_component_colors(attr_value, color)


def _replace_known_component_colors(value: str, color: str) -> str:
    """Replace only the known component colors and keep other style content unchanged."""
    updated_value = value
    for known_color in _known_component_colors():
        updated_value = updated_value.replace(known_color, color)
    return updated_value


def _known_component_colors() -> tuple[str, ...]:
    """Return source and configured component colors that may need replacement."""
    return tuple(
        dict.fromkeys(
            SOURCE_COMPONENT_COLORS
            + (
                settings.component_ok_color,
                settings.component_warning_color,
                settings.component_failing_color,
            )
        )
    )
