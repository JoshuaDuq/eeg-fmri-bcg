"""Strict BrainVision Core Data Format marker-file I/O."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from os import PathLike
from pathlib import Path

_MARKER_FILE_IDENTIFIERS = frozenset(
    {
        "BrainVision Data Exchange Marker File Version 1.0",
        "BrainVision Data Exchange Marker File Version 2.0",
    }
)
_WRITTEN_MARKER_FILE_IDENTIFIER = (
    "BrainVision Data Exchange Marker File Version 1.0"
)
_WRITTEN_MARKER_FILE_IDENTIFIER_WITH_USER_INFOS = (
    "BrainVision Data Exchange Marker File Version 2.0"
)


class BrainVisionMarkerError(ValueError):
    """Raised when BrainVision marker data or syntax is invalid."""


def _validate_marker_text(value: str, *, field_name: str, allow_empty: bool) -> None:
    if not isinstance(value, str):
        raise BrainVisionMarkerError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise BrainVisionMarkerError(f"{field_name} cannot be empty")
    if any(character in value for character in ("\n", "\r", "\0")):
        raise BrainVisionMarkerError(f"{field_name} contains an invalid character")
    if r"\1" in value:
        message = f"{field_name} contains the ambiguous BrainVision comma placeholder"
        raise BrainVisionMarkerError(message)


def _validate_marker_integer(value: int, *, field_name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise BrainVisionMarkerError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )


def _validate_marker_date(value: str | None) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 20
        or not value.isascii()
        or not value.isdigit()
    ):
        raise BrainVisionMarkerError(
            "date must use the 20-digit YYYYMMDDhhmmssuuuuuu format"
        )
    try:
        datetime.strptime(value, "%Y%m%d%H%M%S%f")
    except ValueError as error:
        raise BrainVisionMarkerError("date is not a valid calendar date") from error


@dataclass(frozen=True, slots=True)
class BrainVisionMarker:
    """One losslessly represented BrainVision marker."""

    marker_type: str
    description: str
    position: int
    size: int
    channel: int
    date: str | None = None
    user_infos: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        _validate_marker_text(
            self.marker_type,
            field_name="marker_type",
            allow_empty=False,
        )
        _validate_marker_text(
            self.description,
            field_name="description",
            allow_empty=True,
        )
        _validate_marker_integer(self.position, field_name="position", minimum=1)
        _validate_marker_integer(self.size, field_name="size", minimum=0)
        _validate_marker_integer(self.channel, field_name="channel", minimum=0)
        _validate_marker_date(self.date)
        if not isinstance(self.user_infos, tuple) or any(
            not isinstance(user_info, tuple)
            or len(user_info) < 3
            or any(not isinstance(field, str) for field in user_info)
            for user_info in self.user_infos
        ):
            raise BrainVisionMarkerError(
                "user_infos must contain tuples of string fields"
            )
        for user_info in self.user_infos:
            for field in user_info:
                _validate_marker_text(
                    field,
                    field_name="user_info",
                    allow_empty=False,
                )


def _parse_marker_integer_field(value: str, *, field_name: str) -> int:
    if not value.isascii() or not value.isdigit():
        raise BrainVisionMarkerError(
            f"marker {field_name} must contain only decimal digits"
        )
    return int(value)


def _parse_marker_line(line: str) -> tuple[int, BrainVisionMarker]:
    key, separator, value = line.partition("=")
    index_text = key.removeprefix("Mk")
    if (
        not separator
        or not index_text.isascii()
        or not index_text.isdigit()
        or index_text != str(int(index_text))
    ):
        raise BrainVisionMarkerError(f"malformed marker declaration: {line}")

    fields = value.split(",")
    if len(fields) not in (5, 6):
        raise BrainVisionMarkerError(f"malformed marker declaration: {line}")

    marker_type, description, position, size, channel = fields[:5]
    date = (fields[5] or None) if len(fields) == 6 else None
    try:
        marker = BrainVisionMarker(
            marker_type=marker_type.replace(r"\1", ","),
            description=description.replace(r"\1", ","),
            position=_parse_marker_integer_field(position, field_name="position"),
            size=_parse_marker_integer_field(size, field_name="size"),
            channel=_parse_marker_integer_field(channel, field_name="channel"),
            date=date,
        )
    except ValueError as error:
        raise BrainVisionMarkerError(f"invalid marker declaration: {line}") from error
    return int(index_text), marker


def _parse_common_info_line(line: str, common_infos: dict[str, str]) -> None:
    key, separator, value = line.partition("=")
    if not separator or key not in ("Codepage", "DataFile"):
        raise BrainVisionMarkerError(f"unsupported Common Infos line: {line}")
    if key in common_infos:
        raise BrainVisionMarkerError(f"duplicate Common Infos {key} declaration")
    common_infos[key] = value


def _enter_marker_file_section(line: str, sections: list[str]) -> str:
    expected_sections = (
        ("[Common Infos]",)
        if not sections
        else ("[Marker Infos]",)
        if len(sections) == 1
        else ("[Marker User Infos]",)
        if len(sections) == 2
        else ()
    )
    if line not in expected_sections:
        raise BrainVisionMarkerError(f"unexpected or duplicate section: {line}")
    sections.append(line)
    return line


def _validate_data_file_name(data_file_name: str) -> None:
    if not isinstance(data_file_name, str) or not data_file_name:
        raise BrainVisionMarkerError("data_file_name must be a nonempty string")
    if any(character in data_file_name for character in ("\n", "\r", "\0")):
        raise BrainVisionMarkerError("data_file_name contains an invalid character")
    if data_file_name in (".", "..") or any(
        separator in data_file_name for separator in ("/", "\\")
    ):
        raise BrainVisionMarkerError(
            "data_file_name must be a same-directory filename"
        )


def read_brainvision_markers(
    path: str | PathLike[str],
) -> tuple[str, tuple[BrainVisionMarker, ...]]:
    """Read a strict BrainVision marker file without losing marker fields."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] not in _MARKER_FILE_IDENTIFIERS:
        raise BrainVisionMarkerError("invalid BrainVision marker-file identifier")

    section: str | None = None
    sections: list[str] = []
    common_infos: dict[str, str] = {}
    indexed_markers: list[tuple[int, BrainVisionMarker]] = []
    user_info_records: list[tuple[int, int, tuple[str, ...]]] = []
    for line in lines[1:]:
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            section = _enter_marker_file_section(line, sections)
        elif section == "[Common Infos]":
            _parse_common_info_line(line, common_infos)
        elif section == "[Marker Infos]":
            indexed_markers.append(_parse_marker_line(line))
        elif section == "[Marker User Infos]":
            user_info_records.append(_parse_user_info_line(line))
        else:
            raise BrainVisionMarkerError(f"content outside a supported section: {line}")

    if sections not in (
        ["[Common Infos]", "[Marker Infos]"],
        ["[Common Infos]", "[Marker Infos]", "[Marker User Infos]"],
    ):
        raise BrainVisionMarkerError(
            "marker file must contain one Common Infos and one Marker Infos section"
        )
    if common_infos.get("Codepage") != "UTF-8":
        raise BrainVisionMarkerError("Common Infos Codepage must be declared as UTF-8")
    data_file_name = common_infos.get("DataFile")
    if not data_file_name:
        raise BrainVisionMarkerError("missing Common Infos DataFile declaration")
    _validate_data_file_name(data_file_name)
    indices = [index for index, _ in indexed_markers]
    if indices != list(range(1, len(indices) + 1)):
        raise BrainVisionMarkerError(
            "marker indices must appear in exact Mk1, Mk2, ... file order"
        )
    markers = _attach_user_infos(indexed_markers, user_info_records)
    return data_file_name, markers


def _parse_user_info_line(line: str) -> tuple[int, int, tuple[str, ...]]:
    key, separator, value = line.partition("=")
    property_index_text = key.removeprefix("Prop")
    if (
        not separator
        or not property_index_text.isascii()
        or not property_index_text.isdigit()
        or property_index_text != str(int(property_index_text))
    ):
        raise BrainVisionMarkerError(f"malformed marker user info: {line}")
    fields = value.split(",")
    if len(fields) < 4 or not fields[0].startswith("Mk"):
        raise BrainVisionMarkerError(f"malformed marker user info: {line}")
    marker_index_text = fields[0][2:]
    if (
        not marker_index_text.isascii()
        or not marker_index_text.isdigit()
        or marker_index_text != str(int(marker_index_text))
    ):
        raise BrainVisionMarkerError(f"malformed marker user info: {line}")
    decoded_fields = tuple(field.replace(r"\1", ",") for field in fields[1:])
    return int(property_index_text), int(marker_index_text), decoded_fields


def _attach_user_infos(
    indexed_markers: list[tuple[int, BrainVisionMarker]],
    user_info_records: list[tuple[int, int, tuple[str, ...]]],
) -> tuple[BrainVisionMarker, ...]:
    marker_indices = {index for index, _ in indexed_markers}
    property_indices = [record[0] for record in user_info_records]
    if property_indices != list(range(1, len(property_indices) + 1)):
        raise BrainVisionMarkerError(
            "marker user info properties must be in exact Prop1, Prop2, ... order"
        )
    user_infos_by_marker: dict[int, list[tuple[str, ...]]] = {}
    for _, marker_index, user_info in user_info_records:
        if marker_index not in marker_indices:
            raise BrainVisionMarkerError(
                "marker user info refers to an unknown marker"
            )
        user_infos_by_marker.setdefault(marker_index, []).append(user_info)
    return tuple(
        replace(
            marker,
            user_infos=tuple(user_infos_by_marker.get(index, ())),
        )
        for index, marker in indexed_markers
    )


def _format_marker_line(index: int, marker: BrainVisionMarker) -> str:
    marker_type = marker.marker_type.replace(",", r"\1")
    description = marker.description.replace(",", r"\1")
    fields = [
        marker_type,
        description,
        str(marker.position),
        str(marker.size),
        str(marker.channel),
    ]
    if marker.date is not None:
        fields.append(marker.date)
    return f"Mk{index}={','.join(fields)}"


def write_brainvision_markers(
    path: str | PathLike[str],
    data_file_name: str,
    markers: Iterable[BrainVisionMarker],
) -> None:
    """Write a lossless BrainVision marker file without overwriting existing data."""
    _validate_data_file_name(data_file_name)
    marker_values = tuple(markers)
    if any(not isinstance(marker, BrainVisionMarker) for marker in marker_values):
        raise BrainVisionMarkerError("markers must contain BrainVisionMarker instances")

    has_user_infos = any(marker.user_infos for marker in marker_values)
    lines = [
        (
            _WRITTEN_MARKER_FILE_IDENTIFIER_WITH_USER_INFOS
            if has_user_infos
            else _WRITTEN_MARKER_FILE_IDENTIFIER
        ),
        "[Common Infos]",
        "Codepage=UTF-8",
        f"DataFile={data_file_name}",
        "",
        "[Marker Infos]",
    ]
    lines.extend(
        _format_marker_line(index, marker)
        for index, marker in enumerate(marker_values, start=1)
    )
    if has_user_infos:
        lines.extend(("", "[Marker User Infos]"))
        property_index = 1
        for marker_index, marker in enumerate(marker_values, start=1):
            for user_info in marker.user_infos:
                encoded_fields = ",".join(
                    field.replace(",", r"\1") for field in user_info
                )
                lines.append(
                    f"Prop{property_index}=Mk{marker_index},{encoded_fields}"
                )
                property_index += 1
    content = "\n".join(lines) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as marker_file:
        marker_file.write(content)
