from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any, cast

from ..messages import ModelMessage, ModelRequest, ModelRequestPart, ToolReturnPart, UploadedFile
from ..native_tools import AbstractNativeTool, CodeExecutionTool


def replace_code_execution_files_in_tool_returns(
    messages: list[ModelMessage],
    native_tools: Sequence[AbstractNativeTool],
    provider_name: str,
) -> list[ModelMessage]:
    mounted_files = {
        file.file_id: file
        for tool in native_tools
        if isinstance(tool, CodeExecutionTool) and tool.files
        for file in tool.files
        if file.provider_name == provider_name
    }
    if not mounted_files:
        return messages

    updated_messages: list[ModelMessage] = []
    changed = False
    for message in messages:
        if not isinstance(message, ModelRequest):
            updated_messages.append(message)
            continue

        updated_parts, message_changed = _replace_message_tool_return_files(message, mounted_files)
        updated_messages.append(replace(message, parts=updated_parts) if message_changed else message)
        changed = changed or message_changed

    return updated_messages if changed else messages


def _replace_message_tool_return_files(
    message: ModelRequest, mounted_files: dict[str, UploadedFile]
) -> tuple[list[ModelRequestPart], bool]:
    updated_parts: list[ModelRequestPart] = []
    changed = False
    for part in message.parts:
        updated_part = part
        if isinstance(part, ToolReturnPart):
            content, content_changed = _replace_mounted_file_content(part.content, mounted_files)
            if content_changed:
                updated_part = replace(part, content=content)
                changed = True
        updated_parts.append(updated_part)
    return updated_parts, changed


def _replace_mounted_file_content(content: Any, mounted_files: dict[str, UploadedFile]) -> tuple[Any, bool]:
    if isinstance(content, UploadedFile):
        mounted_file = mounted_files.get(content.file_id)
        if mounted_file is not None and content.provider_name == mounted_file.provider_name:
            label = _uploaded_file_display_identifier(content, mounted_file)
            media_type = _uploaded_file_display_media_type(content, mounted_file)
            media_label = f' ({media_type})' if media_type else ''
            return f'File {label}{media_label} is available in the code execution container.', True

    if isinstance(content, Mapping):
        mapping = cast(Mapping[str, Any], content)
        updated_mapping: dict[str, Any] = {}
        changed = False
        for key, value in mapping.items():
            updated_value, value_changed = _replace_mounted_file_content(value, mounted_files)
            changed = changed or value_changed
            updated_mapping[key] = updated_value
        return _rebuild_mapping(mapping, updated_mapping) if changed else mapping, changed

    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        sequence = cast(Sequence[Any], content)
        updated_items: list[Any] = []
        changed = False
        for item in sequence:
            updated_item, item_changed = _replace_mounted_file_content(item, mounted_files)
            changed = changed or item_changed
            updated_items.append(updated_item)
        return _rebuild_sequence(sequence, updated_items) if changed else sequence, changed

    return content, False


def _rebuild_mapping(content: Mapping[str, Any], updated: dict[str, Any]) -> Mapping[str, Any]:
    if type(content) is dict:
        return updated
    try:
        constructor = cast(Callable[[dict[str, Any]], Mapping[str, Any]], type(content))
        return constructor(updated)
    except (TypeError, ValueError):
        return updated


def _rebuild_sequence(content: Sequence[Any], updated: list[Any]) -> Sequence[Any]:
    if isinstance(content, tuple):
        return tuple(updated)
    if type(content) is list:
        return updated
    try:
        constructor = cast(Callable[[list[Any]], Sequence[Any]], type(content))
        return constructor(updated)
    except (TypeError, ValueError):
        return updated


def _uploaded_file_display_identifier(content: UploadedFile, mounted_file: UploadedFile) -> str:
    if _uploaded_file_identifier_has_suffix(content.identifier):
        return content.identifier
    if mounted_file.identifier:
        return mounted_file.identifier
    return content.identifier or content.file_id


def _uploaded_file_display_media_type(content: UploadedFile, mounted_file: UploadedFile) -> str | None:
    if content.media_type and content.media_type != 'application/octet-stream':
        return content.media_type
    return mounted_file.media_type or content.media_type


def _uploaded_file_identifier_has_suffix(identifier: str | None) -> bool:
    return bool(identifier and PurePosixPath(identifier).suffix)
