from __future__ import annotations as _annotations

from collections.abc import Sequence
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
    if isinstance(content, UploadedFile) and content.file_id in mounted_files:
        mounted_file = mounted_files[content.file_id]
        label = _uploaded_file_display_identifier(content, mounted_file)
        media_type = _uploaded_file_display_media_type(content, mounted_file)
        media_label = f' ({media_type})' if media_type else ''
        return f'File {label}{media_label} is available in the code execution container.', True

    if isinstance(content, list):
        content_items = cast(list[Any], content)
        updated_items: list[Any] = []
        changed = False
        for item in content_items:
            updated_item, item_changed = _replace_mounted_file_content(item, mounted_files)
            changed = changed or item_changed
            updated_items.append(updated_item)
        return updated_items if changed else content_items, changed

    return content, False


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
