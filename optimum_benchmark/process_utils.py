import os
import pickle
import tempfile
import uuid
from logging import Logger
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional, TypeVar

DeserializedType = TypeVar("DeserializedType")


def sync_with_parent(child_connection: Connection) -> None:
    child_connection.recv()
    child_connection.send(0)


def sync_with_child(parent_connection: Connection) -> None:
    parent_connection.send(0)
    parent_connection.recv()


def send_serializable(
    connection: Connection, obj: Any, logger: Logger, serializer: Callable[[Any], bytes] = pickle.dumps
) -> None:
    """Send any serializable object via temporary file to avoid pipe buffer issues.

    Args:
        connection: The connection to send file path to
        obj: Any serializable object to send
        logger: Logger for debugging
        serializer: Function to serialize object to bytes (default: pickle.dumps)
    """

    unique_id = str(uuid.uuid4())
    temp_file_path = os.path.join(tempfile.gettempdir(), f"transfer_{unique_id}.pickle")

    logger.debug(f"Serializing object to temporary file: {temp_file_path}")
    with open(temp_file_path, "wb") as f:
        serializer(obj, f)

    file_size = os.path.getsize(temp_file_path)
    logger.debug(f"Serialized object to file of size {file_size} bytes")

    connection.send(temp_file_path)
    logger.debug(f"Sent temporary file path: {temp_file_path}")


def receive_serializable(
    connection: Connection, logger: Logger, deserializer: Callable[[bytes], DeserializedType] = pickle.loads
) -> DeserializedType:
    """Receive any serializable object via temporary file to avoid pipe buffer issues.

    Args:
        connection: The connection to receive file path from
        logger: Logger for debugging
        deserializer: Function to deserialize bytes back to object (default: pickle.loads)
                      Note: This should accept a file object if passed

    Returns:
        The complete deserialized object
    """
    logger.debug("Waiting to receive file path")
    file_path = connection.recv()
    logger.debug(f"Received file path: {file_path}")

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transfer file not found: {file_path}")

        file_size = os.path.getsize(file_path)
        logger.debug(f"Loading object from file of size {file_size} bytes")

        with open(file_path, "rb") as f:
            try:
                obj = deserializer(f)
            except TypeError:
                obj = deserializer(f.read())

        logger.debug("Successfully loaded object from file")
        connection.send("file_received")

        return obj
    except Exception as e:
        logger.error(f"Error reading transfer file: {e}")
        raise
    finally:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")
