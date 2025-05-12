import pickle
from logging import Logger
from multiprocessing.connection import Connection
from typing import Any, Callable, TypeVar

DeserializedType = TypeVar("DeserializedType")


def sync_with_parent(child_connection: Connection) -> None:
    child_connection.recv()
    child_connection.send(0)


def sync_with_child(parent_connection: Connection) -> None:
    parent_connection.send(0)
    parent_connection.recv()


def send_serializable(
    connection: Connection,
    obj: Any,
    logger: Logger,
    chunk_size: int = 1_000_000,
    serializer: Callable[[Any], bytes] = pickle.dumps,
) -> None:
    """Send any serializable object in chunks to avoid pipe buffer issues.

    Args:
        connection: The connection to send chunks to
        obj: Any serializable object to send
        logger: Logger for debugging
        chunk_size: The size of each chunk in bytes
        serializer: Function to serialize object to bytes (default: pickle.dumps)
    """
    serialized = serializer(obj)

    logger.debug(f"Sending object of size {len(serialized)} bytes")

    for i in range(0, len(serialized), chunk_size):
        chunk = serialized[i : i + chunk_size]
        connection.send(chunk)
        logger.debug(f"Sent chunk of size {len(chunk)} bytes")

    logger.debug("Finished sending object")
    connection.send(None)  # End of transmission
    logger.debug("Sent end of transmission signal")


def receive_serializable(
    connection: Connection, logger: Logger, deserializer: Callable[[bytes], DeserializedType] = pickle.loads
) -> DeserializedType:
    """Receive any serializable object in chunks to avoid pipe buffer issues.

    Args:
        connection: The connection to receive chunks from
        logger: Logger for debugging
        deserializer: Function to deserialize bytes back to object (default: pickle.loads)

    Returns:
        The complete deserialized object
    """
    logger.debug("Receiving object")
    chunks = []

    while True:
        chunk = connection.recv()
        if chunk is None:  # End of transmission
            break
        chunks.append(chunk)
        logger.debug(f"Received chunk of size {len(chunk)} bytes")

    logger.debug("Finished receiving object")
    serialized_data = b"".join(chunks)
    obj = deserializer(serialized_data)
    logger.debug(f"Received object of size {len(serialized_data)} bytes")

    return obj
