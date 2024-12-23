import json
import logging


def create_session_string(client, config, config_path):
    """Save Telegram session string to config file.

    Args:
        client: Telegram client instance
        config (dict): Configuration dictionary
        config_path (str): Path to config file

    Raises:
        IOError: If unable to write to config file
    """
    config["tg_string_session"] = client.session.save()
    try:
        with open(config_path, "w") as file:
            json.dump(config, file, indent=4)
        logging.info(
            "Session string saved. Please rerun the script to use the new session string."
        )
    except IOError as error:
        logging.error(f"Error writing session string to config: {error}")
        raise


def get_channel_link_header(entity):
    """Get Telegram channel/chat link from entity.

    Args:
        entity: Telegram entity object (channel/chat)

    Returns:
        str: Channel/chat URL or None if neither username nor ID is available
    """
    if hasattr(entity, "username") and entity.username:
        return f"https://t.me/{entity.username}/"
    if hasattr(entity, "id"):
        return f"https://t.me/c/{entity.id}/"

    logging.warning("Entity lacks 'username' and 'id' attributes")
    return None
