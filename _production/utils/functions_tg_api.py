import json
import logging


def create_session_string(client, config, config_path):
    session_string = client.session.save()
    config["tg_string_session"] = session_string
    try:
        with open(config_path, "w") as file:
            json.dump(config, file)
        logging.info(
            "Session string saved. Please rerun the script to use the new session string."
        )

    except IOError as error:
        logging.error(f"Error writing session string to config: {error}")
        raise


def get_channel_link_header(entity):
    if hasattr(entity, "username") and entity.username:
        return f"https://t.me/{entity.username}/"

    elif hasattr(entity, "id"):
        return f"https://t.me/c/{entity.id}/"

    logging.warning("Entity lacks 'username' and 'id' attributes")
    return None
