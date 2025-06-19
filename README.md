# UnitreeH1VoiceAI

This project contains various client scripts for interacting with a voice control system.

## SoundCloud integration

A new module `soundcloud_player.py` provides basic playback from SoundCloud. To use it, create a `config.json` with your SoundCloud API credentials based on `config_template.json` and set the environment variables `SOUNDCLOUD_CLIENT_ID` and `SOUNDCLOUD_CLIENT_SECRET`.

In `Client_For_Robot.py` you can say commands starting with `soundcloud` followed by a search query. The first result from SoundCloud will be played through the selected output device.

```bash
export SOUNDCLOUD_CLIENT_ID=your_client_id
export SOUNDCLOUD_CLIENT_SECRET=your_client_secret
python Client_For_Robot.py
```

Use the voice command `soundcloud <artist or track name>` to play music.
