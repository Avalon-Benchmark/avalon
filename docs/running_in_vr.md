# Running Avalon in VR

While Avalon's Mouse & Keyboard mode may be more immediately accessible,
it is really designed to be played in VR.

This guide will focus on generating, playing, and debugging worlds remotely on the Oculus Quest.

Depending on your platform, you may be able to get other headsets and setups to work
(i.e. HTC Vive as a VR device on Linux).

## Dependencies

You'll need:
1. Avalon: `pip install avalon-rl`
2. The custom godot editor build for your platform:
   `python -m avalon.install_godot_binary`
3. The Android SDK for connecting to the oculus if not yet installed,
   [specifically components used by godot export](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_android.html)

   * Ubuntu:
     ```sh
     sudo apt update && sudo apt install android-sdk
     androidsdk # Prints SDK_ROOT
     ```
   * Homebrew:
     ```sh
     brew install --cask android-commandlinetools
     # default, override if desired, and save this lines in your .bashrc or .zshrc
     export ANDROID_SDK=/usr/local/share/android-commandlinetools

     # Install platform tools (adb) and SDK
     sdkmanager --sdk_root="$ANDROID_SDK" \
       "platform-tools" "build-tools;30.0.3" "platforms;android-31" \
       "cmdline-tools;latest" "cmake;3.10.2.4988404" "ndk;21.4.7075529"

     # again, you'll want to save this in your .*rc
     export PATH="$ANDROID_SDK/platform-tools:$PATH"
     ```
   * [android studio](https://developer.android.com/studio#downloads)
     following [these steps](https://developer.android.com/studio/intro/update#sdk-manager)

   Check your installation with `adb --version`


## Generating Sample Worlds

Avalon comes with a CLI tool for generating worlds for interactive use:
```sh
# python -m avalon.for_humans generate_evaluation_worlds --help # for all options
# python -m avalon.for_humans # to view all available commands
python -m avalon.for_humans generate_evaluation_worlds eat,hunt --worlds_per_task=10
```

This will generate 20 worlds into the internal worlds directory so that they're visible in the editor.
The placement and naming of these worlds is important – anywhere else and avalon won't find them.

## Launching the Editor

Now that we have some worlds, lets launch the editor and take a look.
In a dedicated terminal, run:

```sh
# Launch our custom build of the godot editor
python -m  avalon.for_humans launch_editor
```

In the bottom right file navigator, scroll to the bottom and select
`worlds/practice__eat__10001__0/main.tscn`:

## Configuring the Editor

Before we can deploy to the oculus, we'll need to install the openxr plugin and configure android APK export.
Luckily, `avalon.for_humans` has utilities to help with all of that.

### Installing OpenXR

1. Stop the editor. We do this because this command edits project settings, which can confuse godot.
1. Run `python -m avalon.for_humans install_openxr_plugin`,
   which will download, position, and enable the [latest release from GodotVR/godot_openxr](https://github.com/GodotVR/godot_openxr/releases/tag/1.3.0)
2. Relaunch the editor with `python -m  avalon.for_humans launch_editor`
2. Open **Project > Project Settings > Plugins** and everify the plugin is enabled:
   ![enable_openxr_plugin](https://user-images.githubusercontent.com/8343799/197244046-f313a5d2-a13a-4d66-941d-782e2820f11d.png)

If the plugin causes any issues, it can be removed with `remove_openxr_plugin`,
and the project path can be retrieved with `python -m avalon.for_humans print_godot_project_path` for hands-on troubleshooting.

### Configuring APK export

First, we need to go into **Editor > Export > Android** and point Godot to the android sdk path obtained at installation:
![android sdk setting](https://user-images.githubusercontent.com/8343799/197281970-877cfaf5-7363-4439-83a2-7dbbe88e106c.png)

Now we need to generate a keystore, and create an export template.

#### Generate a `debug.keystore`

If you already have an `~/android/debug.keystore`, you can skip this step and update the later arguments to `setup_android_export_presets` accordingly.
```sh
ANDROID_CONFIG_DIR=$(realpath ~/.android/)
KEYSTORE_USER=androiddebugkey
KEYSTORE_PASSWORD=android
mkdir -p $ANDROID_CONFIG_DIR
pushd $ANDROID_CONFIG_DIR
# Adapted from from godot's guide
keytool -keyalg RSA -genkeypair -alias $KEYSTORE_USER -keypass android \
  -keystore debug.keystore -storepass $KEYSTORE_PASSWORD \
  -dname "CN=Android Debug,O=Android,C=US" \
  -validity 9999 -deststoretype pkcs12
popd
echo "keystore saved to $ANDROID_CONFIG_DIR/debug.keystore"
```

#### Setting up export template

This can be done via the Godot UI, but we strongly recommend using avalon's template as a starting point.
In fact, we have a utility just for that!

```sh
python -m avalon.for_humans setup_android_export_presets \
  --keystore_path=$ANDROID_CONFIG_DIR/debug.keystore \
  --keystore_user=$KEYSTORE_USER \
  --keystore_password=$KEYSTORE_PASSWORD
#> Downloading Debug APK template
#> Downloading Release APK template
#> Configuring export_presets.cfg
#> Android export configured. You will need to restart the editor to see changes.
```

This configures godot's `export_presets.cfg` and installs necessary template APKs.

To verify the setup:
1.  Restart the editor
2.  Go to **Project > Export**, and click on the "Android (Runnable)" item:
    ![android runnable export menu](https://user-images.githubusercontent.com/8343799/197857228-f9d8ec9c-078b-4260-86de-99cc3a7649d8.png)
3. If there are no **red error messages** at the bottom of the screen,
   the setup was successful.
4. If not, either the above script failed, or the android sdk is missing some components.

> Note: `avalon.for_humans` tries to automate much of this process,
> but if you run into issues and need to troubleshoot manually, you might want to look at
> [Godot's "Exporting for Android" guide](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_android.html)
> and the [Android docs on adding a preset](https://developer.android.com/games/engines/godot/godot-export#add-preset).

## Connecting the Oculus

Plug in and turn on your oculus and run `adb devices`. You'll likely see a message like:
> List of devices attached
> 1ABCDE12345678  no permissions (missing udev rules? user is in the plugdev group); see [http://developer.android.com/tools/device.html]

If so, you'll need to put on the headset and [Allow USB debugging](https://developer.oculus.com/documentation/native/android/mobile-device-setup/)
(Unfortunately you'll need to do this pretty much every time you plug in the headset).

Once properly setup (this can take a few tries), `adb devices` should show something like `$DEVICE_ID  device`,

Once the device is sucessfully connected, you should see a "Debug on Android" button appear in the upper corner of the editor:

![android_button](https://user-images.githubusercontent.com/8343799/197067515-b3bbc5f9-2833-49ef-a206-0837505eff8d.png)

Lastly, enable **Debug > Deploy with Remote Debugging** and click the above button.
Avalon should launch on your Oculus.

To troubleshoot any errors, or just to see the debug output, use `adb logcat 'godot:D' '*:S'`

## Debugging over wifi

Once you have USB debugging working, you can [setup debugging over wifi](https://jonassandstedt.se/blog/connect-adb-wirelessly-to-a-oculus-quest/) rather easily:

```sh
# Pull address
_oculus_wlan_settings=`adb shell ip addr show wlan0`
OCULUS_IP_ADDRESS=`echo -e "$_oculus_wlan_settings" | grep 'inet ' | sed 's/inet //;s/\/.*//' | xargs`
echo OCULUS_IP_ADDRESS=$OCULUS_IP_ADDRESS
# restart adb with known port
adb tcpip 5555
adb connect $OCULUS_IP_ADDRESS:5555
adb devices
#> List of devices attached
#> 1ABCDE12345678  device
#> 12.3.4.567:5555 device
echo "
# Oculus connection command:
alias reconnect_oculus=\"adb connect $OCULUS_IP_ADDRESS:5555\"
"
```

Now disconnect USB and go back to the godot editor. The Android button should still be there and usable as before.
The oculus will become disconnected between sessions, but the `reconnect_oculus` command above should still work,
as long as it's on and the IP hasn't changed.
