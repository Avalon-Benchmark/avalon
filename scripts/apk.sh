#!/bin/bash
set -e
set -u


export OCULUS_BASE=/opt/oculus
export EDITOR_BASE=/opt/oculus/godot
export ANDROID_SDK=/opt/oculus/android-sdk


if [[ ! -d "$OCULUS_BASE" ]]
then
mkdir "$OCULUS_BASE"
pushd "$OCULUS_BASE"
sudo tee /etc/apt/sources.list.d/monado.list << EOF
deb https://ppa.launchpadcontent.net/monado-xr/monado/ubuntu focal main
deb-src https://ppa.launchpadcontent.net/monado-xr/monado/ubuntu focal main
EOF
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 5A6166F945641E7AC11907E00D8B4D5E07191FA9
sudo apt-get update
sudo apt-get install -y openjdk-11-jdk-headless
popd
fi


if [[ ! -f "$OCULUS_BASE/android.keystore" ]]
then
pushd "$OCULUS_BASE"
ARGS=(
    -genkeypair
    -keystore   android.keystore
    -storepass  android
    -storetype  pkcs12
    -validity   9990
    -keyalg     RSA
)
/usr/bin/keytool "${ARGS[@]}" -alias godot -dname CN=Godot,O=Android,C=US
popd
fi


if [[ ! -d "$ANDROID_SDK" ]]
then
mkdir "$ANDROID_SDK"
pushd "$ANDROID_SDK"
wget  'https://dl.google.com/android/repository/commandlinetools-linux-8092744_latest.zip'
unzip *.zip | wc -l
rm -f *.zip
mkdir ./latest
mv ./cmdline-tools/* ./latest
mv ./latest ./cmdline-tools
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" --licenses | wc -l
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" "build-tools;30.0.3"
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" "cmake;3.10.2.4988404"
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" "ndk;21.4.7075529"
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" "platforms;android-29"
popd
fi


if [[ ! -d "$EDITOR_BASE" ]]
then
mkdir "$EDITOR_BASE"
pushd "$EDITOR_BASE"
wget  'https://downloads.tuxfamily.org/godotengine/3.4.4/Godot_v3.4.4-stable_linux_headless.64.zip'
unzip *.zip | wc -l
rm -f *.zip
mv ./Godot_*64 ./godot
touch ._sc_
mkdir ./editor_data
tee ./editor_data/editor_settings-3.tres << EOF
[gd_resource type="EditorSettings" format=2]
[resource]
export/android/android_sdk_path = "$ANDROID_SDK"
EOF
wget  'https://mmap.monster/godot/templates/android_debug.apk'
wget  'https://mmap.monster/godot/templates/android_release.apk'
wget  'https://github.com/GodotVR/godot_openxr/releases/download/1.3.0/godot-openxr.zip'
unzip *.zip | wc -l
rm -f *.zip
popd
fi

if [[ ! -f './project.godot' ]]
then
echo "Project settings not found: $(pwd)/project.godot"
exit 1
fi

if [[ ! -d './addons/godot-openxr' ]]
then
mkdir -p './addons'
rsync -av "$EDITOR_BASE/godot_openxr_1.3.0/addons/godot-openxr" './addons/' --exclude assets --exclude scenes
fi

if [[ ! -f './export_presets.cfg' ]]
then
echo
tee ./export_presets.cfg <<'EOF'
[preset.0]
name="Android"
platform="Android"
runnable=true
custom_features=""
export_filter="all_resources"
include_filter="android/*,worlds/*/*.json"
exclude_filter=""
export_path=""
script_export_mode=0
script_encryption_key=""
[preset.0.options]
custom_template/debug="/opt/oculus/godot/android_debug.apk"
custom_template/release="/opt/oculus/godot/android_release.apk"
custom_template/use_custom_build=false
custom_template/export_format=0
architectures/armeabi-v7a=false
architectures/arm64-v8a=true
architectures/x86=false
architectures/x86_64=false
keystore/debug="/opt/oculus/android.keystore"
keystore/debug_user="godot"
keystore/debug_password="android"
keystore/release="/opt/oculus/android.keystore"
keystore/release_user="godot"
keystore/release_password="android"
one_click_deploy/clear_previous_install=false
version/code=1
version/name="1.0"
version/min_sdk=19
version/target_sdk=30
package/unique_name="org.godotengine.avalon"
package/name="avalon"
package/signed=true
package/classify_as_game=true
package/retain_data_on_uninstall=false
package/exclude_from_recents=false
launcher_icons/main_192x192=""
launcher_icons/adaptive_foreground_432x432=""
launcher_icons/adaptive_background_432x432=""
graphics/32_bits_framebuffer=true
graphics/opengl_debug=false
xr_features/xr_mode=2
xr_features/hand_tracking=1
xr_features/hand_tracking_frequency=0
xr_features/passthrough=0
screen/immersive_mode=true
screen/support_small=true
screen/support_normal=true
screen/support_large=true
screen/support_xlarge=true
user_data_backup/allow=false
command_line/extra_args=""
apk_expansion/enable=false
apk_expansion/SALT=""
apk_expansion/public_key=""
permissions/access_network_state=true
permissions/access_wifi_state=true
permissions/internet=true
permissions/read_external_storage=true
permissions/write_external_storage=true
EOF
echo
fi
