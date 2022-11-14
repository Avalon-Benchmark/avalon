extends Reference

class_name CONST

const BRIDGE_LOG_SIGNAL := "Establishing Bridge"
const READY_LOG_SIGNAL := "GODATA!"

const FIXED_FPS := 10

const RGBD_FEATURE := "rgbd"
const ISOMETRIC_RGBD_FEATURE := "isometric_rgbd"
const TOP_DOWN_RGBD_FEATURE := "top_down_rgbd"
const EPISODE_ID_FEATURE := "episode_id"
const FRAME_ID_FEATURE := "frame_id"
const EVENT_HAPPENED_FEATURE := "event_happened"
const ACTION_FEATURE := "action"

const RESET_MESSAGE := 0
const SEED_MESSAGE := 1
const RENDER_MESSAGE := 2
const ACTION_MESSAGE := 3
const SELECT_FEATURES_MESSAGE := 4
const QUERY_AVAILABLE_FEATURES_MESSAGE := 5
const CLOSE_MESSAGE := 6
const DEBUG_CAMERA_ACTION_MESSAGE := 7
const HUMAN_INPUT_MESSAGE := 9

const SAVE_SNAPSHOT_MESSAGE := 10
const LOAD_SNAPSHOT_MESSAGE := 11

const VR_ACTION_SPACE = "VR_ACTION_SPACE"
const MOUSE_KEYBOARD_ACTION_SPACE = "MOUSE_KEYBOARD_ACTION_SPACE"

const LEFT_HAND = "Left"
const RIGHT_HAND = "Right"

const FAKE_TYPE_IMAGE = -1

const SNAPSHOT_JSON := "snapshot_context.json"
const SNAPSHOT_SUBPATH := "snapshots"

const SCENE_ROOT_NODE_PATH := "/root/scene_root"
const WORLD_NODE_PATH := SCENE_ROOT_NODE_PATH + "/world"

const DYNAMIC_TRACKER_NODE_NAME := "dynamic_tracker"
const DYNAMIC_TRACKER_NODE_PATH := WORLD_NODE_PATH + "/Avalon/dynamic_tracker"
