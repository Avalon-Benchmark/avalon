extends Reference

class_name CONST

const READY_LOG_SIGNAL := "GODATA!"

const FIXED_FPS := 10

const RGB_FEATURE := "rgb"
const RGBD_FEATURE := "rgbd"
const DEPTH_FEATURE := "depth"
const LABEL_FEATURE := "label"
const DATASET_ID_FEATURE := "dataset_id"
const VIDEO_ID_FEATURE := "video_id"
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

# TODO we can eventually remove this
# for playing / recording human actions
const HUMAN_INPUT_MESSAGE := 9

const VR_ACTION_SPACE = "VR_ACTION_SPACE"
const MOUSE_KEYBOARD_ACTION_SPACE = "MOUSE_KEYBOARD_ACTION_SPACE"

# TODO: remove this at some point
const LEFT_HAND = "Left"
const RIGHT_HAND = "Right"

const FAKE_TYPE_IMAGE = -1
