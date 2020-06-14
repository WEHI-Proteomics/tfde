import SQLite

let CONVERTED_DATABASE_NAME = "/data/exp-dwm-test-run-190719_Hela_Ecoli_1to1_01-converted.sqlite"
let PUBLISHED_FRAMES_DIR = "/data/published-frames"

// Configure a SQLite database
let sqlite = try SQLiteDatabase(storage: .file(path: CONVERTED_DATABASE_NAME))
